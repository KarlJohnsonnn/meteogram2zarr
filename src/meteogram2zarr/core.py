"""Fast COSMO-SPECS meteogram NetCDF-to-Zarr conversion."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


def detect_nc_engine(path: str | os.PathLike[str]) -> str:
    """Choose an xarray backend from NetCDF magic bytes."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic[:4] == b"\x89HDF":
        return "h5netcdf"
    if magic[:3] == b"CDF":
        return "netcdf4"
    raise ValueError(f"Cannot detect NetCDF engine from file magic: {path}")


def _resolve_nc_engine(path: str | os.PathLike[str], nc_engine: Optional[str]) -> str:
    if nc_engine is None or str(nc_engine).lower() == "auto":
        engine = detect_nc_engine(path)
        print(f"Auto-detected NetCDF engine for {Path(path).name}: {engine}")
        return engine
    engine = str(nc_engine).lower()
    if engine not in ("h5netcdf", "netcdf4", "scipy"):
        raise ValueError(
            "nc_engine must be 'auto', 'h5netcdf', 'netcdf4', or 'scipy', "
            f"got {nc_engine!r}"
        )
    return engine


def discover_meteogram_files(
    data_dir: str | os.PathLike[str],
    *,
    exclude_stations: Sequence[str] = ("SE", "OB"),
    dbg: bool = False,
) -> Dict[str, List[str]]:
    """Return ``{experiment_suffix: [station files]}`` from a meteogram directory."""
    data_dir = str(data_dir)
    all_files = sorted(glob.glob(f"{data_dir}/M_??_??_??????????????.nc"))
    expnames = sorted({Path(path).name.rsplit("_", 1)[-1].split(".")[0] for path in all_files})

    def station_id_of(path: str) -> str:
        return Path(path).name.split("_")[1]

    files_per_exp: Dict[str, List[str]] = {}
    for exp in expnames:
        matched = [
            path
            for path in glob.glob(f"{data_dir}/M_??_??_{exp}.nc")
            if station_id_of(path) not in exclude_stations
        ]
        files_per_exp[exp] = sorted(matched)

    if dbg:
        files_per_exp = dict(list(files_per_exp.items())[:2])
        for exp, files in files_per_exp.items():
            print(f"DBG discover: {exp} ({len(files)} stations)")

    return files_per_exp


def _ncdump_time_size(path: str) -> int:
    result = subprocess.run(
        ["ncdump", "-h", path],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return 0
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if "time = " not in lowered or ";" not in line:
            continue
        if "; // (" in line:
            size_str = line.split("; // (", 1)[1].replace(" currently)", "")
            return int(size_str)
        after_equals = line.split("=", 1)[1].split(";", 1)[0].strip()
        if after_equals.isdigit():
            return int(after_equals)
    return 0


def get_max_timesteps(file_dict: Dict[str, List[str]]) -> int:
    """Return maximum time dimension size using ``ncdump -h``."""
    if shutil.which("ncdump") is None:
        raise RuntimeError("ncdump not found on PATH. Install netcdf-bin/netcdf-c utilities or pass --max-time.")

    all_files = [path for files in file_dict.values() for path in files]
    if not all_files:
        raise ValueError("No meteogram files available for time scan.")

    print(f"Checking time steps in {len(all_files)} files (ncdump)...")
    sizes = [_ncdump_time_size(path) for path in all_files]
    max_time = max(sizes) if sizes else 0
    if max_time <= 0:
        raise RuntimeError("Could not read any positive time dimension from ncdump output.")
    print(f"Max time steps: {max_time}")
    return max_time


def get_variable_names(sample_file: str | os.PathLike[str], *, nc_engine: Optional[str] = "auto") -> List[str]:
    """Read data variable names from one NetCDF file."""
    engine = _resolve_nc_engine(sample_file, nc_engine)
    with xr.open_dataset(sample_file, engine=engine) as ds:
        return [str(name) for name in ds.data_vars.keys()]


def get_station_coords_from_cfg(meta_file: str | os.PathLike[str]) -> Dict[str, Tuple[float, float]]:
    """Read station latitude/longitude from a COSMO-SPECS run JSON."""
    path = Path(meta_file)
    if not path.exists():
        raise FileNotFoundError(f"Run metadata not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg_dict = json.load(fh)
    expname = next(iter(cfg_dict))
    stationlist = cfg_dict[expname]["INPUT_DIA"]["diactl"]["stationlist_tot"]
    if not stationlist:
        raise ValueError("Station list is empty.")
    arr = np.asarray(stationlist, dtype=object).reshape(-1, 5)
    out: Dict[str, Tuple[float, float]] = {}
    for row in arr:
        station_id = str(row[-1]).split("_")[0]
        if station_id in ("SE", "OB"):
            continue
        out[station_id] = (float(row[2]), float(row[3]))
    return out


def provenance_attrs(
    *,
    title: str = "COSMO-SPECS meteogram Zarr",
    summary: str = "Meteogram variables concatenated along expname.",
    input_files: Optional[List[str]] = None,
    processing_level: str = "LV2",
) -> Dict[str, Any]:
    """Build small Zarr-safe global attrs without requiring a git repo."""
    attrs: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processing_level": processing_level,
        "Conventions": "CF-1.8",
    }
    if title:
        attrs["title"] = title
    if summary:
        attrs["summary"] = summary
    if input_files is not None:
        attrs["input_files"] = input_files
    return attrs


def _build_target_time_values(ds_time: np.ndarray, max_time: int) -> np.ndarray:
    if max_time <= 0 or ds_time.size >= max_time:
        return ds_time
    if ds_time.size < 2:
        raise ValueError("Need at least two time steps to infer missing time spacing.")
    missing = max_time - ds_time.size
    dt = ds_time[1] - ds_time[0]
    tail = np.asarray([ds_time[-1] + (i + 1) * dt for i in range(missing)], dtype=ds_time.dtype)
    return np.concatenate([ds_time, tail])


def _align_station_time(ds: xr.Dataset, target_times: np.ndarray) -> xr.Dataset:
    if "time" not in ds.sizes:
        return ds
    cur = ds.time.values
    if cur.shape == target_times.shape and np.array_equal(cur, target_times):
        return ds
    if "time" not in ds.xindexes and "time" in ds.coords:
        ds = ds.set_xindex("time")
    return ds.reindex(time=target_times, method="pad", fill_value=np.nan)


def _preprocess_station_heights(ds: xr.Dataset, max_height_level: int = 20) -> xr.Dataset:
    if "HMLd" in ds.sizes:
        n_levels = ds.sizes["HMLd"]
        ds = ds.swap_dims({"HMLd": "height_level"})
        ds = ds.assign_coords(height_level=np.arange(n_levels))
        ds["HMLd"] = ds.HMLd

    if "HHLd" in ds.sizes:
        n_levels = ds.sizes["HHLd"]
        ds = ds.swap_dims({"HHLd": "height_level2"})
        ds = ds.assign_coords(height_level2=np.arange(n_levels))
        ds["HHLd"] = ds.HHLd

    if "height_level" in ds.sizes:
        ds = ds.isel(height_level=slice(-max_height_level, None))
    if "height_level2" in ds.sizes:
        ds = ds.isel(height_level2=slice(-max_height_level - 1, None))
    return ds


def _open_experiment(
    sorted_files: List[str],
    variables: List[str],
    max_time: int,
    max_height_level: int,
    chunks: Dict[str, int],
    *,
    profile_io: bool = False,
    open_fast: bool = False,
    nc_engine: Optional[str] = None,
) -> Tuple[xr.Dataset, np.ndarray]:
    def station_id_of(path: str) -> str:
        return Path(path).name.split("_")[1]

    station_ids = np.asarray([station_id_of(path) for path in sorted_files], dtype="i4")
    engine = _resolve_nc_engine(sorted_files[0], nc_engine)
    open_kw: Dict[str, Any] = {"engine": engine, "chunks": chunks}
    if open_fast:
        open_kw["create_default_indexes"] = False
        open_kw["cache"] = False

    t_open = t_sel = t_align = t_heights = 0.0
    t_exp0 = perf_counter()
    target_times: Optional[np.ndarray] = None
    datasets: List[xr.Dataset] = []
    for path in sorted_files:
        t0 = perf_counter()
        ds = xr.open_dataset(path, **open_kw)
        t_open += perf_counter() - t0

        t0 = perf_counter()
        missing = [name for name in variables if name not in ds]
        if missing:
            raise KeyError(f"{path} is missing variables: {', '.join(missing)}")
        ds = ds[variables]
        t_sel += perf_counter() - t0

        t0 = perf_counter()
        if target_times is None:
            target_times = _build_target_time_values(ds.time.values, max_time)
        ds = _align_station_time(ds, target_times)
        t_align += perf_counter() - t0

        t0 = perf_counter()
        ds = _preprocess_station_heights(ds, max_height_level)
        t_heights += perf_counter() - t0
        datasets.append(ds)

    t0 = perf_counter()
    ds_exp = xr.concat(datasets, dim="station", coords="minimal", compat="override")
    t_concat = perf_counter() - t0
    ds_exp = ds_exp.assign_coords(station=station_ids)

    if profile_io:
        n_files = max(len(sorted_files), 1)
        print(
            f"    [io] open={t_open:.2f}s sel={t_sel:.2f}s align={t_align:.2f}s "
            f"heights={t_heights:.2f}s concat={t_concat:.2f}s "
            f"total={perf_counter() - t_exp0:.2f}s ({n_files} files, engine={engine})"
        )
    return ds_exp, station_ids


def _coerce_station_id(value: Any) -> int | str:
    try:
        return int(value)
    except Exception:
        return str(value)


def _station_id_array(values: Sequence[Any]) -> np.ndarray:
    vals = [_coerce_station_id(value) for value in values]
    if vals and all(isinstance(value, int) for value in vals):
        return np.asarray(vals, dtype="i4")
    return np.asarray([str(value) for value in vals], dtype="U")


def _compute_bin_coords(
    n_bins: int = 66,
    n_max: float = 2.0,
    r_min: float = 1e-9,
    rhow: float = 1e3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fact = rhow * 4.0 / 3.0 * np.pi
    m0w = fact * r_min**3
    j0w = (n_max - 1.0) / np.log(2.0)
    m_edges = m0w * np.exp(np.arange(n_bins + 1) / j0w)
    r_edges = np.cbrt(m_edges / fact)
    m_centers = np.sqrt(m_edges[1:] * m_edges[:-1])
    r_centers = np.cbrt(m_centers / fact)
    return m_edges, m_centers, r_edges, r_centers


def add_coords_and_metadata(
    ds: xr.Dataset,
    *,
    meta_file: Optional[str] = None,
    station_coords: Optional[Dict[str, Tuple[float, float]]] = None,
) -> xr.Dataset:
    """Add station and COSMO-SPECS bin coordinates."""
    if station_coords is None and meta_file is not None:
        station_coords = get_station_coords_from_cfg(meta_file)

    if station_coords is not None and "station" in ds.sizes:
        station_lat = []
        station_lon = []
        for sid in ds.station.values:
            sid_text = str(sid)
            lat_lon = station_coords.get(sid_text)
            if lat_lon is None and sid_text.isdigit():
                lat_lon = station_coords.get(sid_text.zfill(2))
            if lat_lon is None:
                station_lat.append(np.nan)
                station_lon.append(np.nan)
            else:
                station_lat.append(float(lat_lon[0]))
                station_lon.append(float(lat_lon[1]))
        ds = ds.assign_coords(
            station_lat=xr.DataArray(np.asarray(station_lat, dtype="f8"), dims="station"),
            station_lon=xr.DataArray(np.asarray(station_lon, dtype="f8"), dims="station"),
        )

    if "bins" in ds.sizes:
        n_bins = ds.sizes["bins"]
        m_edges, m_centers, r_edges, r_centers = _compute_bin_coords(n_bins=n_bins)
        ds = ds.assign_coords(
            bins_boundaries=xr.DataArray(np.arange(n_bins + 1), dims="bins_boundaries"),
            mass_centers=xr.DataArray(m_centers, dims="bins", attrs={"units": "kg"}),
            mass_boundaries=xr.DataArray(m_edges, dims="bins_boundaries", attrs={"units": "kg"}),
            radius_centers=xr.DataArray(r_centers, dims="bins", attrs={"units": "m"}),
            radius_boundaries=xr.DataArray(r_edges, dims="bins_boundaries", attrs={"units": "m"}),
        )
    return ds


def _zarr_encoding(ds: xr.Dataset, compression_level: int = 3) -> dict[str, dict[str, Any]]:
    import numcodecs

    compressor = numcodecs.Blosc(
        cname="zstd",
        clevel=compression_level,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )
    return {var: {"compressor": compressor} for var in ds.data_vars}


def build_meteogram_zarr(
    file_dict: Dict[str, List[str]],
    zarr_path: str | os.PathLike[str],
    *,
    variables: List[str],
    max_time: int,
    max_height_level: int = 20,
    station_coords: Optional[Dict[str, Tuple[float, float]]] = None,
    meta_file: Optional[str] = None,
    target_time_chunk: int = 1024,
    target_station_chunk: int = -1,
    target_bins_chunk: int = -1,
    compression_level: int = 3,
    global_attrs: Optional[Dict[str, Any]] = None,
    profile_io: bool = False,
    open_fast: bool = False,
    nc_engine: Optional[str] = None,
) -> str:
    """Build one Zarr v2 store from per-experiment meteogram NetCDF files."""
    from dask.base import compute
    from dask.diagnostics.progress import ProgressBar

    if station_coords is None and meta_file is not None:
        station_coords = get_station_coords_from_cfg(meta_file)

    expnames = list(file_dict.keys())
    if not expnames:
        raise ValueError("No experiments found.")

    target_chunks: Dict[str, int] = {
        "time": min(target_time_chunk, max_time) if max_time > 0 else target_time_chunk,
        "height_level": -1,
        "height_level2": -1,
        "bins": target_bins_chunk,
    }
    exp_datasets: List[xr.Dataset] = []
    station_ids_seen: List[Any] = []

    for i, exp in enumerate(expnames):
        files = sorted(file_dict[exp])
        if not files:
            raise ValueError(f"Experiment {exp} has no station files.")
        print(f"  Opening [{i + 1}/{len(expnames)}] {exp} ({len(files)} stations)")
        ds_exp, sids = _open_experiment(
            files,
            variables,
            max_time,
            max_height_level,
            target_chunks,
            profile_io=profile_io,
            open_fast=open_fast,
            nc_engine=nc_engine,
        )
        station_ids_seen.extend([_coerce_station_id(sid) for sid in sids.tolist()])
        exp_datasets.append(ds_exp)

    if station_coords is not None:
        station_ids = _station_id_array(sorted({_coerce_station_id(key) for key in station_coords.keys()}))
    else:
        station_ids = _station_id_array(sorted(set(station_ids_seen)))

    n_stations = len(station_ids)
    target_chunks["station"] = -1 if target_station_chunk == -1 else min(target_station_chunk, n_stations)
    sample_sizes = dict(exp_datasets[0].sizes)
    sample_sizes["station"] = n_stations
    sample_sizes["expname"] = len(expnames)
    active_chunks: Dict[str, int] = {}
    for dim, value in target_chunks.items():
        if dim not in sample_sizes:
            continue
        active_chunks[dim] = sample_sizes[dim] if value == -1 else min(value, sample_sizes[dim])
    active_chunks["expname"] = 1

    write_chunks = {key: value for key, value in active_chunks.items() if key != "expname"}
    exp_datasets = [ds_exp.reindex(station=station_ids).chunk(write_chunks) for ds_exp in exp_datasets]

    experiments_da = xr.DataArray(np.asarray(expnames, dtype="U"), dims="expname")
    station_da = xr.DataArray(station_ids, dims="station", attrs={"long_name": "Station ID"})
    ds_template = exp_datasets[0].expand_dims(expname=experiments_da)
    ds_template = ds_template.assign_coords(expname=experiments_da, station=station_da)
    if "dim" in ds_template.coords:
        ds_template = ds_template.drop_vars("dim")
    ds_template = add_coords_and_metadata(
        ds_template,
        meta_file=meta_file,
        station_coords=station_coords,
    ).chunk(active_chunks)

    if global_attrs:
        ds_template.attrs.update(global_attrs)

    zarr_path = str(zarr_path)
    if os.path.exists(zarr_path):
        print(f"Removing existing Zarr store: {zarr_path}")
        shutil.rmtree(zarr_path)

    print(f"Initializing Zarr store: {zarr_path}")
    print(f"  shape : {dict(ds_template.sizes)}")
    print(f"  chunks: {dict(ds_template.chunks)}")
    ds_template.to_zarr(
        zarr_path,
        mode="w",
        compute=False,
        encoding=_zarr_encoding(ds_template, compression_level),
        zarr_format=2,
    )
    ds_template.coords.to_dataset().to_zarr(zarr_path, mode="a", zarr_format=2)

    print("Writing experiment regions...")
    for i, (exp, ds_exp) in enumerate(zip(expnames, exp_datasets)):
        print(f"  Writing [{i + 1}/{len(expnames)}] {exp}")
        ds_region = xr.Dataset(
            data_vars={name: da.expand_dims(expname=[exp]) for name, da in ds_exp.data_vars.items()}
        ).chunk(active_chunks)
        drop_names = [str(name) for name, da in ds_region.variables.items() if "expname" not in da.dims]
        ds_region = ds_region.drop_vars(drop_names)
        delayed = ds_region.to_zarr(
            zarr_path,
            mode="r+",
            region={"expname": slice(i, i + 1)},
            compute=False,
            zarr_format=2,
        )
        with ProgressBar(minimum=2):
            compute(delayed)

    print(f"Zarr store complete: {zarr_path}")
    return zarr_path
