from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meteogram2zarr import (  # noqa: E402
    build_meteogram_zarr,
    discover_meteogram_files,
    get_station_coords_from_cfg,
    get_variable_names,
    provenance_attrs,
)


pytest.importorskip("dask")
pytest.importorskip("zarr")


def _write_station_file(path: Path, values: float, n_time: int = 3) -> None:
    ds = xr.Dataset(
        data_vars={
            "T": ("time", np.full(n_time, values, dtype="f4")),
            "Q": (("time", "bins"), np.full((n_time, 2), values + 1, dtype="f4")),
        },
        coords={"time": np.arange(n_time), "bins": np.arange(2)},
    )
    ds.to_netcdf(path, engine="scipy")


def _write_meta(path: Path) -> None:
    payload = {
        "20260304110254": {
            "INPUT_DIA": {
                "diactl": {
                    "stationlist_tot": [
                        0,
                        0,
                        47.1,
                        7.8,
                        "01_station",
                        0,
                        0,
                        47.2,
                        7.9,
                        "02_station",
                        0,
                        0,
                        0.0,
                        0.0,
                        "SE_station",
                    ]
                }
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_and_station_coords(tmp_path: Path) -> None:
    exp = "20260304110254"
    _write_station_file(tmp_path / f"M_01_01_{exp}.nc", 1)
    _write_station_file(tmp_path / f"M_02_01_{exp}.nc", 2)
    _write_station_file(tmp_path / f"M_SE_01_{exp}.nc", 3)
    meta = tmp_path / "run.json"
    _write_meta(meta)

    files = discover_meteogram_files(tmp_path)
    coords = get_station_coords_from_cfg(meta)

    assert list(files) == [exp]
    assert len(files[exp]) == 2
    assert coords == {"01": (47.1, 7.8), "02": (47.2, 7.9)}


def test_variable_names_and_zarr_write(tmp_path: Path) -> None:
    exp = "20260304110254"
    for station, value in [("01", 1), ("02", 2)]:
        _write_station_file(tmp_path / f"M_{station}_01_{exp}.nc", value)
    meta = tmp_path / "run.json"
    _write_meta(meta)
    file_dict = discover_meteogram_files(tmp_path)
    variables = get_variable_names(file_dict[exp][0], nc_engine="scipy")
    out = tmp_path / "meteogram.zarr"

    build_meteogram_zarr(
        file_dict,
        out,
        variables=variables,
        max_time=3,
        meta_file=str(meta),
        nc_engine="scipy",
        global_attrs=provenance_attrs(input_files=[str(meta)]),
    )
    ds = xr.open_zarr(out)

    assert set(variables) == {"T", "Q"}
    assert ds.sizes["expname"] == 1
    assert ds.sizes["station"] == 2
    assert ds.sizes["time"] == 3
    assert np.allclose(ds["station_lat"].values, [47.1, 47.2])
    assert float(ds["T"].sel(station=1).isel(expname=0, time=0)) == 1.0
