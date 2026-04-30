"""Command line interface for meteogram2zarr."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from time import time as time_now

from .core import (
    build_meteogram_zarr,
    discover_meteogram_files,
    get_max_timesteps,
    get_station_coords_from_cfg,
    get_variable_names,
    provenance_attrs,
)


def _parse_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COSMO-SPECS meteogram NetCDF files into one Zarr store."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing M_??_??_*.nc files.")
    parser.add_argument("--output", required=True, help="Output Zarr store path.")
    parser.add_argument("--meta-json", help="Optional COSMO-SPECS run JSON for station lat/lon metadata.")
    parser.add_argument("--variables", help="Comma-separated variables to keep. Default: all data variables.")
    parser.add_argument("--experiments", help="Comma-separated experiment suffixes to process.")
    parser.add_argument("--debug", action="store_true", help="Process first two discovered experiments only.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output Zarr store.")
    parser.add_argument("--max-time", type=int, help="Override max time steps and skip ncdump scan.")
    parser.add_argument("--max-height-level", type=int, default=20, help="Vertical levels to keep from top.")
    parser.add_argument("--target-time-chunk", type=int, default=1024, help="Output time chunk.")
    parser.add_argument("--target-station-chunk", type=int, default=-1, help="Output station chunk, -1 for full.")
    parser.add_argument("--target-bins-chunk", type=int, default=-1, help="Output bins chunk, -1 for full.")
    parser.add_argument("--compression-level", type=int, default=3, help="Blosc zstd compression level.")
    parser.add_argument("--profile-io", action="store_true", help="Print per-experiment open timing.")
    parser.add_argument("--open-fast", action="store_true", help="Disable default indexes/cache where xarray supports it.")
    parser.add_argument(
        "--nc-engine",
        choices=("auto", "h5netcdf", "netcdf4", "scipy"),
        default="auto",
        help="xarray NetCDF backend.",
    )
    parser.add_argument("--slurm", action="store_true", help="Use levante-slurm-utils if installed.")
    parser.add_argument("--slurm-account", default=None, help="Slurm account for --slurm.")
    parser.add_argument("--slurm-partition", default="compute", help="Slurm partition for --slurm.")
    return parser.parse_args(argv)


def _select_experiments(file_dict: dict[str, list[str]], selected: list[str] | None, debug: bool) -> dict[str, list[str]]:
    if selected:
        missing = [exp for exp in selected if exp not in file_dict]
        if missing:
            raise ValueError("Requested experiments not found: " + ", ".join(missing))
        return {exp: file_dict[exp] for exp in selected}
    if debug:
        return dict(list(file_dict.items())[:2])
    return file_dict


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists, use --overwrite to rebuild: {output}")

    file_dict_all = discover_meteogram_files(input_dir, dbg=False)
    if not file_dict_all:
        raise SystemExit(f"No meteogram files found in {input_dir}")
    max_time = args.max_time if args.max_time is not None else get_max_timesteps(file_dict_all)
    file_dict = _select_experiments(file_dict_all, _parse_list(args.experiments), args.debug)

    sample_file = next(iter(file_dict.values()))[0]
    variables = _parse_list(args.variables) or get_variable_names(sample_file, nc_engine=args.nc_engine)
    station_coords = get_station_coords_from_cfg(args.meta_json) if args.meta_json else None
    input_files = [str(Path(path)) for files in file_dict.values() for path in files[:1]]
    if args.meta_json:
        input_files.insert(0, str(Path(args.meta_json)))
    attrs = provenance_attrs(input_files=input_files)
    if len(input_files) > 50:
        attrs["input_files"] = f"{len(input_files)} files"

    cluster = client = None
    if args.slurm and not args.debug:
        try:
            from levante_slurm_utils import allocate_resources, calculate_optimal_scaling
        except Exception as exc:  # pragma: no cover - optional integration
            raise SystemExit("Install levante-slurm-utils to use --slurm.") from exc
        n_nodes, n_cpu, mem_gb, scale_workers, walltime = calculate_optimal_scaling(
            max_time,
            len(file_dict),
            len(station_coords or {}),
        )
        cluster, client = allocate_resources(
            n_cpu=n_cpu,
            n_jobs=n_nodes,
            m=int(mem_gb),
            walltime=walltime,
            part=args.slurm_partition,
            account=args.slurm_account,
        )
        cluster.scale(scale_workers)

    t0 = time_now()
    try:
        build_meteogram_zarr(
            file_dict,
            output,
            variables=variables,
            max_time=max_time,
            max_height_level=args.max_height_level,
            station_coords=station_coords,
            meta_file=args.meta_json,
            target_time_chunk=args.target_time_chunk,
            target_station_chunk=args.target_station_chunk,
            target_bins_chunk=args.target_bins_chunk,
            compression_level=args.compression_level,
            global_attrs=attrs,
            profile_io=args.profile_io,
            open_fast=args.open_fast,
            nc_engine=args.nc_engine,
        )
    except Exception:
        if output.exists() and args.overwrite:
            shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        if cluster is not None:
            cluster.close()
        if client is not None:
            client.close()

    print(f"Done in {time_now() - t0:.1f} s: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
