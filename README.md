# Meteogram2Zarr

Standalone converter for COSMO-SPECS meteogram NetCDF files (`M_??_??_*.nc`)
into one Zarr v2 store. It is extracted from `polarcap_analysis` LV2
processing but does not import `polarcap_analysis/src`.

## Install

From this directory:

```bash
python -m pip install -e '.[netcdf,test]'
```

Runtime dependencies are `numpy`, `xarray`, `dask`, `numcodecs`, and `zarr`.
For normal COSMO-SPECS files install at least one NetCDF backend:
`h5netcdf` for NetCDF-4/HDF5 and `netCDF4` for classic NetCDF.

`get_max_timesteps` and the default CLI scan use `ncdump -h`, so the NetCDF C
utilities must be on `PATH`. If `ncdump` is unavailable, pass `--max-time`.

## Quickstart

```bash
meteogram2zarr \
  --input-dir /work/.../ensemble_output/cs-eriswil__20260328_205320 \
  --meta-json /work/.../ensemble_output/cs-eriswil__20260328_205320/cs-eriswil__20260328_205320.json \
  --output Meteogram_cs-eriswil__20260328_205320.zarr \
  --overwrite
```

Subset for a smoke test:

```bash
meteogram2zarr \
  --input-dir /path/to/run \
  --meta-json /path/to/run/run.json \
  --output smoke.zarr \
  --experiments 20260328123456,20260328123556 \
  --variables U,V,T,QV \
  --max-time 20 \
  --overwrite
```

## Input Assumptions

- Files follow `M_??_??_??????????????.nc`.
- Experiment suffix is the final 14-character field before `.nc`.
- Station id is the second underscore-separated field, for example `01` in
  `M_01_03_20260328123456.nc`.
- Stations `SE` and `OB` are skipped by default.
- Optional run JSON has
  `cfg[exp]["INPUT_DIA"]["diactl"]["stationlist_tot"]`, shaped as rows of five
  values where columns 3 and 4 are latitude and longitude.
- COSMO vertical dims `HMLd` and `HHLd` are renamed to `height_level` and
  `height_level2`, then trimmed from the top.

This is not a generic NetCDF combiner. It preserves the COSMO-SPECS meteogram
schema used by PolarCAP.

## Output Schema

The output Zarr store uses:

- `expname`: experiment suffix, chunked by one experiment for region writes
- `station`: station id
- `time`
- optional `bins`, `height_level`, `height_level2`
- optional bin-center and bin-boundary coordinates for spectral variables
- optional `station_lat` and `station_lon` from the run JSON

Data are written as Zarr v2 with Blosc zstd compression.

## Slurm

If `levante-slurm-utils` is installed, the CLI can request a Dask cluster:

```bash
meteogram2zarr \
  --input-dir /path/to/run \
  --meta-json /path/to/run/run.json \
  --output run.zarr \
  --slurm \
  --slurm-account bb1234 \
  --slurm-partition compute
```

Small jobs are often faster without Slurm startup overhead. Start with a subset,
then scale only when the conversion is I/O or memory limited.

## Tests

```bash
python -m pytest
meteogram2zarr --help
```

