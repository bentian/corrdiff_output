#!/usr/bin/env python3
"""
Split a multi-group NetCDF file into yearly files, flatten truth/prediction
groups into separate outputs, and add more CF-compatible metadata.

This script:
1. Reads a NetCDF file containing root variables and "truth"/"prediction" groups.
2. Replaces the original time coordinate with a CF-compliant noleap daily time axis.
3. Splits the dataset into one file per year.
4. Flattens groups into separate output files:
   - truth_<year>.nc
   - prediction_<year>.nc
5. Renames variables and adds CF-style metadata.

Notes:
- Uses a noleap calendar: 365 days/year.
- Assumes the input time dimension length equals (end_year - start_year + 1) * 365.
- For precipitation, confirm the units before using standard_name="precipitation_flux".

Usage:
    python split_netcdf.py <input_nc> <start_year> <end_year> [--outdir DIR]
"""

from pathlib import Path
import argparse

import numpy as np
import xarray as xr
import cftime

VAR_RENAME = {
    "precipitation": "pr",
    "temperature_2m": "tas",
    "eastward_wind_10m": "uas",
    "northward_wind_10m": "vas",
}


def _build_time(n_time: int, start_year: int, end_year: int) -> tuple[np.ndarray, dict]:
    """Build CF-compliant noleap time."""
    expected = (end_year - start_year + 1) * 365
    if n_time != expected:
        raise ValueError(
            f"time dimension is {n_time}, but expected {expected} "
            f"for {start_year}-{end_year} with a noleap calendar."
        )

    units = f"days since {start_year}-01-01 00:00:00"
    time_values = cftime.num2date(
        np.arange(n_time),
        units=units,
        calendar="noleap",
    )
    encoding = {"units": units, "calendar": "noleap", "dtype": "int32"}
    return time_values, encoding


def _set_coord_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Set CF-compatible coordinate metadata."""
    if "lat" in ds:
        ds["lat"].attrs.update(
            {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
            }
        )

    if "lon" in ds:
        ds["lon"].attrs.update(
            {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
            }
        )

    if "time" in ds.coords:
        ds["time"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "time",
            }
        )

    return ds


def _add_height_coords(ds: xr.Dataset) -> xr.Dataset:
    """Add CF-compatible height coordinates."""
    ds = ds.assign_coords(
        height_2m=xr.DataArray(
            2.0,
            attrs={
                "standard_name": "height",
                "long_name": "height above surface",
                "units": "m",
                "positive": "up",
                "axis": "Z",
            },
        ),
        height_10m=xr.DataArray(
            10.0,
            attrs={
                "standard_name": "height",
                "long_name": "height above surface",
                "units": "m",
                "positive": "up",
                "axis": "Z",
            },
        ),
    )
    return ds


def _set_variable_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Set CF-compatible variable metadata."""
    # Rename variables first
    rename_map = {old: new for old, new in VAR_RENAME.items() if old in ds.data_vars}
    ds = ds.rename(rename_map)

    if "tas" in ds:
        ds["tas"].attrs.update(
            {
                "standard_name": "air_temperature",
                "long_name": "Near-surface air temperature",
                "units": "K",
                "coordinates": "height_2m",
            }
        )

    if "uas" in ds:
        ds["uas"].attrs.update(
            {
                "standard_name": "eastward_wind",
                "long_name": "Eastward wind at 10 m",
                "units": "m s-1",
                "coordinates": "height_10m",
            }
        )

    if "vas" in ds:
        ds["vas"].attrs.update(
            {
                "standard_name": "northward_wind",
                "long_name": "Northward wind at 10 m",
                "units": "m s-1",
                "coordinates": "height_10m",
            }
        )

    if "pr" in ds:
        # Convert mm/day to kg m-2 s-1
        ds["pr"] = ds["pr"] / 86400.0
        ds["pr"].attrs.update(
            {
                "standard_name": "precipitation_flux",
                "long_name": "Precipitation flux",
                "units": "kg m-2 s-1",
            }
        )

    return ds


def _prepare_group(
    group_name: str, shared: xr.Dataset, input_nc: str, time_values: np.ndarray
) -> xr.Dataset:
    """Prepare a single group."""
    ds = xr.open_dataset(input_nc, group=group_name, decode_times=False)
    ds = ds.assign_coords(time=("time", time_values))
    ds = xr.merge([ds, shared])
    ds = _add_height_coords(ds)
    ds = _set_coord_metadata(ds)
    ds = _set_variable_metadata(ds)
    return ds


def _open_and_prepare(
    input_nc: str, time_values: np.ndarray
) -> tuple[xr.Dataset, xr.Dataset, int]:
    """Open and prepare the dataset."""
    root = xr.open_dataset(input_nc, decode_times=False)
    shared = _set_coord_metadata(
        root.assign_coords(time=("time", time_values))[["lat", "lon"]]
    )

    return (
        _prepare_group("truth", shared, input_nc, time_values),
        _prepare_group("prediction", shared, input_nc, time_values),
        root.sizes["time"],
    )


def _write_year(
    ds: xr.Dataset, year: int, output_dir: Path, prefix: str, time_encoding: dict
) -> None:
    """Write a single year to a NetCDF file."""
    t0 = cftime.DatetimeNoLeap(year, 1, 1)
    t1 = cftime.DatetimeNoLeap(year, 12, 31)

    out_file = output_dir / f"{prefix}_{year}.nc"
    ds.sel(time=slice(t0, t1)).to_netcdf(
        out_file,
        mode="w",
        encoding={"time": time_encoding},
    )
    print(f"Wrote: {out_file}")


def convert_and_split_netcdf(
    input_nc: str, start_year: int, end_year: int, output_dir: Path
) -> None:
    """
    Split a NetCDF file by year and separate truth/prediction into standalone files.

    Args:
        input_nc: Path to the input NetCDF file.
        start_year: Start year.
        end_year: End year.
        output_dir: Output directory.
    """
    root = xr.open_dataset(input_nc, decode_times=False)
    time_values, time_encoding = _build_time(root.sizes["time"], start_year, end_year)
    root.close()

    truth_ds, pred_ds, _ = _open_and_prepare(input_nc, time_values)

    output_dir.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        _write_year(truth_ds, year, output_dir, "truth", time_encoding)
        _write_year(pred_ds, year, output_dir, "prediction", time_encoding)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Split NetCDF by year and separate truth/prediction without groups."
    )
    parser.add_argument("input_nc", help="Input NetCDF file")
    parser.add_argument("start_year", type=int, help="Start year, e.g. 2015")
    parser.add_argument("end_year", type=int, help="End year, e.g. 2080")
    parser.add_argument(
        "--outdir",
        default="split_by_year",
        help="Output directory",
    )
    args = parser.parse_args()

    convert_and_split_netcdf(
        input_nc=args.input_nc,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
