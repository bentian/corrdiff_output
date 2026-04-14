"""
Split a multi-group NetCDF file into yearly, CF-compliant outputs.

This script:
1. Reads a NetCDF file containing root variables and groups:
   - "truth"
   - "prediction"
2. Reconstructs a CF-compliant time coordinate:
   - daily frequency
   - noleap calendar (365 days/year)
   - from <start_year> to <end_year>
3. Processes data **year-by-year and group-by-group** to minimize peak memory usage.
4. Flattens groups into standalone NetCDF files organized into folders:
   - truth/
   - prediction_all/
   - prediction_reg/ (regression mode only)
5. Converts variables to more CF-compliant form:
   - renames variables (e.g., precipitation → pr, temperature_2m → tas)
   - adds standard_name, units, and height coordinates where applicable
   - converts precipitation from mm/day → kg m-2 s-1
6. Ensures dimension order compatibility:
   - (time, ensemble, y, x) for prediction
   - (time, y, x) for truth
7. Supports regression mode:
   - skips writing truth outputs
   - keeps only the first ensemble member (ensemble=0)

Output structure:
    <output_dir>/
        truth/
            truth_<year>.nc
        prediction_all/
            prediction_<year>.nc
        prediction_reg/   (only if --regression-mode)
            prediction_<year>.nc

Notes:
- Assumes input time length = (end_year - start_year + 1) * 365
- Uses cftime for CF-compliant time handling
- Lat/lon are treated as static 2D coordinates (y, x)

Usage:
    python split_n_convert_nc.py <input_nc> <start_year> <end_year> \
        [--outdir DIR] [--regression-mode]
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
    """Add CF-compatible scalar height coordinates."""
    return ds.assign_coords(
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


def _set_variable_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Rename variables and set CF-style metadata."""
    ds = ds.rename({old: new for old, new in VAR_RENAME.items() if old in ds.data_vars})

    if "tas" in ds:
        ds["tas"].attrs.update(
            {
                "standard_name": "air_temperature",
                "long_name": "Near-surface air temperature",
                "units": "K",
                "coordinates": "lon lat height_2m",
            }
        )

    if "uas" in ds:
        ds["uas"].attrs.update(
            {
                "standard_name": "eastward_wind",
                "long_name": "Eastward wind at 10 m",
                "units": "m s-1",
                "coordinates": "lon lat height_10m",
            }
        )

    if "vas" in ds:
        ds["vas"].attrs.update(
            {
                "standard_name": "northward_wind",
                "long_name": "Northward wind at 10 m",
                "units": "m s-1",
                "coordinates": "lon lat height_10m",
            }
        )

    if "pr" in ds:
        ds["pr"] = ds["pr"] / 86400.0
        ds["pr"].attrs.update(
            {
                "standard_name": "precipitation_flux",
                "long_name": "Precipitation flux",
                "units": "kg m-2 s-1",
                "coordinates": "lon lat",
            }
        )

    return ds


def _prepare_year_group(
    input_nc: str,
    group_name: str,
    shared_coords: xr.Dataset,
    time_values: np.ndarray,
    year: int,
) -> xr.Dataset:
    """Open one group, slice one year, then convert to CF-compatible form."""
    with xr.open_dataset(input_nc, group=group_name, decode_times=False) as ds:
        year_ds = ds.assign_coords(time=("time", time_values)).sel(
            time=slice(
                cftime.DatetimeNoLeap(year, 1, 1),
                cftime.DatetimeNoLeap(year, 12, 31),
            )
        )
        year_ds = xr.merge([year_ds, shared_coords])

    year_ds = _add_height_coords(year_ds)
    year_ds = _set_coord_metadata(year_ds)
    year_ds = _set_variable_metadata(year_ds)

    # Ensure time is the leading dimension for better tool compatibility
    if "ensemble" in year_ds.dims:
        year_ds = year_ds.transpose("time", "ensemble", "y", "x")

    return year_ds


def split_n_convert_nc(
    input_nc: str,
    start_year: int,
    end_year: int,
    output_dir: Path,
    regression_mode: bool = False,
) -> None:
    """
    Split a NetCDF file by year and separate truth/prediction into standalone files.

    This implementation minimizes peak memory by processing one year and one group at a time.

    Args:
        input_nc: Input NetCDF path.
        start_year: Start year.
        end_year: End year.
        output_dir: Output directory.
        regression_mode: If True, skip writing truth netcdf & write only ensemble=0 for prediction.
    """
    with xr.open_dataset(input_nc, decode_times=False) as root:
        time_values, time_encoding = _build_time(
            root.sizes["time"], start_year, end_year
        )
        shared_coords = _set_coord_metadata(root[["lat", "lon"]]).set_coords(
            ["lat", "lon"]
        )

    groups = ["prediction"] if regression_mode else ["truth", "prediction"]
    for group_name in groups:
        # Determine the target directory based on the group name and regression mode
        target_dir = output_dir / (
            "truth"
            if group_name == "truth"
            else "prediction_reg"
            if regression_mode
            else "prediction_all"
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over each year
        for year in range(start_year, end_year + 1):
            ds = _prepare_year_group(
                input_nc, group_name, shared_coords, time_values, year
            )

            if regression_mode and "ensemble" in ds.dims:
                ds = ds.isel(ensemble=0, drop=True)

            out_file = target_dir / f"{group_name}_{year}.nc"
            ds.to_netcdf(out_file, mode="w", encoding={"time": time_encoding})
            print(f"Wrote: {out_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Split NetCDF by year and separate truth/prediction without groups."
    )
    parser.add_argument("input_nc", help="Input NetCDF file")
    parser.add_argument("start_year", type=int, help="Start year, e.g. 2015")
    parser.add_argument("end_year", type=int, help="End year, e.g. 2080")
    parser.add_argument("--outdir", default="split_by_year", help="Output directory")
    parser.add_argument(
        "--regression-mode",
        action="store_true",
        help="If set, indicate regression mode (ensemble=1 & skip truth output).",
    )
    args = parser.parse_args()

    split_n_convert_nc(
        input_nc=args.input_nc,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=Path(args.outdir),
        regression_mode=args.regression_mode,
    )


if __name__ == "__main__":
    main()
