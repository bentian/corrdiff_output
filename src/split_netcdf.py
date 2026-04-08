"""
Split a multi-group NetCDF file into yearly files with CF-compliant time,
and flatten groups into separate outputs.

This script:
1. Reads a NetCDF file containing:
   - Root variables (e.g., lat, lon, time)
   - Group "truth"
   - Group "prediction"
2. Replaces the original time coordinate with a CF-compliant "noleap" calendar:
   - Daily frequency
   - From <start_year>-01-01 to <end_year>-12-31
   - Total length must equal (end_year - start_year + 1) * 365
3. Splits the dataset into one file per year.
4. Flattens NetCDF groups:
   - Writes "truth" variables into standalone files:  truth_<year>.nc
   - Writes "prediction" variables into standalone files: prediction_<year>.nc
   - No groups are used in output files
5. Preserves shared coordinates (lat, lon) in each output file.

Output:
- truth_<year>.nc
- prediction_<year>.nc

Each output file contains:
- dimensions: time, y, x (and ensemble for prediction)
- coordinates: time (CF noleap), lat, lon
- variables: corresponding truth or prediction variables

Usage:
    python split_nc.py <input_nc> <start_year> <end_year> [--outdir DIR]

Example:
    python split_nc.py output_0_all_masked.nc 2075 2080

Notes:
- Assumes daily data with no leap years (365 days/year).
- Will raise an error if time dimension size does not match expected length.
- Uses cftime for CF-compliant calendar handling.
"""

from pathlib import Path
import argparse

import numpy as np
import xarray as xr
import cftime


def split_netcdf(input_nc: str, start_year: int, end_year: int, output_dir: Path):
    """
    Split a NetCDF file by year and separate truth/prediction (no groups).

    Args:
        input_nc: Path to the input NetCDF file.
        start_year: Start year (e.g., 2075).
        end_year: End year (e.g., 2080).
        output_dir: Directory to save the output files.
    """
    # -----------------------------------------------------------------------------
    # Open datasets
    # -----------------------------------------------------------------------------
    root = xr.open_dataset(input_nc, decode_times=False)
    truth = xr.open_dataset(input_nc, group="truth", decode_times=False)
    prediction = xr.open_dataset(input_nc, group="prediction", decode_times=False)

    # -----------------------------------------------------------------------------
    # Build CF-compatible noleap time
    # -----------------------------------------------------------------------------
    n_time = root.sizes["time"]
    expected_n_time = (end_year - start_year + 1) * 365

    if n_time != expected_n_time:
        raise ValueError(
            f"time dimension is {n_time}, but expected {expected_n_time} "
            f"for {start_year}-{end_year} with a noleap calendar."
        )

    time_values = cftime.num2date(
        np.arange(n_time),
        units=f"days since {start_year}-01-01 00:00:00",
        calendar="noleap",
    )

    time_encoding = {
        "units": f"days since {start_year}-01-01 00:00:00",
        "calendar": "noleap",
        "dtype": "int32",
    }

    # assign time
    root = root.assign_coords(time=("time", time_values))
    truth = truth.assign_coords(time=("time", time_values))
    prediction = prediction.assign_coords(time=("time", time_values))

    # shared coords
    shared_coords = root[["lat", "lon"]]

    truth_flat = xr.merge([truth, shared_coords])
    prediction_flat = xr.merge([prediction, shared_coords])

    # -----------------------------------------------------------------------------
    # Split and write
    # -----------------------------------------------------------------------------
    output_dir.mkdir(exist_ok=True)
    for year in range(start_year, end_year + 1):
        t0 = cftime.DatetimeNoLeap(year, 1, 1)
        t1 = cftime.DatetimeNoLeap(year, 12, 31)

        truth_y = truth_flat.sel(time=slice(t0, t1))
        pred_y = prediction_flat.sel(time=slice(t0, t1))

        truth_out = output_dir / f"truth_{year}.nc"
        pred_out = output_dir / f"prediction_{year}.nc"

        truth_y.to_netcdf(
            truth_out,
            mode="w",
            encoding={"time": time_encoding},
        )

        pred_y.to_netcdf(
            pred_out,
            mode="w",
            encoding={"time": time_encoding},
        )

        print(f"Wrote: {truth_out}")
        print(f"Wrote: {pred_out}")


def main():
    """Main function to split NetCDF by year and separate truth/prediction (no groups)."""
    # -----------------------------------------------------------------------------
    # CLI arguments
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Split NetCDF by year and separate truth/prediction (no groups)."
    )
    parser.add_argument("input_nc", help="Input NetCDF file")
    parser.add_argument("start_year", type=int, help="Start year (e.g., 2075)")
    parser.add_argument("end_year", type=int, help="End year (e.g., 2080)")
    parser.add_argument(
        "--outdir",
        default="split_by_year",
        help="Output directory (default: %(default)s)",
    )

    args = parser.parse_args()

    split_netcdf(
        input_nc=args.input_nc,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
