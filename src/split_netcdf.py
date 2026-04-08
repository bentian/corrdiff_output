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


def _build_time(n_time: int, start_year: int, end_year: int):
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


def _open_and_prepare(input_nc: str, time_values):
    """Open and prepare datasets."""
    root = xr.open_dataset(input_nc, decode_times=False)
    truth = xr.open_dataset(input_nc, group="truth", decode_times=False)
    pred = xr.open_dataset(input_nc, group="prediction", decode_times=False)

    root = root.assign_coords(time=("time", time_values))
    truth = truth.assign_coords(time=("time", time_values))
    pred = pred.assign_coords(time=("time", time_values))

    shared = root[["lat", "lon"]]
    return xr.merge([truth, shared]), xr.merge([pred, shared]), root.sizes["time"]


def _write_year(ds, year: int, output_dir: Path, prefix: str, encoding):
    """Write a single year to a NetCDF file."""
    t0 = cftime.DatetimeNoLeap(year, 1, 1)
    t1 = cftime.DatetimeNoLeap(year, 12, 31)

    out = output_dir / f"{prefix}_{year}.nc"
    ds.sel(time=slice(t0, t1)).to_netcdf(
        out,
        mode="w",
        encoding={"time": encoding},
    )
    print(f"Wrote: {out}")


def split_netcdf(input_nc: str, start_year: int, end_year: int, output_dir: Path):
    """Split a NetCDF file by year and separate truth/prediction (no groups)."""
    # build time first (needs n_time → open root minimally)
    root = xr.open_dataset(input_nc, decode_times=False)
    time_values, encoding = _build_time(root.sizes["time"], start_year, end_year)
    root.close()

    truth_ds, pred_ds, _ = _open_and_prepare(input_nc, time_values)

    output_dir.mkdir(exist_ok=True)

    for year in range(start_year, end_year + 1):
        _write_year(truth_ds, year, output_dir, "truth", encoding)
        _write_year(pred_ds, year, output_dir, "prediction", encoding)


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
