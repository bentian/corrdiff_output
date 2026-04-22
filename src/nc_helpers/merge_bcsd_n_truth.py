"""
Merge regridded BCSD data with grouped model output datasets.

This module performs the following steps:
- Loads grouped NetCDF input containing root, truth, and prediction data.
- Crops datasets to match the BCSD target grid.
- Removes unwanted variables (e.g., wind components).
- Replaces prediction variables (precipitation, temperature) with BCSD values.
- Ensures dimensional consistency across datasets.
- Writes the final merged dataset back to NetCDF with grouped structure.

Key features:
- Handles mismatched calendars by discarding prediction time coordinates and
  inheriting them from the root dataset.
- Uses raw data assignment to avoid xarray coordinate alignment issues.
- Supports multiple SSP scenarios through configurable path building.

This script depends on `regrid_bcsd.py` for generating the BCSD fields and
`utils.py` for path configuration and shared constants.
"""

from typing import Optional

import xarray as xr

from regrid_bcsd import regrid_bcsd
from bcsd_utils import build_paths, TIME_SLICE


DROP_WIND_VARS = ["eastward_wind_10m", "northward_wind_10m"]


def center_crop_slices(
    src_y: int, src_x: int, dst_y: int, dst_x: int
) -> tuple[slice, slice]:
    """Calculate slices for center-cropping a 2D array."""
    if dst_y > src_y or dst_x > src_x:
        raise ValueError(
            f"Cannot crop from {(src_y, src_x)} to larger {(dst_y, dst_x)}"
        )
    y0 = (src_y - dst_y) // 2
    x0 = (src_x - dst_x) // 2
    return slice(y0, y0 + dst_y), slice(x0, x0 + dst_x)


def crop_and_clean_dataset(
    ds: xr.Dataset,
    target_y: int,
    target_x: int,
    drop_vars: Optional[list[str]] = None,
) -> xr.Dataset:
    """Crop and clean a dataset."""
    y_slice, x_slice = center_crop_slices(
        ds.sizes["y"], ds.sizes["x"], target_y, target_x
    )
    ds_new = ds.isel(y=y_slice, x=x_slice)

    if drop_vars is not None:
        ds_new = ds_new.drop_vars(drop_vars, errors="ignore")

    return ds_new


def open_grouped_input(input_nc: str) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Open grouped input dataset."""
    ds_root = xr.open_dataset(input_nc).isel(time=TIME_SLICE)
    ds_truth = xr.open_dataset(input_nc, group="truth").isel(time=TIME_SLICE)
    ds_pred = xr.open_dataset(input_nc, group="prediction").isel(time=TIME_SLICE)
    return ds_root, ds_truth, ds_pred


def build_prediction_dataset(
    ds_pred: xr.Dataset,
    ds_bcsd_out: xr.Dataset,
    target_y: int,
    target_x: int,
    drop_vars: Optional[list[str]] = None,
) -> xr.Dataset:
    """Build prediction dataset."""
    ds_new = crop_and_clean_dataset(ds_pred, target_y, target_x, drop_vars)
    if "ensemble" in ds_new.dims:
        ds_new = ds_new.isel(ensemble=slice(0, 1))

    # Assert shapes match
    assert ds_new.sizes["time"] == ds_bcsd_out.sizes["time"]
    assert ds_new.sizes["y"] == ds_bcsd_out.sizes["y"]
    assert ds_new.sizes["x"] == ds_bcsd_out.sizes["x"]

    # Remove prediction time and BCSD grid coords so root can provide them later
    ds_new = ds_new.drop_vars(["XLAT", "XLONG"], errors="ignore")
    if "time" in ds_new.coords:
        ds_new = ds_new.reset_coords("time", drop=True)

    # Assign BCSD values by raw data, ignoring BCSD time/calendar coords
    for dst, src in {"precipitation": "pr", "temperature_2m": "tas"}.items():
        ds_new[dst].data = ds_bcsd_out[src].data[None, ...]
        ds_new[dst].attrs.update(ds_bcsd_out[src].attrs)

    return ds_new


def validate_shapes(
    ds_root_new: xr.Dataset,
    ds_truth_new: xr.Dataset,
    ds_pred_new: xr.Dataset,
    ds_bcsd_out: xr.Dataset,
) -> None:
    """Validate shapes of datasets."""
    expected = {
        "time": ds_bcsd_out.sizes["time"],
        "y": ds_bcsd_out.sizes["y"],
        "x": ds_bcsd_out.sizes["x"],
    }

    for name, ds in {
        "root": ds_root_new,
        "truth": ds_truth_new,
        "prediction": ds_pred_new,
    }.items():
        for dim, size in expected.items():
            if ds.sizes[dim] != size:
                raise ValueError(
                    f"{dim} mismatch for {name}: {ds.sizes[dim]} vs {size}"
                )


def write_grouped_output(
    ds_root_new: xr.Dataset,
    ds_truth_new: xr.Dataset,
    ds_pred_new: xr.Dataset,
    out_nc: str,
) -> None:
    """Write grouped output dataset."""
    ds_root_new.to_netcdf(out_nc, mode="w")
    ds_truth_new.to_netcdf(out_nc, mode="a", group="truth")
    ds_pred_new.to_netcdf(out_nc, mode="a", group="prediction")
    print("Saved:", out_nc)


def main() -> None:
    """Main function."""
    scenarios = {
        "ssp126": "W1-1",
        "ssp245": "W1-2",
        "ssp370": "W1-3",
        "ssp585": "W1-5",
    }

    for ssp, w1 in scenarios.items():
        print(f"\n=== Processing {ssp} ({w1}) ===")
        bcsd_nc, input_nc, out_nc = build_paths(ssp, w1)

        ds_bcsd = regrid_bcsd(bcsd_nc)
        ty, tx = ds_bcsd.sizes["y"], ds_bcsd.sizes["x"]

        ds_root, ds_truth, ds_pred = open_grouped_input(input_nc)
        ds_root_new = crop_and_clean_dataset(ds_root, ty, tx)
        ds_truth_new = crop_and_clean_dataset(ds_truth, ty, tx, DROP_WIND_VARS)
        ds_pred_new = build_prediction_dataset(ds_pred, ds_bcsd, ty, tx, DROP_WIND_VARS)

        validate_shapes(ds_root_new, ds_truth_new, ds_pred_new, ds_bcsd)
        write_grouped_output(ds_root_new, ds_truth_new, ds_pred_new, out_nc)
        # DEBUG
        # ds_pred_new.to_netcdf(out_nc.replace(".nc", "_prediction.nc"))


if __name__ == "__main__":
    main()
