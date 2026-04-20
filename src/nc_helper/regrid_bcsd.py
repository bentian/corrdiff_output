"""
This script regrids BCSD pr/tas to the CorrDiff grid, then replaces
prediction precipitation/temperature_2m in a grouped NetCDF output.

Changes:
- Center-crop root lat/lon and truth values to 128x96
- Drop eastward_wind_10m and northward_wind_10m from both truth and prediction
- Keep cropped truth precipitation and temperature_2m unchanged
- Replace prediction precipitation with regridded BCSD pr
- Replace prediction temperature_2m with regridded BCSD tas
- Remove ensemble dimension from prediction
"""

from typing import Optional

import xarray as xr
import xesmf as xe

GRID_NC = "./ssp_128x96_grid_coords.nc"
DROP_WIND_VARS = ["eastward_wind_10m", "northward_wind_10m"]

TIME_SLICE = slice(0, 24090)  # 2015-01-01 to 2080-12-31


def print_minmax(da: xr.DataArray, label: str = "") -> None:
    """Print min/max of a DataArray."""
    vmin = float(da.min(skipna=True).values)
    vmax = float(da.max(skipna=True).values)
    if label:
        print(label)
    print(f"  min = {vmin}")
    print(f"  max = {vmax}")


def center_crop_slices(
    src_y: int, src_x: int, dst_y: int, dst_x: int
) -> tuple[slice, slice]:
    """Return center-crop slices."""
    if dst_y > src_y or dst_x > src_x:
        raise ValueError(
            f"Cannot crop from {(src_y, src_x)} to larger {(dst_y, dst_x)}"
        )

    y0 = (src_y - dst_y) // 2
    x0 = (src_x - dst_x) // 2
    return slice(y0, y0 + dst_y), slice(x0, x0 + dst_x)


def open_target_grid(grid_nc: str) -> tuple[xr.DataArray, xr.Dataset]:
    """Open target grid file and build xESMF-compatible target grid."""
    ds_grid = xr.open_dataset(grid_nc)
    grid_dst = xr.Dataset(
        {
            "lat": (("y", "x"), ds_grid["XLAT"].values),
            "lon": (("y", "x"), ds_grid["XLONG"].values),
        }
    )
    landmask = ds_grid["LANDMASK"].rename({"south_north": "y", "west_east": "x"})
    return landmask, grid_dst


def open_bcsd_grid(src_file: str) -> tuple[xr.Dataset, xr.Dataset]:
    """Open first BCSD source file to build source grid."""
    ds_bcsd = xr.open_dataset(src_file).isel(time=TIME_SLICE)

    grid_src = xr.Dataset(
        {
            "lat": ds_bcsd["lat"],
            "lon": ds_bcsd["lon"],
        }
    )
    return ds_bcsd, grid_src


def build_regridder(grid_src: xr.Dataset, grid_dst: xr.Dataset) -> xe.Regridder:
    """Build xESMF regridder."""
    return xe.Regridder(
        grid_src,
        grid_dst,
        method="bilinear",
        periodic=False,
        ignore_degenerate=True,
    )


def apply_landmask(
    ds_out: xr.Dataset,
    landmask: xr.DataArray,
    vars_to_mask: Optional[list[str]] = None,
) -> xr.Dataset:
    """Apply landmask to a dataset."""
    if vars_to_mask is None:
        vars_to_mask = list(ds_out.data_vars)

    for var in vars_to_mask:
        ds_out[var] = ds_out[var].where(landmask == 1)
        print_minmax(ds_out[var], f"\n[{var}] AFTER LANDMASK")

    return ds_out


def build_regridded_dataset(
    src_files: list[str], landmask: xr.DataArray, regridder: xe.Regridder
) -> xr.Dataset:
    """Build regridded dataset."""
    out = {}
    for f in src_files:
        ds = xr.open_dataset(f).isel(time=TIME_SLICE)

        var = list(ds.data_vars)[0]
        da = ds[var].where(ds[var] != ds[var].attrs.get("_FillValue", -99.9))
        print_minmax(da, f"\n[{var}] BEFORE")

        da = regridder(da)
        print_minmax(da, f"[{var}] AFTER")

        out[var] = da

    ds_out = xr.Dataset(out)
    return apply_landmask(ds_out, landmask, ["pr", "tas"])


def open_grouped_input(input_nc: str) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Open grouped input dataset."""
    ds_root = xr.open_dataset(input_nc).isel(time=TIME_SLICE)
    ds_truth = xr.open_dataset(input_nc, group="truth").isel(time=TIME_SLICE)
    ds_pred = xr.open_dataset(input_nc, group="prediction").isel(time=TIME_SLICE)
    return ds_root, ds_truth, ds_pred


def crop_and_clean_dataset(
    ds: xr.Dataset,
    target_y: int,
    target_x: int,
    drop_vars: Optional[list[str]] = None,
) -> xr.Dataset:
    """Center-crop dataset to (target_y, target_x), optionally drop variables and subset time."""
    y_slice, x_slice = center_crop_slices(
        ds.sizes["y"], ds.sizes["x"], target_y, target_x
    )
    ds_new = ds.isel(y=y_slice, x=x_slice, time=TIME_SLICE)

    # Drop unwanted variables
    if drop_vars is not None:
        ds_new = ds_new.drop_vars(drop_vars, errors="ignore")

    return ds_new


def build_prediction_dataset(
    ds_pred: xr.Dataset,
    ds_bcsd_out: xr.Dataset,
    target_y: int,
    target_x: int,
    drop_vars: Optional[list[str]] = None,
) -> xr.Dataset:
    """Replace prediction precip/temp with BCSD and remove wind + ensemble."""
    ds_new = crop_and_clean_dataset(ds_pred, target_y, target_x, drop_vars)
    if "ensemble" in ds_new.dims:
        ds_new = ds_new.drop_dims("ensemble")

    ds_new["precipitation"] = ds_bcsd_out["pr"]
    ds_new["temperature_2m"] = ds_bcsd_out["tas"]

    return ds_new


def validate_shapes(
    ds_root_new: xr.Dataset,
    ds_truth_new: xr.Dataset,
    ds_pred_new: xr.Dataset,
    ds_bcsd_out: xr.Dataset,
) -> None:
    """Validate shapes of datasets."""
    for name, ds in {
        "root": ds_root_new,
        "truth": ds_truth_new,
        "prediction": ds_pred_new,
    }.items():
        if ds.sizes["time"] != ds_bcsd_out.sizes["time"]:
            raise ValueError(
                f"time mismatch for {name}: {ds.sizes['time']} vs {ds_bcsd_out.sizes['time']}"
            )
        if ds.sizes["y"] != ds_bcsd_out.sizes["y"]:
            raise ValueError(
                f"y mismatch for {name}: {ds.sizes['y']} vs {ds_bcsd_out.sizes['y']}"
            )
        if ds.sizes["x"] != ds_bcsd_out.sizes["x"]:
            raise ValueError(
                f"x mismatch for {name}: {ds.sizes['x']} vs {ds_bcsd_out.sizes['x']}"
            )


def build_group_encoding(ds: xr.Dataset) -> dict:
    """Build encoding for grouped dataset."""
    encoding = {}
    for varname, da in ds.data_vars.items():
        fill_value = da.attrs.get("_FillValue", da.attrs.get("missing_value", None))
        encoding[varname] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
        if fill_value is not None:
            encoding[varname]["_FillValue"] = fill_value

    for coord_name in ds.coords:
        if coord_name in ds.variables and coord_name not in encoding:
            encoding[coord_name] = ds[coord_name].encoding.copy()

    return encoding


def write_grouped_output(
    ds_root_new: xr.Dataset,
    ds_truth_new: xr.Dataset,
    ds_pred_new: xr.Dataset,
    out_nc: str,
) -> None:
    """Write grouped output dataset."""
    ds_root_new.to_netcdf(out_nc, mode="w")
    ds_truth_new.to_netcdf(
        out_nc,
        mode="a",
        group="truth",
    )
    ds_pred_new.to_netcdf(
        out_nc,
        mode="a",
        group="prediction",
    )
    print("Saved:", out_nc)


def write_output(ds: xr.Dataset, out_nc: str) -> None:
    """Write truth and prediction to separate NetCDF files."""
    ds.to_netcdf(out_nc)
    print("Saved:", out_nc)


def main() -> None:
    """Main function."""
    scenarios = {
        "ssp126": "W1-1",
        "ssp245": "W1-2",
        "ssp370": "W1-3",
        "ssp585": "W1-5",
    }

    landmask, grid_dst = open_target_grid(GRID_NC)
    for ssp, w1 in scenarios.items():
        print(f"\n=== Processing {ssp} ({w1}) ===")

        bcsd_nc = [
            f"/lfs/archive/TCCIP_data/CMIP6_QDM/pr/{ssp}/TaiESM1/r1i1p1f1/pr_QDM_TaiESM1.nc",
            f"/lfs/archive/TCCIP_data/CMIP6_QDM/tas/{ssp}/TaiESM1/r1i1p1f1/tas_QDM_TaiESM1.nc",
        ]
        input_nc = f"../SSP_result/W1/{w1}/netcdf/output_0_all_masked.nc"
        out_nc = f"./B-{w1[3]}/netcdf/bcsd_masked.nc"

        _, grid_src = open_bcsd_grid(bcsd_nc[0])
        regridder = build_regridder(grid_src, grid_dst)

        ds_bcsd = build_regridded_dataset(bcsd_nc, landmask, regridder)
        ty, tx = ds_bcsd.sizes["y"], ds_bcsd.sizes["x"]
        # write_output(ds_bcsd, out_nc.replace(".nc", "_regridded.nc"))

        print("\nOpening, cropping and cleaning datasets...")
        ds_root, ds_truth, ds_pred = open_grouped_input(input_nc)
        ds_root_new = crop_and_clean_dataset(ds_root, ty, tx)
        ds_truth_new = crop_and_clean_dataset(ds_truth, ty, tx, DROP_WIND_VARS)
        ds_pred_new = build_prediction_dataset(ds_pred, ds_bcsd, ty, tx, DROP_WIND_VARS)

        print("Validating shapes & writing grouped output...")
        validate_shapes(ds_root_new, ds_truth_new, ds_pred_new, ds_bcsd)
        write_grouped_output(ds_root_new, ds_truth_new, ds_pred_new, out_nc)
        write_output(ds_pred_new, out_nc.replace(".nc", "_prediction.nc"))  # DEBUG


if __name__ == "__main__":
    main()
