"""
Regrid BCSD (Bias-Corrected Spatial Disaggregation) data onto a target grid.

This module:
- Loads BCSD variables precipitation and temperature.
- Applies fill-value masking to remove invalid data.
- Converts temperature units from Celsius to Kelvin.
- Regrids data to a target grid using xESMF (bilinear interpolation).
- Applies a landmask to filter out non-land grid cells.

Key features:
- Uses bilinear interpolation with safeguards for degenerate grid cells.
- Handles variable-specific preprocessing (e.g., unit conversion).
- Ensures output fields are spatially aligned and masked consistently.

The output dataset is used downstream for merging with model predictions
(e.g., in `merge_bcsd_n_truth.py`).

Dependencies:
- xarray for data handling
- xESMF for regridding
"""

import xarray as xr
import xesmf as xe

from bcsd_config import TIME_SLICE

GRID_NC = "./ssp_128x96_grid_coords.nc"


def _fmt_idx(idx) -> str:
    """Format an index to a string."""
    t, lat, lon = idx
    t_str = f"{t.year:04d}-{t.month:02d}-{t.day:02d}"
    return f"({t_str}, {lat:.2f}, {lon:.2f})"


def _print_minmax(da: xr.DataArray, label: str = "") -> None:
    """Print min and max of a DataArray."""
    vmin = float(da.min(skipna=True).values)
    vmax = float(da.max(skipna=True).values)

    stacked = da.stack(points=da.dims)
    imin = _fmt_idx(stacked.idxmin(dim="points", skipna=True).item())
    imax = _fmt_idx(stacked.idxmax(dim="points", skipna=True).item())

    if label:
        print(label)
        print(f"  min = {vmin:.2f} at {imin}")
        print(f"  max = {vmax:.2f} at {imax}")


def _open_target_grid() -> tuple[xr.DataArray, xr.Dataset]:
    """Open target grid."""
    ds_grid = xr.open_dataset(GRID_NC)
    grid_dst = xr.Dataset(
        {
            "lat": (("y", "x"), ds_grid["XLAT"].values),
            "lon": (("y", "x"), ds_grid["XLONG"].values),
        }
    )
    landmask = ds_grid["LANDMASK"].rename({"south_north": "y", "west_east": "x"})
    return landmask, grid_dst


def _open_bcsd_grid(src_file: str) -> tuple[xr.Dataset, xr.Dataset]:
    """Open BCSD grid."""
    ds_bcsd = xr.open_dataset(src_file).isel(time=TIME_SLICE)
    grid_src = xr.Dataset({"lat": ds_bcsd["lat"], "lon": ds_bcsd["lon"]})
    return ds_bcsd, grid_src


def _build_regridder(grid_src: xr.Dataset, grid_dst: xr.Dataset) -> xe.Regridder:
    """Build regridder."""
    return xe.Regridder(
        grid_src,
        grid_dst,
        method="bilinear",
        periodic=False,
        ignore_degenerate=True,
    )


def _apply_landmask(ds_out: xr.Dataset, landmask: xr.DataArray) -> xr.Dataset:
    """Apply landmask to dataset."""
    for var in ["pr", "tas"]:
        ds_out[var] = ds_out[var].where(landmask > 0.5)
        _print_minmax(ds_out[var], f"\n[{var}] AFTER landmasking")
    return ds_out


def _build_regridded_dataset(
    src_files: list[str], landmask: xr.DataArray, regridder: xe.Regridder
) -> xr.Dataset:
    """Build regridded dataset."""
    out = {}
    for src_file in src_files:
        ds = xr.open_dataset(src_file).isel(time=TIME_SLICE)

        var = list(ds.data_vars)[0]
        da = ds[var].where(ds[var] != ds[var].attrs.get("_FillValue", -99.9))
        _print_minmax(da, f"\n[{var}] BEFORE regridding")

        # Convert Celsius to Kelvin
        if var == "tas":
            da = da + 273.15
            da.attrs["units"] = "K"
            _print_minmax(da, f"[{var}] Celsius -> Kelvin")

        da = regridder(da)
        _print_minmax(da, f"[{var}] AFTER regridding")
        out[var] = da

    ds_out = xr.Dataset(out)
    ds_out = _apply_landmask(ds_out, landmask)
    return ds_out


def regrid_bcsd(src_files: list[str]) -> xr.Dataset:
    """Regrid BCSD data."""
    landmask, grid_dst = _open_target_grid()
    _, grid_src = _open_bcsd_grid(src_files[0])
    regridder = _build_regridder(grid_src, grid_dst)
    return _build_regridded_dataset(src_files, landmask, regridder)
