"""
Stream-apply a landmask to truth and prediction samples in a NetCDF file.

This module reads an input NetCDF containing a root group and two subgroups,
``truth`` and ``prediction``, applies a spatial landmask to all sample variables,
and writes a new NetCDF file using a streaming, chunked approach suitable for
very large datasets.

Output characteristics:
- The root group is preserved and written first, including the ``time`` coordinate
  and ``lat``/``lon`` data variables, so existing consumers (e.g., ``open_samples``)
  continue to work unchanged.
- The ``truth`` and ``prediction`` groups contain only dimensions and masked data
  variables, matching the original input format exactly (i.e., dimensions without
  coordinates and no ``lat``/``lon`` variables).
- No concatenation is performed in memory; data are processed and written in
  time chunks to keep memory usage bounded.

Assumptions:
- Truth variables have dimensions ``(time, y, x)``.
- Prediction variables have dimensions ``(ensemble, time, y, x)`` or
  ``(time, y, x)``.
- The landmask is provided as a static 2D field on ``(y, x)``.

This design ensures compatibility with existing downstream analysis code while
supporting efficient processing of very large NetCDF files.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import xarray as xr
from netCDF4 import Dataset, Variable
from tqdm.auto import tqdm

LANDMASK_NC = "../data/ssp_208x208_grid_coords.nc"
TCHUNK_SIZE = 365


def get_timestamp() -> str:
    """Return a human-readable timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _ensure_dim(group: Dataset, name: str, size: Optional[int]) -> None:
    """Create a netCDF dimension if missing."""
    if name not in group.dimensions:
        group.createDimension(name, size)

def _get_or_create_group(nc: Dataset, name: str) -> Dataset:
    """Return an existing netCDF group, or create it if missing."""
    return nc.groups[name] if name in nc.groups else nc.createGroup(name)

def _create_out_vars(src: xr.Dataset, group: Dataset) -> dict[str, Variable]:
    """
    Create netCDF variables matching src.data_vars and return a name->Variable mapping.
    (No coords are created; only data vars.)
    """
    out: dict[str, Variable] = {}
    for name, da in src.data_vars.items():
        dims = tuple(da.dims)
        dtype = np.dtype(da.dtype)
        if name not in group.variables:
            group.createVariable(name, dtype, dims, zlib=True, complevel=1)
        out[name] = group.variables[name]
    return out

def _init_group_schema_as_input(group: Dataset, src: xr.Dataset) -> dict[str, Variable]:
    """
    Create only dimensions and data variables to match the input group format
    (i.e., dimensions without coordinates).
    """
    for d in src.dims:
        _ensure_dim(group, d, src.sizes[d])
    return _create_out_vars(src, group)

def _mask_chunk(truth_t: xr.Dataset, pred_t: xr.Dataset,
                landmask_t: xr.DataArray) -> tuple[xr.Dataset, xr.Dataset]:
    """Apply landmask to chunk; expand mask for ensemble if needed."""
    truth_masked = truth_t.where(landmask_t == 1)

    lm_pred = landmask_t
    if "ensemble" in pred_t.dims:
        lm_pred = lm_pred.expand_dims(ensemble=pred_t.sizes["ensemble"])
    pred_masked = pred_t.where(lm_pred == 1)

    # Clamp to remove negative precipitation
    # if "precipitation" in pred_masked.data_vars:
    #     pred_masked["precipitation"] = pred_masked["precipitation"].clip(min=0)

    return truth_masked, pred_masked

def _write_vars_chunk(out_vars: dict[str, Variable], ds_chunk: xr.Dataset,
                      t0: int, t1: int) -> None:
    """
    Write a masked dataset chunk to netCDF variables, aligning dims and slicing on 'time'
    wherever it appears (time may not be the first dimension, e.g. (ensemble,time,y,x)).
    """
    for name, da in ds_chunk.data_vars.items():
        ncvar = out_vars[name]
        arr = np.asarray(da.transpose(*ncvar.dimensions).values)

        sl = [slice(None)] * arr.ndim
        sl[ncvar.dimensions.index("time")] = slice(t0, t1)

        ncvar[tuple(sl)] = arr

def _copy_root_group(input_file, output_file) -> None:
    """Copy root group (time, lat, lon, attrs) unchanged to output file."""
    with xr.open_dataset(input_file, engine="netcdf4") as root_in:
        xr.Dataset(
            coords={"time": root_in["time"]},
            data_vars={"lat": root_in["lat"], "lon": root_in["lon"]},
            attrs=dict(root_in.attrs),
        ).to_netcdf(output_file, mode="w")

def _stream_mask_and_write(
    nc: Dataset,
    truth: xr.Dataset,
    pred: xr.Dataset,
    landmask_xy: xr.DataArray,
    tchunk: int,
) -> None:
    """Stream over time chunks, apply landmask, and write masked variables."""
    truth_out = _init_group_schema_as_input(_get_or_create_group(nc, "truth"), truth)
    pred_out = _init_group_schema_as_input(_get_or_create_group(nc, "prediction"), pred)
    ntime = truth.sizes["time"]

    for t0 in tqdm(range(0, ntime, tchunk), desc="Mask+write", unit="chunk"):
        t1 = min(t0 + tchunk, ntime)
        s = slice(t0, t1)

        truth_m, pred_m = _mask_chunk(
            truth.isel(time=s).load(),
            pred.isel(time=s).load(),
            landmask_xy.expand_dims(time=t1 - t0)
        )

        _write_vars_chunk(truth_out, truth_m, t0, t1)
        _write_vars_chunk(pred_out, pred_m, t0, t1)


def save_masked_samples(input_file, output_file, tchunk: int = TCHUNK_SIZE) -> None:
    """
    Stream-apply a spatial landmask to truth and prediction samples in a NetCDF file.

    The root group is copied unchanged (including ``time``, ``lat``, and ``lon``)
    to preserve compatibility with existing readers. The ``truth`` and
    ``prediction`` groups are rewritten to contain only masked data variables and
    dimensions (no coordinates), matching the original input format. Processing
    is performed in time chunks to keep memory usage bounded.
    """
    _copy_root_group(input_file, output_file)

    grid = xr.open_dataset(LANDMASK_NC, engine="netcdf4")
    landmask_xy = grid.LANDMASK.rename({"south_north": "y", "west_east": "x"})

    truth = xr.open_dataset(input_file, group="truth", engine="netcdf4")
    pred = xr.open_dataset(input_file, group="prediction", engine="netcdf4")

    print(f"[{get_timestamp()}] Start sample masking with chunk size={tchunk} ...")
    with Dataset(output_file, mode="a") as nc:
        _stream_mask_and_write(nc, truth, pred, landmask_xy, tchunk)

    truth.close()
    pred.close()
    grid.close()

    print(f"[{get_timestamp()}] Masked dataset saved to {output_file}")
