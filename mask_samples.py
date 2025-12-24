"""
Apply a 2D landmask to sample datasets stored in a grouped NetCDF, and write
masked outputs in fixed-length time blocks (typically one year).

This module reads an input NetCDF with a root group and at least two subgroups,
``truth`` and ``prediction``. It applies a static landmask defined on ``(y, x)``
to every variable in those groups and writes one output file per time block.

Output file naming:
- Files are written as ``{output_prefix}_{YYYY}.nc`` where ``YYYY`` is taken from
  the first timestep in the block (via the root ``time`` coordinate).

Output layout (matches the input "dims at root, vars in groups" style):
- Root group contains:
  - Dimensions: ``time``, ``y``, ``x``, and ``ensemble`` (ensemble size is taken
    from the input ``prediction`` group; defaults to 1 if missing).
  - Variables: ``time(time)``, ``lat(y, x)``, ``lon(y, x)``, plus global attrs.
- Subgroups ``truth`` and ``prediction`` contain only data variables.
  - No coordinates are created inside groups.
  - Variables reference the *root* dimensions by name (no group-local dims),
    which avoids ncview dimension-resolution issues with duplicated dim names.

Processing characteristics:
- The module processes one time block at a time, loading that block into memory,
  masking it, and writing it to a single output file in one pass.
- This is designed for bounded memory usage while keeping the on-disk structure
  compatible with existing readers and visualization tools (e.g., ncview).

Assumptions:
- Truth variables have dimensions ``(time, y, x)``.
- Prediction variables have dimensions ``(ensemble, time, y, x)`` or
  ``(time, y, x)``.
- The landmask is a static 2D field on ``(y, x)``, where mask==1 indicates
  points to keep (others are set to missing via ``where``).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr
from netCDF4 import Dataset, Variable

LANDMASK_NC = "./data/ssp_208x208_grid_coords.nc"


def get_timestamp() -> str:
    """Return a human-readable timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _get_or_create_group(nc: Dataset, name: str) -> Dataset:
    """Return an existing netCDF group, or create it if missing."""
    return nc.groups[name] if name in nc.groups else nc.createGroup(name)

def _init_group_vars_only(group: Dataset, src: xr.Dataset) -> dict[str, Variable]:
    """
    Create variables inside the group that reference root dimensions by name.
    Do NOT create any group-local dimensions (to match input layout).
    """
    out: dict[str, Variable] = {}
    for name, da in src.data_vars.items():
        dims = tuple(da.dims)  # e.g. ('ensemble','time','y','x') or ('time','y','x')
        dtype = np.dtype(da.dtype)
        if name not in group.variables:
            group.createVariable(name, dtype, dims, zlib=True, complevel=1)
        out[name] = group.variables[name]
    return out


def _mask_block(
    truth_t: xr.Dataset,
    pred_t: xr.Dataset,
    landmask_xy: xr.DataArray,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Apply landmask to chunk; expand mask for ensemble if needed."""
    # landmask_xy is (y,x); xarray will broadcast over time/ensemble automatically
    truth_masked = truth_t.where(landmask_xy == 1)

    lm_pred = landmask_xy
    if "ensemble" in pred_t.dims:
        lm_pred = lm_pred.expand_dims(ensemble=pred_t.sizes["ensemble"])
    pred_masked = pred_t.where(lm_pred == 1)

    return truth_masked, pred_masked

def _write_vars_full(out_vars: dict[str, Variable], ds: xr.Dataset) -> None:
    """Write entire ds into variables (no slicing)."""
    for name, da in ds.data_vars.items():
        ncvar = out_vars[name]
        arr = np.asarray(da.transpose(*ncvar.dimensions).values)
        ncvar[...] = arr


def _create_root(
    input_file: str,
    output_file: str,
    time_sel: xr.DataArray,
    y_size: int,
    x_size: int,
    ensemble_size: int,
) -> None:
    """Create output root group matching input style; explicitly preserve time units/calendar."""
    # Read lat/lon values + attrs via xarray
    with xr.open_dataset(input_file, engine="netcdf4") as root_in:
        lat_vals = np.asarray(root_in["lat"].values)
        lon_vals = np.asarray(root_in["lon"].values)
        lat_attrs = dict(root_in["lat"].attrs)
        lon_attrs = dict(root_in["lon"].attrs)
        global_attrs = dict(root_in.attrs)

    # Read time units/calendar robustly via netCDF4 (xarray attrs can be empty)
    with Dataset(input_file, "r") as src_nc:
        src_time = src_nc.variables["time"]
        time_units = getattr(src_time, "units", None)
        time_calendar = getattr(src_time, "calendar", None)
        # (optional) copy any other time attrs present in the source
        other_time_attrs = {
            a: getattr(src_time, a)
            for a in src_time.ncattrs()
            if a not in {"units", "calendar", "_FillValue"}
        }

    if not time_units or "since" not in time_units:
        raise ValueError(f"Unsupported or missing time units in input: {time_units!r}")

    # Parse base datetime from units: "days since YYYY-MM-DD HH:MM:SS"
    base_str = time_units.split("since", 1)[1].strip()
    # Accept both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            base_dt = datetime.strptime(base_str, fmt)
            break
        except ValueError:
            base_dt = None
    if base_dt is None:
        raise ValueError(f"Could not parse base time from units: {time_units!r}")

    # Compute numeric day offsets for this block
    # time_sel is datetime-like; convert to numpy datetime64[ns]
    tvals = np.asarray(time_sel.values).astype("datetime64[ns]")
    base64 = np.datetime64(base_dt, "ns")
    # integer days since base
    time_numeric = ((tvals - base64) / np.timedelta64(1, "D")).astype("int64")

    with Dataset(output_file, "w", format="NETCDF4") as nc:
        # Root dims (match input style)
        nc.createDimension("time", time_sel.sizes["time"])  # fixed-length per split file is OK
        nc.createDimension("y", y_size)
        nc.createDimension("x", x_size)
        nc.createDimension("ensemble", ensemble_size)

        # Root vars
        latv = nc.createVariable("lat", "f4", ("y", "x"))
        lonv = nc.createVariable("lon", "f4", ("y", "x"))
        timev = nc.createVariable("time", "i8", ("time",))

        # Copy lat/lon attrs
        if lat_attrs:
            latv.setncatts(lat_attrs)
        if lon_attrs:
            lonv.setncatts(lon_attrs)

        # Ensure time attrs exist
        if time_units is not None:
            timev.setncattr("units", time_units)
        if time_calendar is not None:
            timev.setncattr("calendar", time_calendar)
        if other_time_attrs:
            timev.setncatts(other_time_attrs)

        # Write data
        latv[:, :] = lat_vals
        lonv[:, :] = lon_vals
        timev[:] = time_numeric

        # Copy global attrs
        if global_attrs:
            nc.setncatts(global_attrs)


def save_masked_samples_per_year(
    input_file: str,
    output_prefix: str,
    chunk_size: int = 365,
) -> None:
    # Landmask
    grid = xr.open_dataset(LANDMASK_NC, engine="netcdf4")
    landmask_xy = grid.LANDMASK.rename({"south_north": "y", "west_east": "x"})

    # Read root time + sizes once
    with xr.open_dataset(input_file, engine="netcdf4") as root_in:
        time_all = root_in["time"]
        ntime = time_all.sizes["time"]
        y_size = root_in.sizes["y"]
        x_size = root_in.sizes["x"]

    # Read ensemble size
    with xr.open_dataset(input_file, group="prediction", engine="netcdf4") as pred_in:
        ensemble_size = pred_in.sizes.get("ensemble", 1)

    print(f"[{get_timestamp()}] Start masking {chunk_size} samples per file:"
          f" ntime={ntime}, ensemble={ensemble_size}")

    for t0 in range(0, ntime, chunk_size):
        t1 = min(t0 + chunk_size, ntime)
        time_sel = time_all.isel(time=slice(t0, t1))

        year = int(time_sel.dt.year.isel(time=0).item())
        out_file = f"{output_prefix}_{year}.nc"
        print(f"[{get_timestamp()}] Writing block time[{t0}:{t1}] -> {out_file}")

        # 1) Create root group like input (dims at root + lat/lon/time)
        _create_root(
            input_file=input_file,
            output_file=out_file,
            time_sel=time_sel,
            y_size=y_size,
            x_size=x_size,
            ensemble_size=ensemble_size,
        )

        # 2) Read truth/pred block (keep all ensembles)
        truth_src = xr.open_dataset(input_file, group="truth", engine="netcdf4")
        pred_src  = xr.open_dataset(input_file, group="prediction", engine="netcdf4")
        try:
            truth_blk = truth_src.isel(time=slice(t0, t1)).load()
            pred_blk  = pred_src.isel(time=slice(t0, t1)).load()
        finally:
            truth_src.close()
            pred_src.close()

        # 3) Mask
        truth_m, pred_m = _mask_block(truth_blk, pred_blk, landmask_xy)

        # 4) Write groups: vars only, referencing root dims
        with Dataset(out_file, mode="a") as nc:
            truth_out = _init_group_vars_only(_get_or_create_group(nc, "truth"), truth_m)
            pred_out  = _init_group_vars_only(_get_or_create_group(nc, "prediction"), pred_m)

            _write_vars_full(truth_out, truth_m)
            _write_vars_full(pred_out, pred_m)

    grid.close()
    print(f"[{get_timestamp()}] Done. Files written with prefix: {output_prefix}")
