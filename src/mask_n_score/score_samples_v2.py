"""
Scoring and diagnostics utilities for spatiotemporal ensemble prediction datasets.

This module provides end-to-end workflows for evaluating model predictions against
ground truth over time, including:

- computation of per-time-step skill metrics (RMSE, MAE, CORR, CRPS, etc.)
- extraction of top-N worst-performing samples for detailed inspection
- generation of multi-year (e.g. decadal) percentile grids (p90) for selected variables
- efficient multiprocessing support for large time series
- optimized handling of ensemble dimensions to avoid redundant I/O and computation

The implementation builds on lower-level helpers from `process_sample` and is
designed to work with large xarray datasets backed by NetCDF or Dask.

Assumed data layout
-------------------
Truth datasets:
    variables with dims (time, y, x)

Prediction datasets:
    variables with dims (ensemble, time, y, x)
    (or already reduced to (time, y, x))

Key design principles
---------------------
- Ensemble reductions are explicitly controlled via `n_ensemble`
- Expensive operations (ensemble mean, file I/O) are minimized and reused
- All reductions are NaN-aware and spatially consistent
- Multiprocessing is used at the time-step level for scalability
- Variable naming is normalized via `VAR_MAPPING` for downstream plotting

Primary entry points
--------------------
- score_samples:
    Compute metrics, spatial errors, top samples, and N-year p90 grids
    for a single ensemble size.

- score_samples_multi_ensemble:
    Efficiently compute metrics for multiple ensemble sizes in one pass,
    reusing loaded data.

Configuration constants
-----------------------
- TOP_NUM: number of top samples to extract per metric
- N_YEARS: window length (years) for p90 aggregation
- NTH_PERCENTILE: percentile used for 2D grid computation
- DEBUG: enable optional NetCDF output for p90 grids

This module performs no plotting; all outputs are returned as xarray objects
for flexible downstream analysis and visualization.
"""
from __future__ import annotations

import multiprocessing
from functools import partial
from typing import Iterable, Dict, List, Tuple, Callable

import tqdm
import numpy as np
import pandas as pd
import xarray as xr

from .mask_samples import get_timestamp
from .process_sample import (
    open_samples, compute_abs_difference,
    process_sample, process_sample_multi_ensemble,
)


DEBUG = False       # Enable to dump p90 netcdf
TOP_NUM = 5         # Number of top samples to extract
N_YEARS = 10        # Number of years per period
NTH_PERCENTILE = 90 # N-th percentile value to compute 2D grids per period
VAR_MAPPING: Dict[str, str] = {
    "precipitation": "prcp",
    "temperature_2m": "t2m",
    "eastward_wind_10m": "u10m",
    "northward_wind_10m": "v10m",
}


# -----------------------------------------------------------------------------
# Top samples
# -----------------------------------------------------------------------------
def _extract_top_samples(
    truth: xr.Dataset,
    pred: xr.Dataset,
    n_ensemble: int,
    metrics_ds: xr.Dataset,
    metric: str
) -> dict:
    """
    Extracts truth and pred data for selected times based on a given metric,
    computes absolute/squared error, and includes metric values.

    Parameters:
    - truth (xarray.Dataset): The ground truth dataset.
    - pred (xarray.Dataset): The predicted dataset with an ensemble dimension.
    - n_ensemble (int): Number of ensemble members to use when computing prediction statistics.
    - metrics_ds (xarray.Dataset): Dataset containing top date selections.
    - metric (str): The metric to use for selecting top dates (e.g., "RMSE").

    Returns:
    - dict: A dictionary where each variable contains:
      * "sample": xarray.DataArray with dimensions (time, y, x, type) containing:
          - truth: Ground truth values
          - pred: Mean of ensemble predictions
          - error: Absolute or squared error based on metric type
      * "metric_value": xarray.DataArray containing corresponding metric values
                        for each selected time.
    """
    out: dict = {}

    for var in (v for v in truth.data_vars if v in metrics_ds.data_vars):
        # Select top N dates based on metric values
        top = metrics_ds.sel(metric=metric)[var].to_series().nlargest(TOP_NUM)
        times = np.array(top.index, dtype="datetime64[ns]")

        t = truth[var].sel(time=times).load()
        p = (
            pred[var].sel(time=times)
            .isel(ensemble=slice(0, n_ensemble))
            .mean("ensemble", skipna=True)
            .load()
        )

        abs_error = compute_abs_difference(t, p)
        error = abs_error if metric == "MAE" else abs_error ** 2

        out[var] = {
            "metric_value": xr.DataArray(top.values, dims=["time"], coords={"time": times}),
            "sample": xr.concat([t, p, error], dim="type")
                      .assign_coords(type=["truth", "pred", "error"])
        }

    return out


# -----------------------------------------------------------------------------
# P90 grids
# -----------------------------------------------------------------------------
def _window_time_quantile_2d(
    truth: xr.Dataset,
    pred: xr.Dataset,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compute per-variable time-quantile (e.g., p90) 2D fields over a fixed time window.

    For the interval [start, end), this function:
      1) selects the corresponding time slice from `truth` and `pred`
      2) reduces `pred` by ensemble mean if an ``ensemble`` dimension is present
      3) computes the q01-th quantile over the ``time`` dimension for each variable

    The result for each variable is a 2D field (y, x).

    Notes
    -----
    - The slice end is treated as exclusive by using ``end - 1ns``, because
      label-based slicing in xarray is inclusive.
    - ``skipna=True`` ensures NaNs are ignored; grid points with all-NaN
      values over the window will remain NaN.

    Parameters
    ----------
    truth : xr.Dataset
        Dataset with variables shaped (time, y, x).
    pred : xr.Dataset
        Dataset with variables shaped (ensemble, time, y, x) or (time, y, x).
    start : pd.Timestamp
        Inclusive start of the time window.
    end : pd.Timestamp
        Exclusive end of the time window.

    Returns
    -------
    (truth_q, pred_q) : Tuple[xr.Dataset, xr.Dataset]
        Two datasets containing per-variable quantile fields with dims (y, x).
        Variable names match those in `truth`.
    """
    # Use end - 1ns to mimic [start, end) given slice end is
    # inclusive in label-based selection.
    sel_end = end - pd.Timedelta("1ns")
    q01 = NTH_PERCENTILE / 100  # Quantile in [0, 1] (e.g., 0.9 for the 90th percentile).
    var_names = ['prcp', 't2m'] # list(truth.data_vars)

    truth_w = truth[var_names].sel(time=slice(start, sel_end))
    pred_w = pred[var_names].sel(time=slice(start, sel_end))
    if "ensemble" in pred_w.dims:
        pred_w = pred_w.mean("ensemble", skipna=True)

    return (
        truth_w.quantile(q01, "time", skipna=True).squeeze(drop=True),
        pred_w.quantile(q01, "time", skipna=True).squeeze(drop=True)
    )

def p90_by_nyear_period(
    truth: xr.Dataset,
    pred: xr.Dataset,
    n_ensemble: int
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compute two 2D percentile datasets (y, x):
      1) truth_pXX: q-th percentile over time within a years-long window
      2) pred_pXX : q-th percentile over time within the same window, using ensemble mean

    Requirements enforced:
      - returns TWO datasets: (truth_pXX_ds, pred_pXX_ds)
      - uses a time window of `years` (default 10)

    Assumes your layouts:
      truth[var]: (time, y, x)
      pred[var]:  (ensemble, time, y, x)  (or possibly (time, y, x))

    Parameters
    ----------
    truth, pred : xr.Dataset
    n_ensemble : int
        Number of ensemble members to use when computing prediction statistics.

    Returns
    -------
    (truth_p_ds, pred_p_ds) : Tuple[xr.Dataset, xr.Dataset]
        Each dataset has variables for each requested var, dims (y, x).
        Variable names are the same as input variable names.
    """
    if "time" not in truth.dims or "time" not in pred.dims:
        raise ValueError("Both truth and pred must have a 'time' dimension")

    # Slice n_ensemble only
    pred_s = pred.isel(ensemble=slice(0, n_ensemble)) if "ensemble" in pred.dims else pred

    start = pd.to_datetime(truth.time.min().item())
    tmax = pd.to_datetime(truth.time.max().item())

    truth_blocks, pred_blocks = [], []
    labels = []
    while start <= tmax:
        end = start + pd.DateOffset(years=N_YEARS)

        t_blk, p_blk = _window_time_quantile_2d(truth, pred_s, start, end)
        truth_blocks.append(t_blk)
        pred_blocks.append(p_blk)

        labels.append(f"{start.year:04d}-{min(start.year + N_YEARS - 1, tmax.year):04d}")

        start = end

    truth_p90 = xr.concat(truth_blocks, dim=xr.IndexVariable("period", labels))
    pred_p90 = xr.concat(pred_blocks, dim=xr.IndexVariable("period", labels))

    # Optional NetCDF export
    if DEBUG:
        encoding = { v: {"zlib": True, "complevel": 4} for v in truth_p90.data_vars }
        truth_p90.to_netcdf(f"truth_p{NTH_PERCENTILE}.nc", encoding=encoding)
        pred_p90.to_netcdf(f"pred_p{NTH_PERCENTILE}.nc", encoding=encoding)

    return truth_p90, pred_p90


# -----------------------------------------------------------------------------
# Multiprocessing helpers
# -----------------------------------------------------------------------------
def _run_over_time(n_time: int, worker_fn: Callable[[int], dict],
                  pool_size: int = 32) -> List[dict]:
    """Run `worker_fn(time_idx)` over [0..n_time-1] in parallel and collect results."""
    with multiprocessing.Pool(pool_size) as pool:
        return list(
            tqdm.tqdm(
                pool.imap(worker_fn, range(n_time)),
                total=n_time,
            )
        )

def _concat_from_results(results: List[dict], key: str, dim: str) -> xr.Dataset:
    """Concat `results[i][key]` along `dim` and apply VAR_MAPPING rename."""
    return xr.concat([res[key] for res in results], dim=dim).rename(VAR_MAPPING)


# -----------------------------------------------------------------------------
# Scoring entrypoints
# -----------------------------------------------------------------------------
def score_samples(
    filepath: str,
    n_ensemble: int = 1
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, dict]:
    """
    Compute evaluation metrics and diagnostics for all time steps in a sample dataset.

    This function:
      - loads truth and prediction samples from `filepath`
      - processes each time step (optionally in parallel) to compute skill metrics
      - aggregates metrics and spatial errors across time
      - extracts top-N worst cases for selected metrics (e.g., MAE, RMSE)
      - computes decadal (N-year) p90 grids for selected variables

    Parameters
    ----------
    filepath : str
        Path to the dataset file containing truth and prediction samples.
    n_ensemble : int, optional
        Number of ensemble members to use when computing prediction statistics.
        Only the first `n_ensemble` members are used. Default is 1.

    Returns
    -------
    metrics : xr.Dataset
        Time-series dataset of computed evaluation metrics (e.g., RMSE, MAE, CRPS).
    error : xr.Dataset
        Spatial error fields for each time step.
    top_samples : dict
        Dictionary of top-N worst samples per metric and variable, including
        truth, prediction, and error maps.
    flats : tuple of xr.Dataset
        Flattened truth and prediction values (points dimension), useful for
        PDFs and scatter plots.
    p90s : tuple of xr.Dataset
        Tuple of (truth_p90, pred_p90) decadal p90 grids computed over N-year windows.

    Notes
    -----
    - This function may be computationally expensive for long time series or
      large ensemble sizes.
    - For best performance, ensure datasets are properly chunked if using Dask.
    """
    print(f"[{get_timestamp()}] score_samples: filepath={filepath} n_ensemble={n_ensemble}")

    truth, pred, _ = open_samples(filepath)
    results = _run_over_time(truth.sizes["time"],
                partial(process_sample, filepath=filepath, n_ensemble=n_ensemble))

    # Metrics
    metrics = _concat_from_results(results, "metrics", dim="time")
    metrics.attrs["n_ensemble"] = n_ensemble

    # Spatial error & flattened truth/pred
    error = _concat_from_results(results, "error", dim="time")
    flats = (
        _concat_from_results(results, "truth_flat", dim="points"),
        _concat_from_results(results, "pred_flat", dim="points"),
    )

    # Top samples & p90 grids
    truth_m = truth.rename(VAR_MAPPING)
    pred_m = pred.rename(VAR_MAPPING)
    top_samples = {
        m: _extract_top_samples(truth_m, pred_m, n_ensemble, metrics, m)
        for m in ["MAE", "RMSE"]
    }
    p90s = p90_by_nyear_period(truth_m, pred_m, n_ensemble)

    print(f"[{get_timestamp()}] score_samples completed")

    return metrics, error, top_samples, flats, p90s


def score_samples_multi_ensemble(
    filepath: str,
    n_ensembles: Iterable[int]
) -> Dict[int, xr.Dataset]:
    """
    Faster multi-ensemble scoring that avoids re-loading predictions for each n_ensemble.

    Returns
    -------
    Dict[int, xr.Dataset]
        Mapping n_ensemble -> combined metrics dataset (dim="time").
    """
    n_ensembles = tuple(n_ensembles)
    if not n_ensembles:
        raise ValueError("n_ensembles must be non-empty")

    print(
        f"[{get_timestamp()}] score_samples_multi_ensemble: "
        f"filepath={filepath} n_ensembles={n_ensembles}"
    )

    truth, _, _ = open_samples(filepath)
    results = _run_over_time(truth.sizes["time"],
                partial(process_sample_multi_ensemble, filepath=filepath, n_ensembles=n_ensembles))

    combined: Dict[int, xr.Dataset] = {}
    for n_ens in n_ensembles:
        # Reuse _concat_from_results without changing signature
        wrapped = [{"metrics": res["metrics_by_n"][n_ens]} for res in results]
        ds = _concat_from_results(wrapped, "metrics", dim="time")
        ds.attrs["n_ensemble"] = n_ens
        combined[n_ens] = ds

    print(f"[{get_timestamp()}] score_samples_multi_ensemble completed")

    return combined
