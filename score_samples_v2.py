"""
Module for processing and scoring dataset samples using xarray and xskillscore.

Utilities included:
- Open NetCDF dataset samples (truth and predictions).
- Compute evaluation metrics (RMSE, MAE, CORR, CRPS, STD_DEV).
- Extract top-ranked samples based on selected metrics.
- Flatten and filter NaN values for pointwise exports.
- Multiprocessing helpers for running over time indices.

Notes on performance:
- `score_samples()` computes full outputs
  (metrics + error + flattened exports + top samples + p90 grids).
- `score_samples_multi_ensemble()` is specialized for plotting metrics vs ensembles:
  it computes *metrics only* for multiple ensemble sizes in a single pass, avoiding
  flattening/error computation and avoiding re-running the whole pipeline per ensemble.
"""
from __future__ import annotations

import multiprocessing
import warnings
from functools import partial
from typing import Iterable, Dict, List, Tuple, Callable

import tqdm
import numpy as np
import pandas as pd
import xarray as xr

from mask_samples import get_timestamp

try:
    import xskillscore as xs
except ImportError as exc:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`") from exc

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
# IO
# -----------------------------------------------------------------------------
def _open_samples(f: str) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Open prediction and truth samples from a dataset file.

    Parameters:
        f: Path to the dataset file.

    Returns:
        tuple: A tuple containing truth, prediction, and root datasets.
    """
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group="prediction").merge(root)
    truth = xr.open_dataset(f, group="truth").merge(root)

    return truth.set_coords(["lon", "lat"]), pred.set_coords(["lon", "lat"]), root


# -----------------------------------------------------------------------------
# Metrics / transforms
# -----------------------------------------------------------------------------
def _compute_crps(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """
    Computes the CRPS while filtering out NaN values.

    Parameters:
        truth (xr.Dataset): The truth dataset.
        pred (xr.Dataset): The prediction dataset.

    Returns:
        xr.Dataset: CRPS values computed only for valid points.
    """
    dim = ["x", "y"]

    valid_mask = np.isfinite(truth) & pred.notnull().any(dim="ensemble")
    truth_valid = truth.where(valid_mask)
    pred_valid = pred.where(valid_mask)

    return xs.crps_ensemble(truth_valid, pred_valid, member_dim="ensemble", dim=dim)


def _compute_metrics(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """
    Compute RMSE, MAE, CORR, CRPS, and STD_DEV between truth and prediction ensembles.

    Parameters:
        truth (xr.Dataset): The truth dataset.
        pred (xr.Dataset): The prediction dataset.

    Returns:
        xr.Dataset: A dataset containing computed metrics.
    """
    dim = ["x", "y"]
    pred_mean = pred.mean("ensemble")

    rmse = xs.rmse(truth, pred_mean, dim=dim, skipna=True)
    mae = xs.mae(truth, pred_mean, dim=dim, skipna=True)
    corr = xs.pearson_r(truth, pred_mean, dim=dim, skipna=True)

    # Compute CRPS and STD_DEV (suppress expected NaN-related warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std_dev = pred.std("ensemble").mean(dim, skipna=True)
        crps = _compute_crps(truth, pred)

    return (
        xr.concat([rmse, mae, corr, crps, std_dev], dim="metric")
        .assign_coords(metric=["RMSE", "MAE", "CORR", "CRPS", "STD_DEV"])
        .load()
    )


def _flatten_and_filter_nan(truth: xr.Dataset, pred: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Extract flattened truth and prediction for all variables in the truth dataset.

    Parameters:
        truth (xarray.Dataset): The truth dataset.
        pred (xarray.Dataset): The prediction dataset.

    Returns:
        tuple: Two xarray.Datasets, one for truth and one for prediction.
    """
    truth_data: Dict[str, Tuple[str, np.ndarray]] = {}
    pred_data: Dict[str, Tuple[str, np.ndarray]] = {}

    for var in truth.data_vars:
        if var not in pred:
            continue

        truth_flat = truth[var].values.ravel()
        pred_flat = pred[var].mean("ensemble").values.ravel() \
                    if "ensemble" in pred[var].dims else pred[var].values.ravel()

        # Filter out NaNs and align truth and pred
        valid = np.isfinite(truth_flat) & np.isfinite(pred_flat)
        truth_data[var] = ("points", truth_flat[valid])
        pred_data[var] = ("points", pred_flat[valid])

    n_points = len(next(iter(truth_data.values()))[1])
    coords = {"points": np.arange(n_points)}

    return xr.Dataset(truth_data, coords=coords), xr.Dataset(pred_data, coords=coords)


def _compute_abs_difference(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """
    Computes the absolute difference between truth and prediction datasets
    while filtering NaN values.

    Parameters:
        truth (xr.Dataset): The truth dataset.
        pred (xr.Dataset): The prediction dataset.

    Returns:
        xr.Dataset: The absolute difference dataset with NaNs removed.
    """
    pred_mean = pred.mean("ensemble").expand_dims("ensemble") \
                if "ensemble" in pred.dims else pred
    abs_diff = abs(pred_mean - truth)

    # Filter out NaNs by setting them to NaN where either truth or pred is NaN
    valid_mask = np.isfinite(truth) & np.isfinite(pred_mean)
    return abs_diff.where(valid_mask)


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
# Per-time processing
# -----------------------------------------------------------------------------
def _select_time_and_ensemble(
    filepath: str,
    index: int,
    n_ensemble: int
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Helper to open samples and select a single time step with optional ensemble slicing.
    Returns (truth_t, pred_t) loaded into memory.

    Parameters:
        filepath (str): Path to the dataset file.
        index (int): The time index to process.
        n_ensemble (int): Number of ensemble members to consider.

    Returns:
        Dict[str, xr.Dataset]: A dictionary containing computed metrics, errors,
        and flattened truth/prediction datasets.
    """
    truth, pred, _ = _open_samples(filepath)

    truth_t = truth.isel(time=index).load()

    pred_t = pred.isel(time=index)
    if n_ensemble > 0 and "ensemble" in pred_t.dims:
        pred_t = pred_t.isel(ensemble=slice(0, n_ensemble))
    pred_t = pred_t.load()

    return truth_t, pred_t


def _process_sample(index: int, filepath: str, n_ensemble: int) -> Dict[str, xr.Dataset]:
    """
    Process a single time step from the dataset.

    Parameters:
        index (int): The time index to process.
        filepath (str): Path to the dataset file.
        n_ensemble (int): Number of ensemble members to consider.

    Returns:
        Dict[str, xr.Dataset]: A dictionary containing computed metrics, errors,
        and flattened truth/prediction datasets.
    """
    truth_t, pred_t = _select_time_and_ensemble(filepath, index, n_ensemble)

    truth_flat, pred_flat = _flatten_and_filter_nan(truth_t, pred_t)

    return {
        "metrics": _compute_metrics(truth_t, pred_t),
        "error": _compute_abs_difference(truth_t, pred_t),
        "truth_flat": truth_flat,
        "pred_flat": pred_flat,
    }


def _process_sample_multi_ensemble(
    index: int,
    filepath: str,
    n_ensembles: Tuple[int, ...]
) -> Dict[int, xr.Dataset]:
    """
    Compute metrics for multiple ensemble sizes at a single time index, efficiently.

    Strategy:
      - open samples once
      - load truth at time=index once
      - load prediction at time=index for ensemble[:max_n] once
      - compute metrics for each n_ens by slicing the already-loaded pred

    Parameters
    ----------
    index : int
        Time index to process.
    filepath : str
        Path to NetCDF samples file.
    n_ensembles : Tuple[int, ...]
        Ensemble sizes to evaluate (e.g., (1, 4, 16)).

    Returns
    -------
    Dict[int, xr.Dataset]
        Mapping n_ens -> metrics dataset for this single time step.
    """
    # Load truth & pred with max ensemble
    truth_t, pred_t = _select_time_and_ensemble(filepath, index, max(n_ensembles))

    # Compute per-ensemble metrics by slicing in-memory
    out: Dict[int, xr.Dataset] = {}
    for n_ens in n_ensembles:
        pred_n = pred_t.isel(ensemble=slice(0, n_ens)) if "ensemble" in pred_t.dims else pred_t
        out[n_ens] = _compute_metrics(truth_t, pred_n)

    return {"metrics_by_n": out}


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

        abs_error = _compute_abs_difference(t, p)
        error = abs_error if metric == "MAE" else abs_error ** 2

        out[var] = {
            "metric_value": xr.DataArray(top.values, dims=["time"], coords={"time": times}),
            "sample": xr.concat([t, p, error], dim="type")
                      .assign_coords(type=["truth", "pred", "error"])
        }

    return out


# -----------------------------------------------------------------------------
# p90 grids
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

    truth, pred, _ = _open_samples(filepath)
    results = _run_over_time(truth.sizes["time"],
                partial(_process_sample, filepath=filepath, n_ensemble=n_ensemble))

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

    truth, _, _ = _open_samples(filepath)
    results = _run_over_time(truth.sizes["time"],
                partial(_process_sample_multi_ensemble, filepath=filepath, n_ensembles=n_ensembles))

    combined: Dict[int, xr.Dataset] = {}
    for n_ens in n_ensembles:
        # Reuse _concat_from_results without changing signature
        wrapped = [{"metrics": res["metrics_by_n"][n_ens]} for res in results]
        ds = _concat_from_results(wrapped, "metrics", dim="time")
        ds.attrs["n_ensemble"] = n_ens
        combined[n_ens] = ds

    print(f"[{get_timestamp()}] score_samples_multi_ensemble completed")

    return combined
