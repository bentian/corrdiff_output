"""
Module for processing and scoring dataset samples using xarray and xskillscore.

Utilities included:
- Open NetCDF dataset samples (truth and predictions).
- Compute evaluation metrics (RMSE, MAE, CORR, CRPS, STD_DEV).
- Extract top-ranked samples based on selected metrics.
- Flatten and filter NaN values for pointwise exports.
- Multiprocessing helpers for running over time indices.

Notes on performance:
- `score_samples()` computes full outputs (metrics + error + flattened exports + top samples).
- `score_samples_metrics_multi_ensemble()` is specialized for plotting metrics vs ensembles:
  it computes *metrics only* for multiple ensemble sizes in a single pass, avoiding
  flattening/error computation and avoiding re-running the whole pipeline per ensemble.

Dependencies:
    multiprocessing, warnings, numpy, tqdm, xarray, xskillscore
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


VAR_MAPPING: Dict[str, str] = {
    "precipitation": "prcp",
    "temperature_2m": "t2m",
    "eastward_wind_10m": "u10m",
    "northward_wind_10m": "v10m",
}
N_YEARS = 10

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def open_samples(f: str) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
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
def compute_crps(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
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


def compute_metrics(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
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
        crps = compute_crps(truth, pred)

    return (
        xr.concat([rmse, mae, corr, crps, std_dev], dim="metric")
        .assign_coords(metric=["RMSE", "MAE", "CORR", "CRPS", "STD_DEV"])
        .load()
    )


def flatten_and_filter_nan(truth: xr.Dataset, pred: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
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


def compute_abs_difference(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
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
def run_over_time(n_time: int, worker_fn: Callable[[int], dict],
                  pool_size: int = 32) -> List[dict]:
    """Run `worker_fn(time_idx)` over [0..n_time-1] in parallel and collect results."""
    with multiprocessing.Pool(pool_size) as pool:
        return list(
            tqdm.tqdm(
                pool.imap(worker_fn, range(n_time)),
                total=n_time,
            )
        )


def concat_from_results(results: List[dict], key: str, dim: str) -> xr.Dataset:
    """Concat `results[i][key]` along `dim` and apply VAR_MAPPING rename."""
    return xr.concat([res[key] for res in results], dim=dim).rename(VAR_MAPPING)


# -----------------------------------------------------------------------------
# Per-time processing
# -----------------------------------------------------------------------------
def select_time_and_ensemble(
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
    truth, pred, _ = open_samples(filepath)

    truth_t = truth.isel(time=index).load()

    pred_t = pred.isel(time=index)
    if n_ensemble > 0 and "ensemble" in pred_t.dims:
        pred_t = pred_t.isel(ensemble=slice(0, n_ensemble))
    pred_t = pred_t.load()

    return truth_t, pred_t


def process_sample(index: int, filepath: str, n_ensemble: int) -> Dict[str, xr.Dataset]:
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
    truth_t, pred_t = select_time_and_ensemble(filepath, index, n_ensemble)

    truth_flat, pred_flat = flatten_and_filter_nan(truth_t, pred_t)

    return {
        "metrics": compute_metrics(truth_t, pred_t),
        "error": compute_abs_difference(truth_t, pred_t),
        "truth_flat": truth_flat,
        "pred_flat": pred_flat,
    }


# -----------------------------------------------------------------------------
# Top samples
# -----------------------------------------------------------------------------
def extract_top_samples(
    truth: xr.Dataset,
    pred: xr.Dataset,
    combined_metrics: xr.Dataset,
    metric: str,
    top_num: int = 5
) -> dict:
    """
    Extracts truth and pred data for selected times based on a given metric,
    computes absolute/squared error, and includes metric values.

    Parameters:
    - truth (xarray.Dataset): The ground truth dataset.
    - pred (xarray.Dataset): The predicted dataset with an ensemble dimension.
    - combined_metrics (xarray.Dataset): Dataset containing top date selections.
    - metric (str): The metric to use for selecting top dates (e.g., "RMSE").
    - top_num (int): Number of top dates to select (default: 5).

    Returns:
    - dict: A dictionary where each variable contains:
      * "sample": xarray.DataArray with dimensions (time, y, x, type) containing:
          - truth: Ground truth values
          - pred: Mean of ensemble predictions
          - error: Absolute or squared error based on metric type
      * "metric_value": xarray.DataArray containing corresponding metric values
                        for each selected time.
    """
    data_vars: dict = {}

    for var in truth.data_vars:
        if var not in combined_metrics:
            continue

        # Select top N dates based on metric values
        top_dates = combined_metrics.sel(metric=metric)[var].to_series().nlargest(top_num)
        selected_times = np.array(top_dates.index, dtype="datetime64[ns]")

        truth_selected = truth[var].sel(time=selected_times).load()
        pred_selected = pred[var].sel(time=selected_times).mean("ensemble").load()

        abs_error = compute_abs_difference(truth_selected, pred_selected)
        error = abs_error if metric == "MAE" else abs_error ** 2

        data_vars[var] = {
            "sample": xr.concat([truth_selected, pred_selected, error], dim="type")
                      .assign_coords(type=["truth", "pred", "error"]),
            "metric_value": xr.DataArray(top_dates.values, dims=["time"],
                                         coords={"time": selected_times})
        }

    return data_vars


# -----------------------------------------------------------------------------
# p90 grids
# -----------------------------------------------------------------------------
def time_quantile_2d(ds: xr.Dataset, var_names, q01: float) -> xr.Dataset:
    """Return per-variable q-th quantile over time as 2D (y, x) fields."""
    return ds[var_names].quantile(q01, "time", skipna=True).squeeze(drop=True)

def p90_by_nyear_period(
    truth: xr.Dataset,
    pred: xr.Dataset,
    n_years: int = N_YEARS,
    q: float = 90.0,
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
    n_years : int
        Window length in years (default 10).
    q : float
        Percentile in [0, 100] (default 90).

    Returns
    -------
    (truth_p_ds, pred_p_ds) : Tuple[xr.Dataset, xr.Dataset]
        Each dataset has variables for each requested var, dims (y, x).
        Variable names are the same as input variable names.
    """
    if "time" not in truth.dims or "time" not in pred.dims:
        raise ValueError("Both truth and pred must have a 'time' dimension")

    var_names = list(truth.data_vars)
    q01 = q / 100.0

    truth_blocks: List[xr.Dataset] = []
    pred_blocks: List[xr.Dataset] = []
    labels: List[str] = []

    start = pd.to_datetime(truth.time.min().item())
    tmax = pd.to_datetime(truth.time.max().item())
    while start <= tmax:
        end = start + pd.DateOffset(years=n_years)
        # Use end - 1ns to mimic [start, end) given slice end is
        # inclusive in label-based selection.
        sel_end = end - pd.Timedelta("1ns")

        truth_w = truth.sel(time=slice(start, sel_end))
        pred_w = pred.sel(time=slice(start, sel_end)).map(
            # always use ensemble mean if present
            lambda da: da.mean("ensemble", skipna=True) if "ensemble" in da.dims else da
        )

        truth_blocks.append(time_quantile_2d(truth_w, var_names, q01))
        pred_blocks.append(time_quantile_2d(pred_w, var_names, q01))

        label_end_year = min(start.year + n_years - 1, tmax.year)
        labels.append(f"{start.year:04d}-{label_end_year:04d}")

        start = end

    truth_p90 = xr.concat(truth_blocks, dim=xr.IndexVariable("period", labels))
    pred_p90 = xr.concat(pred_blocks, dim=xr.IndexVariable("period", labels))

    # Optional NetCDF export
    # encoding = {
    #     v: {"zlib": True, "complevel": 4}
    #     for v in truth_p90.data_vars
    # }
    # truth_p90.to_netcdf(f"truth_p{int(q)}.nc", encoding=encoding)
    # pred_p90.to_netcdf(f"pred_p{int(q)}.nc", encoding=encoding)

    return truth_p90, pred_p90

# -----------------------------------------------------------------------------
# Scoring entrypoints
# -----------------------------------------------------------------------------
def score_samples(
    filepath: str,
    n_ensemble: int = 1
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, dict]:
    """
    Score the dataset by computing various metrics over all time steps.

    Parameters:
        filepath (str): Path to the dataset file.
        n_ensemble (int, optional): Number of ensemble members. Defaults to 1.

    Returns:
        Tuple:
            - xr.Dataset: Computed metrics across all time steps.
            - xr.Dataset: Spatial error dataset.
            - xr.Dataset: Flattened truth dataset.
            - xr.Dataset: Flattened prediction dataset.
            - dict: Dictionary containing datasets for the top N samples
                    with the highest values for selected metrics (e.g., RMSE and MAE).
    """
    print(f"[{get_timestamp()}] score_samples: filepath={filepath} n_ensemble={n_ensemble}")

    truth, pred, _ = open_samples(filepath)
    results = run_over_time(truth.sizes["time"],
                partial(process_sample, filepath=filepath, n_ensemble=n_ensemble))

    metrics = concat_from_results(results, "metrics", dim="time")
    metrics.attrs["n_ensemble"] = n_ensemble

    error = concat_from_results(results, "error", dim="time")
    flats = (
        concat_from_results(results, "truth_flat", dim="points"),
        concat_from_results(results, "pred_flat", dim="points"),
    )

    # Top samples & p90 grids
    truth_m = truth.rename(VAR_MAPPING)
    pred_m = pred.rename(VAR_MAPPING)
    top_samples = {m: extract_top_samples(truth_m, pred_m, metrics, m) for m in ["MAE", "RMSE"]}
    p90s = p90_by_nyear_period(truth_m, pred_m)

    print(f"[{get_timestamp()}] score_samples completed")

    return metrics, error, top_samples, flats, p90s


def process_sample_multi_ensemble(
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
    truth_t, pred_t = select_time_and_ensemble(filepath, index, max(n_ensembles))

    # Compute per-ensemble metrics by slicing in-memory
    out: Dict[int, xr.Dataset] = {}
    for n_ens in n_ensembles:
        pred_n = pred_t.isel(ensemble=slice(0, n_ens)) if "ensemble" in pred_t.dims else pred_t
        out[n_ens] = compute_metrics(truth_t, pred_n)

    return {"metrics_by_n": out}


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
    results = run_over_time(truth.sizes["time"],
                partial(process_sample_multi_ensemble, filepath=filepath, n_ensembles=n_ensembles))

    combined: Dict[int, xr.Dataset] = {}
    for n_ens in n_ensembles:
        # Reuse concat_from_results without changing signature
        wrapped = [{"metrics": res["metrics_by_n"][n_ens]} for res in results]
        ds = concat_from_results(wrapped, "metrics", dim="time")
        ds.attrs["n_ensemble"] = n_ens
        combined[n_ens] = ds

    print(f"[{get_timestamp()}] score_samples_multi_ensemble completed")

    return combined
