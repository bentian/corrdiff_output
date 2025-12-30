"""
Utilities for computing evaluation metrics and diagnostics on spatiotemporal
ensemble prediction datasets using xarray and xskillscore.

This module provides low-level building blocks to:
- compute common skill metrics (RMSE, MAE, CORR, CRPS, ensemble spread)
- handle NaN-safe reductions over space, time, and ensemble dimensions
- extract flattened truth/prediction pairs for distributional analysis
- process individual time steps (optionally for multiple ensemble sizes)
- support efficient multiprocessing workflows

Assumed data layout
-------------------
Truth datasets:
    variables with dims (time, y, x)

Prediction datasets:
    variables with dims (ensemble, time, y, x)
    (or already reduced to (time, y, x))

Key design notes
----------------
- Ensemble reductions always use explicit `skipna=True` for robustness.
- Expensive I/O and reductions are structured to minimize repeated work
  in multiprocessing contexts.
- All metric computations are NaN-aware and spatially consistent.

Dependencies
------------
- numpy
- xarray
- xskillscore

This module does not perform any plotting or file writing; it focuses
purely on computation and data transformation.
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import xarray as xr

try:
    import xskillscore as xs
except ImportError as exc:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`") from exc


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
    truth_t, pred_t = _select_time_and_ensemble(filepath, index, n_ensemble)

    truth_flat, pred_flat = _flatten_and_filter_nan(truth_t, pred_t)

    return {
        "metrics": _compute_metrics(truth_t, pred_t),
        "error": compute_abs_difference(truth_t, pred_t),
        "truth_flat": truth_flat,
        "pred_flat": pred_flat,
    }


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
    truth_t, pred_t = _select_time_and_ensemble(filepath, index, max(n_ensembles))

    # Compute per-ensemble metrics by slicing in-memory
    out: Dict[int, xr.Dataset] = {}
    for n_ens in n_ensembles:
        pred_n = pred_t.isel(ensemble=slice(0, n_ens)) if "ensemble" in pred_t.dims else pred_t
        out[n_ens] = _compute_metrics(truth_t, pred_n)

    return {"metrics_by_n": out}

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
