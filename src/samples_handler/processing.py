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
from functools import partial

import numpy as np
import xarray as xr

try:
    import xskillscore as xs
except ImportError as exc:
    raise ImportError(
        "xskillscore not installed. Try `pip install xskillscore`"
    ) from exc


# -----------------------------------------------------------------------------
# Metrics / transforms
# -----------------------------------------------------------------------------
def _metric_dataset(items: Dict[str, xr.Dataset]) -> xr.Dataset:
    """Return a dataset with metrics concatenated along the 'metric' dimension."""
    return (
        xr.concat(items.values(), dim="metric").assign_coords(metric=list(items)).load()
    )


def _mask_valid_points(
    truth: xr.Dataset, pred: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    """Mask locations where truth is invalid or all ensemble members are NaN."""
    valid = np.isfinite(truth) & pred.notnull().all("ensemble")
    return truth.where(valid), pred.where(valid)


def _compute_crps(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """Computes the CRPS while filtering out NaN values."""
    truth_valid, pred_valid = _mask_valid_points(truth, pred)
    return xs.crps_ensemble(
        truth_valid, pred_valid, member_dim="ensemble", dim=["x", "y"]
    )


def _compute_rank_histogram(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """Computes the rank histogram while filtering out NaN values."""
    truth_valid, pred_valid = _mask_valid_points(truth, pred)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return xs.rank_histogram(
            truth_valid, pred_valid, member_dim="ensemble", dim=["x", "y"]
        )


def _compute_metrics(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """
    Compute RMSE, MAE, CORR, CRPS, STD_DEV, and SSR between truth and prediction ensembles.

    Parameters:
        truth (xr.Dataset): The truth dataset.
        pred (xr.Dataset): The prediction dataset.

    Returns:
        xr.Dataset: A dataset containing computed metrics.
    """
    dim = ["x", "y"]
    pred_mean = pred.mean("ensemble")

    with warnings.catch_warnings():
        # Suppress expected NaN-related warnings
        warnings.simplefilter("ignore", RuntimeWarning)

        rmse = xs.rmse(truth, pred_mean, dim=dim, skipna=True)
        std_dev = pred.std("ensemble").mean(dim, skipna=True)

        return _metric_dataset(
            {
                "RMSE": rmse,
                "MAE": xs.mae(truth, pred_mean, dim=dim, skipna=True),
                "CORR": xs.pearson_r(truth, pred_mean, dim=dim, skipna=True),
                "CRPS": _compute_crps(truth, pred),
                "STD_DEV": std_dev,
                "SSR": std_dev / rmse,
            }
        )


def _flatten_and_filter_nan(
    truth: xr.Dataset, pred: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
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
    coords = {}

    for var in truth.data_vars:
        if var not in pred:
            continue

        truth_flat = truth[var].values.ravel()
        pred_flat = (
            pred[var].mean("ensemble").values.ravel()
            if "ensemble" in pred[var].dims
            else pred[var].values.ravel()
        )

        # Filter out NaNs and align truth and pred
        valid = np.isfinite(truth_flat) & np.isfinite(pred_flat)

        # Use per-variable dimension to avoid mismatched non-NaN sample sizes (e.g., BCSD data)
        dim = f"points_{var}"
        truth_data[var] = (dim, truth_flat[valid])
        pred_data[var] = (dim, pred_flat[valid])
        coords[dim] = np.arange(valid.sum())

    return {
        "truth_flat": xr.Dataset(truth_data, coords=coords),
        "pred_flat": xr.Dataset(pred_data, coords=coords),
    }


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
    pred_mean = (
        pred.mean("ensemble").expand_dims("ensemble")
        if "ensemble" in pred.dims
        else pred
    )
    abs_diff = abs(pred_mean - truth)

    # Filter out NaNs by setting them to NaN where either truth or pred is NaN
    valid_mask = np.isfinite(truth) & np.isfinite(pred_mean)
    return abs_diff.where(valid_mask)


# -----------------------------------------------------------------------------
# Per-time processing
# -----------------------------------------------------------------------------
def _select_time_and_ensemble(
    filepath: str, index: int, n_ensemble: int
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
    return {
        "metrics": _compute_metrics(truth_t, pred_t),
        "rank_histogram": _compute_rank_histogram(truth_t, pred_t),
        "error": compute_abs_difference(truth_t, pred_t),
        **_flatten_and_filter_nan(truth_t, pred_t),
    }


def process_sample_multi_ensemble(
    index: int, filepath: str, n_ensembles: Tuple[int, ...]
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
        pred_n = (
            pred_t.isel(ensemble=slice(0, n_ens))
            if "ensemble" in pred_t.dims
            else pred_t
        )
        out[n_ens] = _compute_metrics(truth_t, pred_n)

    return {"metrics_by_n": out}


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def _center_crop(ds: xr.Dataset, size: tuple[int, int]) -> xr.Dataset:
    """
    Center crop the dataset to a specified (y, x) size.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to crop.
    size : tuple[int, int]
        (size_y, size_x) to crop.

    Returns
    -------
    xr.Dataset
        The center-cropped dataset.
    """
    if "y" not in ds.dims or "x" not in ds.dims:
        return ds

    size_y, size_x = size
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    if size_y <= 0 or size_x <= 0 or size_y > ny or size_x > nx:
        return ds

    y0 = (ny - size_y) // 2
    x0 = (nx - size_x) // 2

    return ds.isel(
        y=slice(y0, y0 + size_y),
        x=slice(x0, x0 + size_x),
    )


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

    # Crop to (128, 96) if needed
    crop_center = False
    if crop_center:
        truth, pred, root = map(
            partial(_center_crop, size=(128, 96)), (truth, pred, root)
        )

    return truth.set_coords(["lon", "lat"]), pred.set_coords(["lon", "lat"]), root
