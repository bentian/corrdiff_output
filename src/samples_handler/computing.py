"""
Computation utilities for ensemble forecast verification.

This module provides metric and diagnostic calculations for comparing
ensemble predictions against observations using xarray and xskillscore.

Features
--------
- Deterministic metrics:
    - RMSE
    - MAE
    - CORR (Pearson correlation)

- Probabilistic metrics:
    - CRPS (Continuous Ranked Probability Score)

- Ensemble diagnostics:
    - Ensemble spread (STD_DEV)
    - Spread-skill ratio (SSR)
    - Rank histogram (Talagrand diagram)

- Spatial diagnostics:
    - Absolute error fields

Assumed data layout
-------------------
Truth datasets:
    variables with dims (..., y, x)

Prediction datasets:
    variables with dims (ensemble, ..., y, x)

Design notes
------------
- Spatial reductions are performed over (x, y).
- Probabilistic metrics use only grid points with valid observations and valid ensemble members.
- NaN handling is consistent across metrics and diagnostics.
- Functions operate on in-memory xarray datasets and perform no I/O.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import xarray as xr

try:
    import xskillscore as xs
except ImportError as exc:
    raise ImportError(
        "xskillscore not installed. Try `pip install xskillscore`"
    ) from exc


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


def compute_rank_histogram(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """Computes the rank histogram while filtering out NaN values."""
    truth_valid, pred_valid = _mask_valid_points(truth, pred)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return xs.rank_histogram(
            truth_valid, pred_valid, member_dim="ensemble", dim=["x", "y"]
        ).assign_coords(time=truth.coords["time"])


def compute_metrics(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
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


def compute_flattened_samples(
    truth: xr.Dataset, pred: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compute flattened truth and prediction samples for all variables.

    Parameters:
        truth (xr.Dataset): The truth dataset.
        pred (xr.Dataset): The prediction dataset.

    Returns:
        tuple: Two xarray.Datasets, one for truth and one for prediction.
    """
    truth_out, pred_out = {}, {}

    for var in truth.data_vars:
        if var not in pred:
            continue

        t = truth[var].values.ravel()
        p = (
            pred[var].mean("ensemble").values.ravel()
            if "ensemble" in pred[var].dims
            else pred[var].values.ravel()
        )

        valid = np.isfinite(t) & np.isfinite(p)
        truth_out[var] = ("points", t[valid])
        pred_out[var] = ("points", p[valid])

    return {
        "truth_flat": xr.Dataset(truth_out),
        "pred_flat": xr.Dataset(pred_out),
    }
