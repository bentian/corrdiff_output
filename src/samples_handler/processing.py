"""
Utilities for per-time-step processing of ensemble prediction datasets.

This module provides building blocks for:
- computing forecast verification metrics and diagnostics
- generating rank histograms for ensemble calibration analysis
- computing spatial error fields
- extracting flattened truth/prediction pairs for PDF analysis
- processing individual time steps for multiprocessing workflows
- evaluating multiple ensemble sizes efficiently

Assumed data layout
-------------------
Truth datasets:
    variables with dims (time, y, x)

Prediction datasets:
    variables with dims (ensemble, time, y, x)
    (or already reduced to (time, y, x))

Key design notes
----------------
- Metrics include RMSE, MAE, CORR, CRPS, ensemble spread, and SSR.
- Rank histograms are computed from valid grid points only.
- Expensive I/O is minimized for multiprocessing workloads.
- Variable naming is preserved for downstream aggregation.
"""

from __future__ import annotations

from typing import Dict, Tuple
from functools import partial

import xarray as xr

from .computing import (
    compute_abs_difference,
    compute_metrics,
    compute_flattened_samples,
    compute_rank_histogram,
)


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
        "metrics": compute_metrics(truth_t, pred_t),
        "rank_histogram": compute_rank_histogram(truth_t, pred_t),
        "error": compute_abs_difference(truth_t, pred_t),
        **compute_flattened_samples(truth_t, pred_t),
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
        out[n_ens] = compute_metrics(truth_t, pred_n)

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
