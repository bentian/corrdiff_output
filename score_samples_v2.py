"""
Module for processing and scoring dataset samples using xarray and xskillscore.

This module provides utilities to:
- Open NetCDF dataset samples (truth and predictions).
- Compute various evaluation metrics (RMSE, MAE, CRPS).
- Extract top-ranked samples based on selected metrics.
- Flatten and filter NaN values for efficient processing.
- Apply multiprocessing for performance optimization.

Dependencies:
    - multiprocessing (for parallel processing)
    - numpy (for numerical operations)
    - tqdm (for progress tracking)
    - xarray (for working with NetCDF datasets)
    - xskillscore (for computing skill scores)

Constants:
    VAR_MAPPING (Dict[str, str]): Mapping of long variable names to short variable names.

Functions:
    - open_samples(f: str) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        Opens NetCDF dataset samples, returning truth, prediction, and root datasets.

    - compute_crps(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
        Computes the Continuous Ranked Probability Score (CRPS) for ensemble predictions.

    - compute_metrics(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
        Computes RMSE and MAE between truth and prediction datasets.

    - flatten_and_filter_nan(truth: xr.Dataset, pred: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        Flattens datasets and removes NaN values for all variables.

    - compute_abs_difference(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
        Computes absolute differences between truth and prediction datasets.

    - process_sample(index: int, filepath: str, n_ensemble: int) -> Dict[str, xr.Dataset]:
        Processes a single time step from the dataset, computing metrics and errors.

    - extract_top_samples(truth: xr.Dataset, pred: xr.Dataset, combined_metrics: xr.Dataset,
                          metric: str, N: int = 5) -> dict:
        Extracts top N samples with the highest metric values.

    - score_samples(filepath: str, n_ensemble: int = 1) ->
                    Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, dict]:
        Computes evaluation metrics, applies multiprocessing, and extracts top-ranked samples.

Usage Example:
    >>> from dataset_scoring import score_samples
    >>> metrics, spatial_error, truth_flat, pred_flat, top_samples = \
    >>>   score_samples("data.nc", n_ensemble=10)
    >>> print(metrics)

"""
import multiprocessing
from functools import partial
from typing import Tuple, Dict
import numpy as np
import tqdm
import xarray as xr

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

    # Compute mean across ensemble
    pred_mean = pred.mean("ensemble") if "ensemble" in pred.dims else pred

    # Create a mask for finite values in both truth and prediction
    valid_mask = np.isfinite(truth) & np.isfinite(pred_mean)

    # Apply the mask (remove NaNs)
    truth_valid = truth.where(valid_mask)
    pred_valid = pred.where(valid_mask)

    # Compute CRPS only for valid values
    crps = xs.crps_ensemble(truth_valid, pred_valid, member_dim="ensemble", dim=dim)

    return crps

def compute_metrics(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """
    Compute RMSE, CRPS, and standard deviation between truth and prediction datasets.

    Parameters:
        truth (xr.Dataset): The truth dataset.
        pred (xr.Dataset): The prediction dataset.

    Returns:
        xr.Dataset: A dataset containing computed metrics.
    """
    dim = ["x", "y"]

    rmse = xs.rmse(truth, pred.mean("ensemble"), dim=dim, skipna=True)
    mae = xs.mae(truth, pred.mean("ensemble"), dim=dim, skipna=True)
    # std_dev = pred.std("ensemble").mean(dim, skipna=True)
    # crps = compute_crps(truth, pred)

    return (
        xr.concat([rmse, mae], dim="metric")
        .assign_coords(metric=["RMSE", "MAE"])
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
    truth_data = {}
    pred_data = {}

    for var in truth.data_vars:
        if var in pred:
            truth_flat = truth[var].values.flatten()
            pred_flat = pred[var].mean("ensemble").values.flatten() \
                        if "ensemble" in pred.dims else pred[var].values.flatten()

            # Filter out NaNs and align truth and pred
            valid_mask = np.isfinite(truth_flat) & np.isfinite(pred_flat)
            truth_data[var] = ("points", truth_flat[valid_mask])
            pred_data[var] = ("points", pred_flat[valid_mask])

    combined_truth = xr.Dataset(
        truth_data,
        coords={"points": np.arange(len(next(iter(truth_data.values()))[1]))},
    )
    combined_pred = xr.Dataset(
        pred_data,
        coords={"points": np.arange(len(next(iter(pred_data.values()))[1]))},
    )

    return combined_truth, combined_pred


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
    # Compute mean across ensemble
    pred_mean = pred.mean("ensemble").expand_dims("ensemble") if "ensemble" in pred.dims else pred

    # Compute absolute difference
    abs_diff = abs(pred_mean - truth)

    # Filter out NaNs by setting them to NaN where either input is NaN
    valid_mask = np.isfinite(truth) & np.isfinite(pred_mean)
    abs_diff = abs_diff.where(valid_mask)

    return abs_diff

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
    truth, pred, _ = open_samples(filepath)
    truth = truth.isel(time=index).load()
    if n_ensemble > 0:
        pred = pred.isel(time=index, ensemble=slice(0, n_ensemble))
    pred = pred.load()

    truth_flat, pred_flat = flatten_and_filter_nan(truth, pred)
    result = {
        "metrics": compute_metrics(truth, pred),
        "error": compute_abs_difference(truth, pred),
        "truth_flat": truth_flat,
        "pred_flat": pred_flat
    }

    return result


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
    data_vars = {}

    for var in truth.data_vars:
        if var not in combined_metrics:
            continue

        # Select top N dates based on metric values
        top_dates = combined_metrics.sel(metric=metric)[var].to_series().nlargest(top_num)
        selected_times = np.array(top_dates.index, dtype="datetime64[ns]")

        # Extract data
        truth_selected = truth[var].sel(time=selected_times).load()
        pred_selected = pred[var].sel(time=selected_times).mean("ensemble").load()

        abs_error = compute_abs_difference(truth_selected, pred_selected)
        error = abs_error if metric == "MAE" else abs_error ** 2

        # Store results in a dictionary
        data_vars[var] = {
            "sample": xr.concat([truth_selected, pred_selected, error], dim="type")
                      .assign_coords(type=["truth", "pred", "error"]),
            "metric_value": xr.DataArray(top_dates.values, dims=["time"],
                                         coords={"time": selected_times})
        }

    return data_vars

def score_samples(
    filepath: str, n_ensemble: int = 1
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
    truth, pred, _ = open_samples(filepath)

    with multiprocessing.Pool(32) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(
                    partial(process_sample, filepath=filepath, n_ensemble=n_ensemble),
                    range(truth.sizes["time"]),
                ),
                total=truth.sizes["time"],
            )
        )

    # Combine metrics
    combined_metrics = \
        xr.concat([res["metrics"] for res in results], dim="time").rename(VAR_MAPPING)
    combined_metrics.attrs["n_ensemble"] = n_ensemble

    # Combine spatial error and flattened data
    combined_data = {
        key: xr.concat([res[key] for res in results], dim=dim).rename(VAR_MAPPING)
        for key, dim in zip(["error", "truth_flat", "pred_flat"], ["time", "points", "points"])
    }

    # Extract top N dates with the highest metric values for RMSE and MAE
    top_samples = {
        metric: extract_top_samples(
            truth.rename(VAR_MAPPING), pred.rename(VAR_MAPPING), combined_metrics, metric
        ) for metric in ["MAE", "RMSE"]
    }

    return combined_metrics, combined_data["error"], \
        combined_data["truth_flat"], combined_data["pred_flat"], top_samples
