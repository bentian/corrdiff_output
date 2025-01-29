import multiprocessing
import numpy as np
import tqdm
import xarray as xr
from functools import partial
from typing import Tuple, Dict

try:
    import xskillscore
except ImportError:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`")

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
    pred = xr.open_dataset(f, group="prediction")
    truth = xr.open_dataset(f, group="truth")

    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])

    return truth, pred, root


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

    rmse = xskillscore.rmse(truth, pred.mean("ensemble"), dim=dim)
    crps = xskillscore.crps_ensemble(truth, pred, member_dim="ensemble", dim=dim)

    std_dev = pred.std("ensemble").mean(dim)
    crps_mean = xskillscore.crps_ensemble(
        truth,
        pred.mean("ensemble").expand_dims("ensemble"),
        member_dim="ensemble",
        dim=dim,
    )

    return (
        xr.concat([rmse, crps_mean, crps, std_dev], dim="metric")
        .assign_coords(metric=["rmse", "mae", "crps", "std_dev"])
        .load()
    )


def get_flat(truth: xr.Dataset, pred: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
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

            truth_data[var] = ("points", truth_flat)
            pred_data[var] = ("points", pred_flat)

    combined_truth = xr.Dataset(
        truth_data,
        coords={"points": np.arange(len(next(iter(truth_data.values()))[1]))},
    )
    combined_pred = xr.Dataset(
        pred_data,
        coords={"points": np.arange(len(next(iter(pred_data.values()))[1]))},
    )

    return combined_truth, combined_pred


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

    truth_flat, pred_flat = get_flat(truth, pred)
    result = {
        "metrics": compute_metrics(truth, pred),
        "error": abs(pred.mean("ensemble").expand_dims("ensemble") - truth),
        "truth_flat": truth_flat,
        "pred_flat": pred_flat
    }

    return result


def score_samples(filepath: str, n_ensemble: int = 1
                  ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Score the dataset by computing various metrics over all time steps.

    Parameters:
        filepath (str): Path to the dataset file.
        n_ensemble (int, optional): Number of ensemble members. Defaults to 1.

    Returns:
        Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]: Computed metrics, spatial errors
        flattened truth, and flattened prediction datasets.
    """
    truth, _, _ = open_samples(filepath)

    with multiprocessing.Pool(32) as pool:
        results = []
        for result in tqdm.tqdm(
            pool.imap(
                partial(process_sample,
                        filepath=filepath, n_ensemble=n_ensemble),
                range(truth.sizes["time"]),
            ),
            total=truth.sizes["time"],
        ):
            results.append(result)

    # Combine metrics
    combined_metrics = \
        xr.concat([res["metrics"] for res in results], dim="time").rename(VAR_MAPPING)
    combined_metrics.attrs["n_ensemble"] = n_ensemble

    # Combine spatial error and flattened data
    combined_data = {
        "error": xr.concat([res["error"] for res in results], dim="time"),
        "truth_flat": xr.concat([res["truth_flat"] for res in results], dim="points"),
        "pred_flat": xr.concat([res["pred_flat"] for res in results], dim="points"),
    }

    combined_data = {key: value.rename(VAR_MAPPING) for key, value in combined_data.items()}
    return combined_metrics, combined_data["error"], \
        combined_data["truth_flat"], combined_data["pred_flat"]
