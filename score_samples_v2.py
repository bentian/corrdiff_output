import multiprocessing
import numpy as np
import tqdm
import xarray as xr
from functools import partial

try:
    import xskillscore
except ImportError:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`")

APPLY_LANDMASK = False # Whether to apply the landmask to the data
LANDMASK_NC = "./data/wrf_208x208_grid_coords.nc" # Path to the landmask NetCDF file
VAR_MAPPING ={
    "precipitation": "prcp",
    "temperature_2m": "t2m",
    "eastward_wind_10m": "u10m",
    "northward_wind_10m": "v10m",
}

def apply_landmask(truth, pred):
    grid = xr.open_dataset(LANDMASK_NC, engine='netcdf4')
    landmask = grid.LANDMASK.rename({"south_north": "y", "west_east": "x"})

    # Apply the landmask to both datasets and fill NaN with 0
    landmask_expanded = landmask.expand_dims(dim={"time": truth.sizes["time"]})
    truth = truth.where(landmask_expanded == 1, 0)

    landmask_expanded = landmask_expanded.expand_dims(dim={"ensemble": pred.sizes["ensemble"]})
    pred = pred.where(landmask_expanded == 1, 0)

    return truth, pred

def open_samples(f):
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

    if not APPLY_LANDMASK:
        return truth, pred, root

    truth, pred = apply_landmask(truth, pred)
    return truth, pred, root

def compute_metrics(truth, pred):
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

def get_flat(truth, pred, var="precipitation"):
    truth_flat = truth[var].values.flatten()
    pred_flat = pred[var].mean("ensemble").values.flatten() \
                if "ensemble" in pred.dims else pred[var].values.flatten()

    return xr.Dataset(
        {
            "truth": ("points", truth_flat),
            "prediction": ("points", pred_flat)
        },
        coords={"points": np.arange(len(truth_flat))},
    )

def process_sample(index, filepath, n_ensemble, metrics_only):
    truth, pred, _ = open_samples(filepath)
    truth = truth.isel(time=index).load()
    if n_ensemble > 0:
        pred = pred.isel(time=index, ensemble=slice(0, n_ensemble))
    pred = pred.load()

    result = { "metrics": compute_metrics(truth, pred) }
    if not metrics_only:
        result["flat"] = get_flat(truth, pred)
        result["error"] = abs(pred.mean("ensemble").expand_dims("ensemble") - truth)

    return result

def score_samples(filepath, n_ensemble=1, metrics_only=False):
    truth, _, _ = open_samples(filepath)

    with multiprocessing.Pool(32) as pool:
        results = []
        for result in tqdm.tqdm(
            pool.imap(
                partial(process_sample,
                        filepath=filepath, n_ensemble=n_ensemble, metrics_only=metrics_only),
                range(truth.sizes["time"]),
            ),
            total=truth.sizes["time"],
        ):
            results.append(result)

    # Combine metrics
    combined_metrics = xr.concat([res["metrics"] for res in results], dim="time")
    combined_metrics.attrs["n_ensemble"] = n_ensemble
    if metrics_only:
        return combined_metrics

    # Combine spatial error and flat data
    combined_error = xr.concat([res["error"] for res in results], dim="time")
    combined_flat = xr.concat([res["flat"] for res in results], dim="points")

    return combined_metrics, combined_error, combined_flat
