import dask.diagnostics
import dask
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr
from functools import partial

try:
    import xskillscore
except ImportError:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`")

VAR_MAPPING ={
    "precipitation": "prcp",
    "temperature_2m": "t2m",
    "eastward_wind_10m": "u10m",
    "northward_wind_10m": "v10m",
}
REF_GRID_NC = "./data/wrf_208x208_grid_coords.nc"

def apply_landmask(truth, pred):
    grid = xr.open_dataset(REF_GRID_NC, engine='netcdf4')
    landmask = grid.LANDMASK.rename({"south_north": "y", "west_east": "x"})

    # Apply the landmask to both datasets and fill NaN with 0
    landmask_expanded = landmask.expand_dims(dim={"time": truth.sizes["time"]})
    truth = truth.where(landmask_expanded == 1, 0)

    landmask_expanded = landmask_expanded.expand_dims(dim={"ensemble": pred.sizes["ensemble"]})
    pred = pred.where(landmask_expanded == 1, 0)

    return truth, pred

def open_samples(f, masked):
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

    if not masked:
        return truth, pred, root

    truth, pred = apply_landmask(truth, pred)
    return truth, pred, root

def compute_metrics(truth, pred):
    dim = ["x", "y"]

    a = xskillscore.rmse(truth, pred.mean("ensemble"), dim=dim)
    b = xskillscore.crps_ensemble(truth, pred, member_dim="ensemble", dim=dim)

    c = pred.std("ensemble").mean(dim)
    crps_mean = xskillscore.crps_ensemble(
        truth,
        pred.mean("ensemble").expand_dims("ensemble"),
        member_dim="ensemble",
        dim=dim,
    )

    return (
        xr.concat([a, b, c, crps_mean], dim="metric")
        .assign_coords(metric=["rmse", "crps", "std_dev", "mae"])
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
        coords={
            "points": np.arange(len(truth_flat))  # Index for each flattened point
        },
        attrs={
            "description": "Flattened PRCP truth and prediction values",
            "units": "mm"
        }
    )

def process_sample(index, filepath, n_ensemble, masked):
    truth, pred, _ = open_samples(filepath, masked)
    truth = truth.isel(time=index).load()
    if n_ensemble > 0:
        pred = pred.isel(time=index, ensemble=slice(0, n_ensemble))
    pred = pred.load()

    return {
        "flat": get_flat(truth, pred), # Flattened PRCP truth and prediction
        "error": abs(pred.mean("ensemble").expand_dims("ensemble") - truth), # Spatial error
        "metrics": compute_metrics(truth, pred), # CorrDiff scores
    }

def combine_results(results):
    combined_flat = xr.concat([res["flat"] for res in results], dim="points")
    combined_error = xr.concat([res["error"] for res in results], dim="time")
    combined_metrics = xr.concat([res["metrics"] for res in results], dim="time")

    return combined_flat, combined_error, combined_metrics

def plot_pdf(truth, prediction, output_path):
    truth_title = f"{len(truth)} [{min(truth)}, {max(truth)}]"
    pred_title = f"{len(prediction)} [{min(prediction)}, {max(prediction)}]"

    plt.figure(figsize=(10, 6))
    plt.hist(truth, bins=50, alpha=0.5, label="Truth", density=True)
    plt.hist(prediction, bins=50, alpha=0.5, label="Prediction", density=True)
    plt.title(f"PDF of PRCP\n Truth ({truth_title}) and\n Prediction ({pred_title})")
    plt.xlabel("PRCP (mm)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)

def plot_cdf(truth, prediction, output_path):
    truth_title = f"{len(truth)} [{min(truth)}, {max(truth)}]"
    pred_title = f"{len(prediction)} [{min(prediction)}, {max(prediction)}]"

    truth_sorted = np.sort(truth)
    pred_sorted = np.sort(prediction)

    truth_cdf = np.arange(1, len(truth_sorted) + 1) / len(truth_sorted)
    pred_cdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)

    plt.figure(figsize=(10, 6))
    plt.plot(truth_sorted, truth_cdf, label="Truth CDF", color="blue", linewidth=2)
    plt.plot(pred_sorted, pred_cdf, label="Prediction CDF", color="orange", linewidth=2)
    plt.title(f"CDF of PRCP\n Truth ({truth_title}) and\n Prediction ({pred_title})")
    plt.xlabel("PRCP (mm)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)

def plot_density(dataset, output_path_prefix):
    plot_pdf(dataset["truth"], dataset["prediction"], output_path_prefix + "_pdf.png")
    plot_cdf(dataset["truth"], dataset["prediction"], output_path_prefix + "_cdf.png")

def plot_monthly_mean(ds, output_path_prefix):
    """
    Group variables by month, compute the monthly mean, and plot the results.

    Parameters:
        dataset_path (str): Path to the NetCDF dataset file.
    """
    # Compute monthly mean for all variables
    monthly_mean = ds.groupby("time.month").mean(dim="time")

    # Variables to plot
    variables = list(ds.data_vars.keys())
    colormaps = ["Blues", "Oranges", "Greens", "Reds"]

    for index, var in enumerate(variables):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for month in range(1, 13):  # Iterate over months (1-12)
            # Select data for the variable and current month
            data = monthly_mean[var].sel(month=month).mean(dim="ensemble")  # Mean over ensemble

            # Plot the data for the current month
            im = axes[month - 1].imshow(data, cmap=colormaps[index], origin="lower")
            axes[month - 1].set_title(f"Month {month}", fontsize=10)
            axes[month - 1].set_axis_off()
            fig.colorbar(im, ax=axes[month - 1], shrink=0.8)

        # Adjust layout and add a main title
        fig.suptitle(f"Monthly Mean Error of {var.replace('_', ' ').capitalize()}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        plt.savefig(output_path_prefix + f"_monthly_mean_{var}.png")

def score_samples_n_plot(filepath, output_path_prefix, n_ensemble=1, masked=True):
    truth, _, _ = open_samples(filepath, masked)
    with multiprocessing.Pool(32) as pool:
        results = list(tqdm.tqdm(
            pool.imap(
                partial(process_sample, filepath=filepath, n_ensemble=n_ensemble, masked=masked),
                range(truth.sizes["time"])),
            total=truth.sizes["time"]
        ))

    combined_flat, combined_error, combined_metrics = combine_results(results)

    # Plot results
    plot_density(combined_flat, output_path_prefix)
    plot_monthly_mean(combined_error.rename(VAR_MAPPING), output_path_prefix)

    # Save metrics
    combined_metrics.attrs["n_ensemble"] = n_ensemble
    score_filename = output_path_prefix + "_score.nc"
    with dask.config.set(scheduler="single-threaded"):
        combined_metrics.to_netcdf(score_filename)

    return score_filename
