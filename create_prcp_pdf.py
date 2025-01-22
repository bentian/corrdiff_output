import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr
from functools import partial
from score_samples import open_samples

try:
    import xskillscore
except ImportError:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`")


def process_sample(index, filepath, n_ensemble):
    truth, pred, _ = open_samples(filepath)
    truth = truth.isel(time=index).load()
    if n_ensemble > 0:
        pred = pred.isel(time=index, ensemble=slice(0, n_ensemble))
    pred = pred.load()

    truth_flat = truth['precipitation'].values.flatten()
    pred_flat = pred['precipitation'].mean("ensemble").values.flatten() \
                if "ensemble" in pred.dims else pred['precipitation'].values.flatten()
    error = pred.mean("ensemble").expand_dims("ensemble") - truth

    return {
        "truth": truth_flat, "prediction": pred_flat, "error": error
    }


def combine_results(results):
    combined_truth = np.concatenate([res["truth"] for res in results])
    combined_pred = np.concatenate([res["prediction"] for res in results])
    combined_error = xr.concat([res["error"] for res in results], dim="time")
    return combined_truth, combined_pred, combined_error


def plot_pdf(truth, prediction, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(truth, bins=50, alpha=0.5, label="Truth", density=True)
    plt.hist(prediction, bins=50, alpha=0.5, label="Prediction", density=True)
    plt.title("PDF of PRCP Truth and Prediction")
    plt.xlabel("PRCP (mm)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)


def plot_cdf(truth, prediction, output_path):
    truth_sorted = np.sort(truth)
    pred_sorted = np.sort(prediction)

    truth_cdf = np.arange(1, len(truth_sorted) + 1) / len(truth_sorted)
    pred_cdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)

    plt.figure(figsize=(10, 6))
    plt.plot(truth_sorted, truth_cdf, label="Truth CDF", color="blue", linewidth=2)
    plt.plot(pred_sorted, pred_cdf, label="Prediction CDF", color="orange", linewidth=2)
    plt.title("CDF of PRCP Truth and Prediction")
    plt.xlabel("PRCP (mm)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)


def group_and_plot_monthly_mean(ds, output_path_prefix):
    """
    Group variables by month, compute the monthly mean, and plot the results.

    Parameters:
        dataset_path (str): Path to the NetCDF dataset file.
    """
    # Compute monthly mean for all variables
    monthly_mean = ds.groupby("time.month").mean(dim="time")

    # Variables to plot
    variables = list(ds.data_vars.keys())

    for var in variables:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for month in range(1, 13):  # Iterate over months (1-12)
            # Select data for the variable and current month
            data = monthly_mean[var].sel(month=month).mean(dim="ensemble")  # Mean over ensemble

            # Plot the data for the current month
            im = axes[month - 1].imshow(data, cmap="viridis", origin="lower")
            axes[month - 1].set_title(f"Month {month}", fontsize=10)
            axes[month - 1].set_axis_off()
            fig.colorbar(im, ax=axes[month - 1], shrink=0.8)

        # Adjust layout and add a main title
        fig.suptitle(f"Monthly Mean 2D Distribution of {var.replace('_', ' ').capitalize()}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        plt.savefig(output_path_prefix + f"_monthly_mean_{var}.png")

def create_prcp_pdf(filepath, output_path_prefix, n_ensemble=1):
    truth, _, _ = open_samples(filepath)
    with multiprocessing.Pool(32) as pool:
        results = list(tqdm.tqdm(
            pool.imap(
                partial(process_sample, filepath=filepath, n_ensemble=n_ensemble),
                range(truth.sizes["time"])),
            total=truth.sizes["time"]
        ))

    combined_truth, combined_pred, combined_error = combine_results(results)

    plot_pdf(combined_truth, combined_pred, output_path_prefix + "_pdf.png")
    plot_cdf(combined_truth, combined_pred, output_path_prefix + "_cdf.png")

    group_and_plot_monthly_mean(combined_error, output_path_prefix)
