from pathlib import Path
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_metrics(ds: xr.Dataset, output_path: Path, number_format: str) -> None:
    """
    Generate a bar chart for the mean metrics and save the plot.

    Parameters:
        ds (xr.Dataset): Dataset containing metrics.
        output_path (Path): File path to save the output plot.
        number_format (str): Formatting string for displaying numeric values.
    """
    metrics = ds["metric"].values
    variables = list(ds.data_vars.keys())
    data_array = np.array([ds[var] for var in variables])

    x = np.arange(len(metrics))
    width = 0.2

    _, ax = plt.subplots(figsize=(10, 6))
    for i, var in enumerate(variables):
        bars = ax.bar(x + i * width, data_array[i], width, label=var)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:{number_format}}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title("Metrics Mean", fontsize=14)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.set_xticks(x + width * (len(variables) - 1) / 2)
    ax.set_xticklabels([metric for metric in metrics])
    ax.legend(title="Variables")
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)


def plot_monthly_metrics(ds: xr.Dataset, metric: str,
                         output_path: Path, number_format: str) -> None:
    """
    Plot monthly mean values for a given metric.

    Parameters:
        ds (xr.Dataset): Dataset containing monthly mean values.
        metric (str): Metric to plot.
        output_path (Path): File path to save the output plot.
        number_format (str): Formatting string for displaying numeric values.
    """
    _, ax = plt.subplots(figsize=(10, 6))
    df_grouped = ds.to_dataframe()

    for variable in df_grouped.columns:
        ax.plot(df_grouped.index, df_grouped[variable], marker="o", label=variable)
        for x, y in zip(df_grouped.index, df_grouped[variable]):
            ax.annotate(f"{y:{number_format}}", (x, y), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8)

    ax.set_title(f"Monthly Mean for {metric}", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(f"{metric} Value", fontsize=12)
    ax.set_xticks(np.arange(1, 13))
    ax.legend(title="Variables")
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)


def get_bin_count_n_note(ds: xr.DataArray, bin_width: int = 1) -> Tuple[int, str]:
    """
    Compute bin count for histogram plotting and generate summary note.

    Parameters:
        ds (xr.DataArray): DataArray containing values to analyze.
        bin_width (int, optional): Width of bins for histogram. Defaults to 1.

    Returns:
        Tuple[int, str]: The number of bins and a summary note string.
    """
    min_val, max_val = ds.min().item(), ds.max().item()
    bin_count = int((max_val - min_val) / bin_width)
    return bin_count, f"({len(ds):,} pts in [{min_val:.1f}, {max_val:.1f}])"

def plot_pdf(truth: xr.Dataset, pred: xr.Dataset, output_path: Path) -> None:
    """
    Plot PDFs for all variables in the truth dataset, comparing with prediction.

    Parameters:
        truth (xr.Dataset): Truth dataset.
        pred (xr.Dataset): Prediction dataset.
        output_path (Path): File path to save the output plot.
    """
    for var in truth.data_vars:
        if var in pred:
            log_scale = var == 'prcp'  # Apply log scale for 'prcp' only

            truth_flat = truth[var].values.flatten()
            pred_flat = pred[var].mean("ensemble").values.flatten() \
                        if "ensemble" in pred.dims else pred[var].values.flatten()

            # Handle zero values to avoid log(0) errors
            if log_scale:
                truth_flat = np.where(truth_flat > 0, truth_flat, 1e-10)
                pred_flat = np.where(pred_flat > 0, pred_flat, 1e-10)

            # Get bin counts
            truth_bin_count, truth_note = get_bin_count_n_note(truth_flat)
            pred_bin_count, pred_note = get_bin_count_n_note(pred_flat)

            print(f"Variable: {var} | PDF bin count: {truth_bin_count} (truth) / "
                  f"{pred_bin_count} (pred)")

            plt.figure(figsize=(10, 6))

            # Use log-scale bins if needed
            bins_truth = np.logspace(np.log10(min(truth_flat)), np.log10(max(truth_flat)), truth_bin_count) if log_scale else truth_bin_count
            bins_pred = np.logspace(np.log10(min(pred_flat)), np.log10(max(pred_flat)), pred_bin_count) if log_scale else pred_bin_count

            plt.hist(truth_flat, bins=bins_truth, alpha=0.5, label="Truth", density=True)
            plt.hist(pred_flat, bins=bins_pred, alpha=0.5, label="Prediction", density=True)

            # Apply log scales where needed
            if log_scale:
                plt.xscale("log")
                plt.yscale("log")

            plt.title(f"PDF of {var}:\nTruth {truth_note} /\nPrediction {pred_note}")
            plt.xlabel(f"{var} (log scale)" if log_scale else f"{var} (units)")
            plt.ylabel("Density (log scale)" if log_scale else "Density")
            plt.legend()
            plt.grid(which="both" if log_scale else "major", linestyle="--", linewidth=0.5)

            plt.savefig(output_path / f"pdf_{var}.png")
            plt.close()


def plot_monthly_error(ds: xr.Dataset, output_path: Path) -> None:
    """
    Compute monthly mean error and plot results.

    Parameters:
        ds (xr.Dataset): Dataset containing error values.
        output_path (Path): File path to save the output plot.
    """
    monthly_mean = ds.groupby("time.month").mean(dim="time")
    variables = list(ds.data_vars.keys())
    colormaps = ["Blues", "Oranges", "Greens", "Reds"]

    for index, var in enumerate(variables):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for month in range(1, 13):
            data = monthly_mean[var].sel(month=month).mean(dim="ensemble")
            im = axes[month - 1].imshow(data, cmap=colormaps[index], origin="lower")
            axes[month - 1].set_title(f"Month {month}", fontsize=10)
            axes[month - 1].set_axis_off()
            fig.colorbar(im, ax=axes[month - 1], shrink=0.8)

        fig.suptitle(f"Monthly Mean Error of {var}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path / f"monthly_error_{var}.png")


def plot_training_loss(wall_times: List[float], values: List[float], output_file: Path) -> None:
    """
    Create a training loss plot with time on the x-axis and save it to a PNG file.

    Parameters:
        wall_times (List[float]): Wall times (x-axis values).
        values (List[float]): Loss values (y-axis values).
        output_file (Path): File path to save the output plot.
    """
    window_size = 20
    smoothed_values = np.convolve(values, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(wall_times, values, alpha=0.5, label="Raw Loss", color="gray", linewidth=1)
    plt.plot(wall_times[:len(smoothed_values)], smoothed_values,
             label="Smoothed Loss", linestyle="--", linewidth=2)

    plt.xlabel("Time")
    plt.yscale("log")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()  # Format x-axis for better readability
    plt.savefig(output_file)
