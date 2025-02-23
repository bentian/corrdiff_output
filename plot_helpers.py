"""
Module for generating plots and analyzing evaluation metrics for model predictions.

This module provides functionality to:
- Generate plots for various model evaluation metrics.
- Compute and visualize Probability Density Functions (PDFs).
- Generate monthly mean plots for model metrics.
- Save evaluation results and metric tables in structured formats.
- Process and visualize training loss data from TensorBoard logs.

Dependencies:
    - pathlib (for file path management)
    - typing (for type annotations)
    - numpy (for numerical operations)
    - pandas (for handling structured tabular data)
    - xarray (for working with NetCDF datasets)
    - matplotlib (for generating plots)

Constants:
    COLOR_MAPS (List[str]): A list of color maps used for different variables.

Functions:
    - plot_metrics(ds: xr.Dataset, output_path: Path, number_format: str) -> None:
        Generates a bar chart of mean metric values and saves the plot.

    - plot_monthly_metrics(ds: xr.Dataset, metric: str, output_path: Path,
                           number_format: str) -> None:
        Generates and saves monthly mean plots for a specified metric.

    - get_bin_count_n_note(ds: xr.DataArray, bin_width: int = 1) -> Tuple[int, str]:
        Computes the number of bins for histogram plotting and generates a summary note.

    - plot_pdf(truth: xr.Dataset, pred: xr.Dataset, output_path: Path) -> None:
        Generates and saves PDFs for each variable, comparing truth vs. prediction.

    - plot_metrics_pdf(ds: xr.Dataset, metric: str, output_path: Path) -> None:
        Generates PDFs of the specified metric for all variables.

    - plot_top_samples(metric_array: dict, metric: str, output_path: Path) -> None:
        Generates plots of truth, prediction, and error for each time step, sorted by metric value.

    - plot_monthly_error(ds: xr.Dataset, output_path: Path) -> None:
        Computes monthly mean error and generates plots.

    - plot_training_loss(wall_times: List[float], values: List[float], output_file: Path) -> None:
        Generates and saves a training loss plot over time.

Usage Example:
    >>> from pathlib import Path
    >>> import xarray as xr
    >>> ds = xr.open_dataset("metrics.nc")
    >>> plot_metrics(ds, Path("output/metrics.png"), number_format=".2f")
"""
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Color maps for each variable
COLOR_MAPS: List[str] = ["Blues", "Oranges", "Greens", "Reds"]

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
    plt.close()

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
    plt.close()


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

            # print(f"Variable: {var} | PDF bin count: {truth_bin_count} (truth) / "
            #       f"{pred_bin_count} (pred)")

            plt.figure(figsize=(10, 6))

            # Use log-scale bins if needed
            bins_truth = np.logspace(np.log10(min(truth_flat)),
                                     np.log10(max(truth_flat)), truth_bin_count) \
                         if log_scale else truth_bin_count

            bins_pred = np.logspace(np.log10(min(pred_flat)),
                                    np.log10(max(pred_flat)), pred_bin_count) \
                        if log_scale else pred_bin_count

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

            plt.savefig(output_path / f"{var}" / f"pdf.png")
            plt.close()

def plot_metrics_pdf(ds: xr.Dataset, metric: str, output_path: Path) -> None:
    """
    Plot the Probability Density Function (PDF) of a specified metric
    for each variable in the dataset.

    Parameters:
    ds (xr.Dataset): The input dataset containing various metrics for different variables.
    metric (str): The specific metric to plot (e.g., 'RMSE', 'MAE').
    output_path (Path): The directory where the plot image will be saved.

    Returns:
    None
    """
    # Select the specified metric
    metric_data = ds.sel(metric=metric)

    # Define colors for plotting
    colors = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 distinct colors

    # Plot and save PDF for each variable
    for i, (var, data) in enumerate(metric_data.data_vars.items()):
        plt.figure(figsize=(6, 4))
        plt.hist(data.values, bins=36, density=True, alpha=0.5,
                 color=colors[i % len(colors)], edgecolor='black')

        _, note = get_bin_count_n_note(data.values)
        plt.title(f'PDF of {metric} for {var}\n{note}')

        plt.xlabel(f'{metric} values')
        plt.ylabel('Density')
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()

        plt.savefig(output_path / f"{var}" / f"pdf_{metric.lower()}.png")
        plt.close()


def plot_top_samples(metric_array: dict, metric: str, output_path: Path) -> None:
    """
    Plots truth, prediction, and error for each time step in each variable, maintaining the order
    from the dataset and displaying the corresponding metric value in the plot title.

    Parameters:
    - metric_array (dict): A dictionary containing extracted samples and metric values for
                           each variable. Expected structure:
                           {
                               "variable_name": {
                                   "sample": xarray.DataArray with dimensions (time, y, x, type),
                                   "metric_value": xarray.DataArray with dimensions (time)
                               },
                               ...
                           }
    - metric (str): The metric used for extracting top samples (e.g., "RMSE" or "MAE").
    - output_path (Path): Directory where the plots will be saved.

    Output:
    - Saves images for each variable in the format:
        {output_path}/{variable}/top_samples_{metric}.png
    - Each row in the plot represents a different time step, with three columns:
        1. Truth
        2. Prediction
        3. Error (Absolute or Squared, depending on metric type)
    - Titles include the corresponding metric value for each time step.
    """
    for var_index, (var, var_data) in enumerate(metric_array[metric].items()):
        metric_value_data = var_data["metric_value"]
        times, metric_values = metric_value_data.time.values, metric_value_data.values
        samples = var_data["sample"]

        _, axes = plt.subplots(len(times), 3, figsize=(12, 4 * len(times)))
        for i, time in enumerate(times):
            date_str = pd.Timestamp(time).date()

            # Extract slices
            truth = samples.sel(type="truth", time=time).values
            pred = samples.sel(type="pred", time=time).values
            error = samples.sel(type="error", time=time).values

            # Determine shared color limits for truth and pred
            vmin, vmax = np.nanmin([truth, pred]), np.nanmax([truth, pred])

            # Ensure axes is always iterable for single-row cases
            if len(times) == 1:
                axes = [axes]

            # Plot truth
            im1 = axes[i, 0].imshow(
                truth, cmap="viridis_r", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f"Truth on {date_str}")
            axes[i, 0].axis("off")
            plt.colorbar(im1, ax=axes[i, 0])

            # Plot pred
            im2 = axes[i, 1].imshow(
                pred, cmap="viridis_r", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f"Prediction on {date_str}")
            axes[i, 1].axis("off")
            plt.colorbar(im2, ax=axes[i, 1])

            # Plot error
            error_type = "Absolute" if metric == "MAE" else "Square"
            im3 = axes[i, 2].imshow(
                error, cmap=COLOR_MAPS[var_index], aspect="auto", origin="lower")
            axes[i, 2].set_title(
                f"{error_type} error on {date_str}\n({metric}={metric_values[i]:.2f})")
            axes[i, 2].axis("off")
            plt.colorbar(im3, ax=axes[i, 2])

        plt.tight_layout()
        plt.savefig(output_path / f"{var}" / f"top_samples_{metric.lower()}.png")
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

    for index, var in enumerate(variables):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for month in range(1, 13):
            data = monthly_mean[var].sel(month=month).mean(dim="ensemble")
            im = axes[month - 1].imshow(data, cmap=COLOR_MAPS[index], origin="lower")
            axes[month - 1].set_title(f"Month {month}", fontsize=10)
            axes[month - 1].set_axis_off()
            fig.colorbar(im, ax=axes[month - 1], shrink=0.8)

        fig.suptitle(f"Monthly Mean Error of {var}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path / f"{var}" / f"monthly_error.png")
        plt.close()


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
    plt.close()
