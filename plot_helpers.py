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
        for plt_bar in bars:
            height = plt_bar.get_height()
            ax.annotate(f"{height:{number_format}}",
                        xy=(plt_bar.get_x() + plt_bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title("Metrics Mean", fontsize=14)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.set_xticks(x + width * (len(variables) - 1) / 2)
    ax.set_xticklabels(list(metrics))
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

def plot_nyear_metrics(ds: xr.Dataset, metric: str,
                       output_path: Path, number_format: str) -> None:
    """
    Plot N-year mean values for a given metric.

    Parameters:
        ds (xr.Dataset): Dataset containing N-year mean values.
        metric (str): Metric to plot.
        output_path (Path): File path to save the output plot.
        number_format (str): Formatting string for displaying numeric values.
    """
    _, ax = plt.subplots(figsize=(10, 6))
    df_grouped = ds.to_dataframe()

    # Use index as x-axis (e.g., "2015-2020", "2021-2026", ...)
    x = df_grouped.index.astype(str)

    for variable in df_grouped.columns:
        y = df_grouped[variable]
        ax.plot(x, y, marker="o", label=variable)
        # Annotate values
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:{number_format}}", (xi, yi),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=8)

    ax.set_title(f"Decadal Mean for {metric}", fontsize=14)
    ax.set_xlabel("Decade", fontsize=12)
    ax.set_ylabel(f"{metric} Value", fontsize=12)

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

            plt.savefig(output_path / f"{var}" / "pdf.png")
            plt.close()

def plot_metrics_cnt(ds: xr.Dataset, metric: str, output_path: Path) -> None:
    """
    Plot the occurrences of a specified metric for each variable in the dataset.

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

    # Plot and save for each variable
    for i, (var, data) in enumerate(metric_data.data_vars.items()):
        plt.figure(figsize=(6, 4))
        plt.hist(data.values, bins=36, alpha=0.5, color=colors[i % len(colors)], edgecolor='black')

        _, note = get_bin_count_n_note(data.values)
        plt.title(f'{metric} count for {var}\n{note}')

        plt.xlabel(f'{metric} values')
        plt.ylabel('# of days')
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()

        plt.savefig(output_path / f"{var}" / f"cnt_{metric.lower()}.png")
        plt.close()


def plot_sample_images(
    axes: np.ndarray,
    index: int,
    images: List[np.ndarray],
    titles: List[str],
    error_cmap: str
) -> None:
    """
    Plots truth, prediction, and error for a given time step within a subplot grid.

    Parameters:
        axes (np.ndarray):
            A 2D NumPy array of Matplotlib subplot axes for displaying the images.
        index (int):
            The row index in the subplot grid corresponding to the current time step.
        images (List[np.ndarray]):
            A list containing three NumPy arrays:
            - images[0]: Truth values.
            - images[1]: Prediction values.
            - images[2]: Error values.
        titles (List[str]):
            List of three titles corresponding to the truth, prediction, and error subplots.
        error_cmap (str):
            The colormap to be used for the error visualization.

    Generates:
        - Three side-by-side plots for:
            1. **Truth** - Ground truth values.
            2. **Prediction** - Model predictions.
            3. **Error** - Absolute or squared error, depending on the metric.

    Notes:
        - Uses "viridis_r" colormap for truth and prediction.
        - Uses the same color scale (`vmin`, `vmax`) for truth and prediction.
        - Applies a different colormap (`error_cmap`) for error visualization.
    """
    vmin = np.nanmin([images[0], images[1]])
    vmax = np.nanmax([images[0], images[1]])

    for j, title in enumerate(titles):
        # Use the same color scale for truth and prediction for comparison
        im = axes[index, j].imshow(images[j], cmap="viridis_r", aspect="auto", origin="lower",
                                   vmin=vmin, vmax=vmax) if j < 2 else \
             axes[index, j].imshow(images[j], cmap=error_cmap, aspect="auto", origin="lower")
        axes[index, j].set_title(title)
        axes[index, j].axis("off")
        plt.colorbar(im, ax=axes[index, j])

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
        1. **Truth** - Ground truth values.
        2. **Prediction** - Model predictions.
        3. **Error** - Absolute or squared error, depending on the metric.
    - Titles include the corresponding metric value for each time step.
    """
    for var_index, (var, var_data) in enumerate(metric_array[metric].items()):
        times = var_data["metric_value"].time.values
        metric_values = var_data["metric_value"].values

        _, axes = plt.subplots(len(times), 3, figsize=(12, 4 * len(times)))
        axes = [axes] if len(times) == 1 else axes # Ensure axes is iterable for single-row cases

        for i, time in enumerate(times):
            images = [
                var_data["sample"].sel(type=t, time=time).values
                for t in ["truth", "pred", "error"]
            ]

            # Define subplot parameters
            date_str = pd.Timestamp(time).date()
            error_type = "Absolute" if metric == "MAE" else "Square"
            titles = [
                f"Truth on {date_str}",
                f"Prediction on {date_str}",
                f"{error_type} error on {date_str}\n({metric}={metric_values[i]:.2f})",
            ]

            # Generate plots for each time step
            plot_sample_images(axes, i, images, titles, COLOR_MAPS[var_index])

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
        plt.savefig(output_path / f"{var}" / "monthly_error.png")
        plt.close()


def plot_metrics_vs_ensembles(datasets: List[xr.Dataset], output_path: Path):
    """
    Plots metrics vs. number of ensembles (log-scale x-axis), where each line represents
    a different metric. Each data point is labeled with its corresponding value.

    Parameters:
        datasets (List[xr.Dataset]): List of xarray Datasets with different ensemble sizes.
    """
    if not datasets:
        raise ValueError("The dataset list is empty.")

    # Extract number of ensembles from attributes
    ensemble_sizes = np.array([ds.attrs["n_ensemble"] for ds in datasets])

    # Variables to plot
    variables = datasets[0].data_vars
    metrics = datasets[0].metric.values  # ['RMSE', 'MAE', 'CRPS', 'STD_DEV']

    for var in variables:
        plt.figure(figsize=(10, 6))

        for metric_idx, metric_name in enumerate(metrics):
            values = np.array([
                ds[var].isel(metric=metric_idx).mean(dim="time").values for ds in datasets
            ])

            # Filter out missing data (NaNs)
            valid_mask = ~np.isnan(values)
            filtered_x, filtered_y = ensemble_sizes[valid_mask], values[valid_mask]

            # Plot the data
            plt.plot(filtered_x, filtered_y, marker="o", label=metric_name)

            # Add data labels to each point
            for x, y in zip(filtered_x, filtered_y):
                plt.text(x, y, f"{y:.2f}", fontsize=10, ha="right", va="bottom")

        plt.xscale("log")  # Set x-axis to log scale
        plt.xticks(ensemble_sizes, labels=map(str, ensemble_sizes))
        plt.xlabel("Number of Ensembles (log scale)")
        plt.ylabel("Metric Value")
        plt.title(f"Metrics ({var}) vs. Number of Ensembles")

        plt.legend(title="Metric", loc="upper left", bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.85)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        plt.savefig(output_path / var / "metrics_v_ensembles.png")
        plt.close()


def plot_training_loss(wall_times: List[float], values: List[float], output_file: Path) -> None:
    """
    Create a training loss plot with time on the x-axis and save it to a PNG file.

    Parameters:
        wall_times (List[float]): Wall times (x-axis values).
        values (List[float]): Loss values (y-axis values).
        output_file (Path): File path to save the output plot.
    """
    window_size = min(20, len(values))
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
