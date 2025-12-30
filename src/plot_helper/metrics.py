"""
Metric visualization utilities.

This module contains plotting functions for evaluating model performance
metrics such as RMSE, MAE, CORR, CRPS, and ensemble spread. It supports:

- Time-averaged metric plots
- Monthly and N-year aggregated metric plots
- Metric counts and binning visualizations
- Comparisons across different ensemble sizes

All functions operate on xarray Datasets produced by the evaluation
pipeline and save publication-ready figures to disk.
"""
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


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
