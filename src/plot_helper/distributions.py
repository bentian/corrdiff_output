"""
Distribution and PDF plotting utilities.

This module provides helpers for visualizing value distributions,
typically using flattened truth and prediction datasets. It is mainly
used to compare probability density functions (PDFs) between model
outputs and reference data.

Typical use cases:
- Plotting global PDFs of truth vs prediction
- Inspecting distributional bias or spread

Inputs are expected to be 1D or flattened xarray datasets.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def _get_bin_count_n_note(ds: xr.DataArray, bin_width: int = 1) -> Tuple[int, str]:
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

        _, note = _get_bin_count_n_note(data.values)
        plt.title(f'{metric} count for {var}\n{note}')

        plt.xlabel(f'{metric} values')
        plt.ylabel('# of days')
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()

        plt.savefig(output_path / f"{var}" / f"cnt_{metric.lower()}.png")
        plt.close()


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
            truth_bin_count, truth_note = _get_bin_count_n_note(truth_flat)
            pred_bin_count, pred_note = _get_bin_count_n_note(pred_flat)

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
