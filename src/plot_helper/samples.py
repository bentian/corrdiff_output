"""
Spatial sample and error visualization utilities.

This module contains functions for plotting spatial fields such as:
- Individual truth / prediction / error maps
- Top-N worst samples based on evaluation metrics
- Monthly mean spatial error maps

Plots are typically arranged in multi-row, multi-column grids to
facilitate visual comparison across time steps or periods.
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# Color maps for each variable
COLOR_MAPS: List[str] = ["Blues", "Oranges", "Greens", "Reds"]


def _plot_sample_images(
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
            3. **Error** - Error between truth and prediction, depending on the metric.

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
        {output_path}/{var}/top_samples_{metric}.png
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
            _plot_sample_images(axes, i, images, titles, COLOR_MAPS[var_index])

        plt.tight_layout()
        plt.savefig(output_path / f"{var}" / f"top_samples_{metric.lower()}.png")
        plt.close()


def plot_p90_by_nyear(truth_p90: xr.Dataset, pred_p90: xr.Dataset,
                      output_path: Path, period_dim: str = "period") -> None:
    """
    Plot p90 maps by period. For each variable, save one figure where each row is a period and
    columns are: (1) truth p90, (2) pred p90, (3) pred p90 - truth p90.

    Parameters
    ----------
    truth_p90 : xr.Dataset
        Dataset with dims (period_dim, y, x) and data_vars like prcp, t2m, ...
    pred_p90 : xr.Dataset
        Same structure as truth_p90.
    output_path : Path
        Base directory to save plots. Saved as: {output_path}/{var}/p90_by_nyear.png
    period_dim : str
        Name of the period coordinate/dimension (default "period").
    """
    if period_dim not in truth_p90.dims or period_dim not in pred_p90.dims:
        raise ValueError(f"Both datasets must contain dim '{period_dim}'")

    periods = truth_p90[period_dim].values
    n_rows = len(periods)

    for var in truth_p90.data_vars:
        fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])  # shape -> (1, 3)

        for i, p in enumerate(periods):
            label = str(p)
            t2d = truth_p90[var].sel({period_dim: p}).values
            p2d = pred_p90[var].sel({period_dim: p}).values

            # Generate plot for each period
            _plot_sample_images(
                axes, i,
                [t2d, p2d, p2d - t2d],  # images
                [   # titles
                    f"Truth p90 ({label})",
                    f"Prediction p90 ({label})",
                    f"Prediction - Truth ({label})",
                ],
                "plasma_r"  # error
            )

        plt.tight_layout()
        plt.savefig(output_path / f"{var}" / "p90_by_nyear.png")
        plt.close(fig)


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
