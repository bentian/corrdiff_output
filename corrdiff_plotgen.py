import os
import argparse

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from score_samples_v2 import score_samples, VAR_MAPPING
from generate_summary import generate_summary

def plot_metrics(ds, output_path, number_format):
    metrics = ds["metric"].values
    variables = list(ds.data_vars.keys())
    data_array = np.array([ds[var] for var in variables])  # Shape: (4, 4)

    # Bar chart
    x = np.arange(len(metrics))  # Metric indices
    width = 0.2  # Bar width

    _, ax = plt.subplots(figsize=(10, 6))
    for i, var in enumerate(variables):
        bars = ax.bar(x + i * width, data_array[i], width, label=var)
        # Add value annotations on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:{number_format}}", # Text to display
                xy=(bar.get_x() + bar.get_width() / 2, height),  # X and Y position
                xytext=(0, 5),               # Offset text by 5 units above the bar
                textcoords="offset points",  # Interpret `xytext` as an offset
                ha='center', va='bottom',    # Align horizontally (center) and vertically (bottom)
                fontsize=10, color='black'   # Optional styling
            )

    ax.set_title("Metrics Mean", fontsize=14)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.set_xticks(x + width * (len(variables) - 1) / 2)
    ax.set_xticklabels([metric.upper() for metric in metrics])
    ax.legend(title="Variables")
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)

def plot_monthly_metrics(ds, metric, output_path):
    _, ax = plt.subplots(figsize=(10, 6))

    df_grouped = ds.to_dataframe().round(2)
    for variable in df_grouped.columns:
        ax.plot(df_grouped.index, df_grouped[variable], marker="o", label=variable)
        for x, y in zip(df_grouped.index, df_grouped[variable]):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    ax.set_title(f"Monthly Mean for {metric.upper()}", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(f"{metric.upper()} Value", fontsize=12)
    ax.set_xticks(np.arange(1, 13))
    ax.legend(title="Variables")
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)

def get_bin_count_n_note(ds, bin_width=1):
    min_val, max_val = ds.min().item(), ds.max().item()
    bin_count = int((max_val - min_val) / bin_width)
    return bin_count, f"({len(ds):,} pts in [{min_val:.1f}, {max_val:.1f}])"

def plot_pdf(truth, pred, output_path_prefix):
    """
    Plot PDFs for all variables in the truth dataset, comparing with prediction.

    Parameters:
        truth (xarray.Dataset): Truth dataset.
        pred (xarray.Dataset): Prediction dataset.
        output_path_prefix (str): Path prefix for saving output plots.
    """
    for var in truth.data_vars:
        if var in pred:  # Ensure the variable exists in the prediction dataset
            truth_flat = truth[var].values.flatten()
            pred_flat = pred[var].mean("ensemble").values.flatten() \
                        if "ensemble" in pred.dims else pred[var].values.flatten()

            truth_bin_count, truth_note = get_bin_count_n_note(truth_flat)
            pred_bin_count, pred_note = get_bin_count_n_note(pred_flat)
            print(f"Variable: {var} | PDF bin count: {truth_bin_count} (truth) / {pred_bin_count} (pred)")

            plt.figure(figsize=(10, 6))
            plt.hist(truth_flat, bins=truth_bin_count, alpha=0.5, label="Truth", density=True)
            plt.hist(pred_flat, bins=pred_bin_count, alpha=0.5, label="Prediction", density=True)
            plt.title(f"PDF of {var}:\nTruth {truth_note} /\nPrediction {pred_note}")
            plt.xlabel(f"{var} (units)")
            plt.ylabel("Density")
            plt.legend()
            plt.grid()

            if var == 'precipitation':
                plt.xlim([-5, 15])  # Adjust x-limits based on expected variable range

            # Save the figure with variable-specific filename
            plt.savefig(f"{output_path_prefix}-pdf_{VAR_MAPPING[var]}.png")
            plt.close()

def plot_monthly_error(ds, output_path_prefix):
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
        fig.suptitle(f"Monthly Mean Error of {var}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        plt.savefig(f"{output_path_prefix}-monthly_error_{VAR_MAPPING[var]}.png")

def save_to_csv(ds, output_path, number_format=".2f"):
    ds.to_dataframe().to_csv(output_path, float_format=f"%{number_format}")

def save_metric_table_n_plot(ds, metric, output_path_prefix):
    ds_filtered = ds.sel(metric=metric).drop_vars("metric")
    filename = f"{output_path_prefix}-monthly_{metric}"

    save_to_csv(ds_filtered, f"{filename}.csv")
    plot_monthly_metrics(ds_filtered, metric, f"{filename}.png")

def save_tables_n_plot(ds_mean, ds_group_by_month, output_path_prefix, number_format=".2f"):
    filename = f"{output_path_prefix}-metrics_mean"
    save_to_csv(ds_mean, f"{filename}.csv", number_format)
    plot_metrics(ds_mean, f"{filename}.png", number_format)

    for metric in ["mae", "rmse"]:
        save_metric_table_n_plot(ds_group_by_month, metric, output_path_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nc_all", type=str, help="Path for the netcdf file of regression + diffusion.")
    parser.add_argument("nc_reg", type=str, help="Path for the netcdf file of regression only.")
    parser.add_argument("outdir", type=str, help="Folder to save the plots.")
    parser.add_argument("--prefix", type=str, default="Baseline", help="Prefix for the output files.")
    parser.add_argument("--summarize", type=str, default="False", help="Whether to generate summary PDF.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    args = parser.parse_args()

    ### Regression + Diffusion Model

    # Process prediction and truth samples of the regression + diffusion model
    metrics, spatial_error, truth_flat, pred_flat = score_samples(args.nc_all, args.n_ensemble)

    # Plot monthly mean of spatial error, and PDF
    output_path_prefix_all = os.path.join(args.outdir, "all")
    plot_monthly_error(spatial_error, output_path_prefix_all)
    plot_pdf(truth_flat, pred_flat, output_path_prefix_all)

    # Aggregate metrics to create plots and tables
    metric_mean = metrics.mean(dim="time")
    metrics_grouped = metrics.groupby("time.month").mean(dim="time")
    save_tables_n_plot(metric_mean, metrics_grouped, output_path_prefix_all)

    ### Regression + Diffusion Model minus Regression modle only

    # Compare Regression + Diffusion model with Regression only model
    metrics_reg = score_samples(args.nc_reg, args.n_ensemble, metrics_only=True)
    metrics_mean_diff = metric_mean - metrics_reg.mean(dim="time")
    metrics_grouped_diff = metrics_grouped - metrics_reg.groupby("time.month").mean(dim="time")
    save_tables_n_plot(metrics_mean_diff, metrics_grouped_diff,
                       os.path.join(args.outdir, "minus_reg"), number_format=".3f")

    # Generate summary PDF
    if args.summarize:
        generate_summary(args.outdir, args.prefix)
