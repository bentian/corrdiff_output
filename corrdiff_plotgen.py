import os
import argparse
from pathlib import Path

import plot_helpers as ph
from score_samples_v2 import score_samples
from generate_summary import generate_summary

def save_to_csv(ds, output_path, number_format=".2f"):
    ds.to_dataframe().to_csv(output_path, float_format=f"%{number_format}")


def save_metric_table_n_plot(ds, metric, output_path_prefix):
    ds_filtered = ds.sel(metric=metric).drop_vars("metric")
    filename = f"{output_path_prefix}-monthly_{metric}"

    save_to_csv(ds_filtered, f"{filename}.csv")
    ph.plot_monthly_metrics(ds_filtered, metric, f"{filename}.png")


def save_tables_n_plot(ds_mean, ds_group_by_month, output_path_prefix, number_format=".2f"):
    filename = f"{output_path_prefix}-metrics_mean"
    save_to_csv(ds_mean, f"{filename}.csv", number_format)
    ph.plot_metrics(ds_mean, f"{filename}.png", number_format)

    for metric in ["mae", "rmse"]:
        save_metric_table_n_plot(ds_group_by_month, metric, output_path_prefix)


def process_model(metrics, spatial_error, truth_flat, pred_flat, output_path_prefix):
    """
    Process a model (regression + diffusion), plot results, and save metrics.

    Parameters:
        metrics (xarray.Dataset): Metrics dataset from `score_samples`.
        spatial_error (xarray.Dataset): Spatial error dataset from `score_samples`.
        truth_flat (xarray.DataArray): Flattened truth values from `score_samples`.
        pred_flat (xarray.DataArray): Flattened prediction values from `score_samples`.
        output_path_prefix (str): Output path prefix for saving results.
    """
    # Plot monthly mean of spatial error
    ph.plot_monthly_error(spatial_error, output_path_prefix)

    # Plot PDF for flattened truth and prediction
    ph.plot_pdf(truth_flat, pred_flat, output_path_prefix)

    # Compute mean and grouped metrics
    metric_mean = metrics.mean(dim="time")
    metrics_grouped = metrics.groupby("time.month").mean(dim="time")

    # Save tables and plots
    save_tables_n_plot(metric_mean, metrics_grouped, output_path_prefix)


def compare_models(metrics_all, metrics_reg, output_path_prefix):
    """
    Compare regression + diffusion model with regression only model and save results.

    Parameters:
        metrics_all (xarray.Dataset): Metrics dataset for regression + diffusion model.
        metrics_reg (xarray.Dataset): Metrics dataset for regression only model.
        output_path_prefix (str): Output path prefix for saving results.
    """
    # Compute differences
    metrics_mean_diff = metrics_all.mean(dim="time") - metrics_reg.mean(dim="time")
    metrics_grouped_diff = (
        metrics_all.groupby("time.month").mean(dim="time")
        - metrics_reg.groupby("time.month").mean(dim="time")
    )

    # Save tables and plots
    save_tables_n_plot(metrics_mean_diff, metrics_grouped_diff,
                       output_path_prefix, number_format=".3f")

def main():
    """
    Main function to process models and generate plots and summary PDFs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("nc_all", type=str, help="The NetCDF file path of regression + diffusion.")
    parser.add_argument("nc_reg", type=str, help="The NetCDF file path of regression only.")
    parser.add_argument("outdir", type=str, help="Folder to save the plots.")
    parser.add_argument("--prefix", type=str, default="B", help="Prefix for the output files.")
    parser.add_argument("--summary", type=str, default="False", help="Generate summary or not.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Process regression + diffusion model
    metrics_all, spatial_error, truth_flat, pred_flat = score_samples(args.nc_all, args.n_ensemble)
    process_model(metrics_all, spatial_error, truth_flat, pred_flat, os.path.join(args.outdir, "all"))

    # Process regression-only model and compare
    metrics_reg = score_samples(args.nc_reg, args.n_ensemble, metrics_only=True)
    compare_models(metrics_all, metrics_reg, os.path.join(args.outdir, "minus_reg"))

    # Generate summary PDF
    if args.summary.lower() == "true":
        generate_summary(args.outdir, os.path.join(args.outdir, f"{args.prefix}_summary.pdf"))


if __name__ == "__main__":
    main()
