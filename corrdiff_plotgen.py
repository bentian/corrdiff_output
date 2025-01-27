import os
import argparse
from pathlib import Path
import pandas as pd
import yaml

import plot_helpers as ph
from score_samples_v2 import score_samples

def yaml_to_csv(yaml_file_path, csv_filename):
    # Load the YAML content into a Python dictionary
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Convert the list into a dictionary
    parsed_data = {}
    for item in data:
        key, value = item.split("=", 1)  # Split on the first '='
        value = value.strip()

        # Handle different value types explicitly
        if value.startswith("[") and value.endswith("]"):  # Handle lists
            parsed_data[key.strip()] = value.replace(",", "_")
        elif value.lower() == "null":  # Handle null values
            parsed_data[key.strip()] = None
        else:  # Treat everything else as a string
            parsed_data[key.strip()] = value.replace(",", "_")

    # Normalize the dictionary into a flat table
    df = pd.DataFrame([parsed_data])

    # Export the DataFrame to a CSV file
    df.transpose().to_csv(csv_filename, index=True)

    return df

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
    parser.add_argument("in_dir", type=str, help="Folder to read the NetCDF files and config.")
    parser.add_argument("out_dir", type=str, help="Folder to save the plots and tables.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Process regression + diffusion model
    nc_all = os.path.join(args.in_dir, "netcdf", "output_0_all.nc")
    metrics_all, spatial_error, truth_flat, pred_flat = score_samples(nc_all, args.n_ensemble)
    process_model(metrics_all, spatial_error, truth_flat, pred_flat, os.path.join(args.out_dir, "all"))

    # Process regression-only model and compare
    nc_reg = os.path.join(args.in_dir, "netcdf", "output_0_reg.nc")
    metrics_reg = score_samples(nc_reg, args.n_ensemble, metrics_only=True)
    compare_models(metrics_all, metrics_reg, os.path.join(args.out_dir, "minus_reg"))

    # Store hydra config table
    config = os.path.join(args.in_dir, "hydra", "overrides.yaml")
    yaml_to_csv(config, os.path.join(args.out_dir, "generate_config.csv"))

if __name__ == "__main__":
    main()
