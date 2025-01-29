import os
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import plot_helpers as ph
from score_samples_v2 import score_samples
from mask_samples import save_masked_samples

# General utility functions
def ensure_directory_exists(dir, subdir=None):
    """
    Ensure the given directory exists.
    """
    directory = os.path.join(dir, subdir) if subdir else dir
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def read_tensorboard_log(log_dir, scalar_name="training_loss"):
    """
    Read TensorBoard logs and extract scalar values.
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    tags = event_acc.Tags().get('scalars', [])
    if scalar_name not in tags:
        raise ValueError(f"Scalar '{scalar_name}' not found. Available scalars: {tags}")
    scalar_events = event_acc.Scalars(scalar_name)
    wall_times = [datetime.fromtimestamp(event.wall_time) for event in scalar_events]
    values = [event.value for event in scalar_events]
    return wall_times, values


def read_training_loss_and_plot(in_dir, out_dir, label):
    """
    Read training loss from TensorBoard logs and plot.
    """
    log_dir = os.path.join(in_dir, f"tensorboard_{label}")
    wall_times, values = read_tensorboard_log(log_dir)
    ph.plot_training_loss(wall_times, values, os.path.join(out_dir, f"training_loss_{label}.png"))


# YAML processing functions
def yaml_to_tsv(yaml_file_path, tsv_filename):
    """
    Convert a YAML file to a TSV file.
    """
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    parsed_data = {key.strip(): value.strip() for item in data for key, value in [item.split("=", 1)]}
    df = pd.DataFrame([parsed_data]).transpose()
    df.to_csv(tsv_filename, sep='\t', index=True)


def save_to_tsv(ds, output_path, number_format):
    """
    Save a dataset to a TSV file.
    """
    ds.to_dataframe().to_csv(output_path, sep='\t', float_format=f"%{number_format}")


def save_metric_table_and_plot(ds, metric, output_path, number_format):
    """
    Save metric tables and generate plots.
    """
    ds_filtered = ds.sel(metric=metric).drop_vars("metric")
    filename = os.path.join(output_path, f"monthly_{metric}")
    save_to_tsv(ds_filtered, f"{filename}.tsv", number_format)
    ph.plot_monthly_metrics(ds_filtered, metric, f"{filename}.png", number_format)


def save_tables_and_plots(ds_mean, ds_group_by_month, output_path, number_format=".2f"):
    """
    Save summary tables and plots for metrics.
    """
    filename = os.path.join(output_path, "metrics_mean")
    save_to_tsv(ds_mean, f"{filename}.tsv", number_format)
    ph.plot_metrics(ds_mean, f"{filename}.png", number_format)
    for metric in ["mae", "rmse"]:
        save_metric_table_and_plot(ds_group_by_month, metric, output_path, number_format)


# Model processing functions
def process_model(in_dir, out_dir, label, n_ensemble, masked):
    """
    Process a model, generate plots, and save metrics.
    """
    suffix = '_masked' if masked else ''
    nc_file = os.path.join(in_dir, "netcdf", f"output_0_{label}{suffix}.nc")
    metrics, spatial_error, truth_flat, pred_flat = score_samples(nc_file, n_ensemble)
    output_path = ensure_directory_exists(out_dir, label)

    ph.plot_monthly_error(spatial_error, output_path)
    ph.plot_pdf(truth_flat, pred_flat, output_path)
    metric_mean = metrics.mean(dim="time")
    metrics_grouped = metrics.groupby("time.month").mean(dim="time")
    save_tables_and_plots(metric_mean, metrics_grouped, output_path)

    return metrics


def compare_models(metrics_all, metrics_reg, output_path):
    """
    Compare models and save results.
    """
    metrics_mean_diff = metrics_all.mean(dim="time") - metrics_reg.mean(dim="time")
    metrics_grouped_diff = (
        metrics_all.groupby("time.month").mean(dim="time")
        - metrics_reg.groupby("time.month").mean(dim="time")
    )
    save_tables_and_plots(metrics_mean_diff, metrics_grouped_diff, output_path, number_format=".3f")


# Main workflow
def process_models(in_dir, out_dir, n_ensemble, masked):
    """
    Process models and generate results.
    """
    # Process regression + diffusion model
    metrics_all = process_model(in_dir, out_dir, "all", n_ensemble, masked)

    # Process regression-only model
    metrics_reg = process_model(in_dir, out_dir, "reg", n_ensemble, masked)

    # Compare models
    compare_models(metrics_all, metrics_reg, ensure_directory_exists(out_dir, "minus_reg"))


def main():
    """
    Main function to process models and generate plots and summary PDFs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str, help="Folder to read the NetCDF files and config.")
    parser.add_argument("out_dir", type=str, help="Folder to save the plots and tables.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    parser.add_argument("--masked", type=str, default='yes', help="Whether to apply landmask")
    args = parser.parse_args()

    # Some Python argument parsers, such as argparse, automatically treat boolean flags as True
    # if they are present, even if the value is explicitly set to False.
    masked = True if args.masked.lower() == 'yes' else False
    print(f'corrdiff_plotgen: in_dir={args.in_dir} out_dir={args.out_dir} masked={masked}')

    # Ensure masked NetCDF files exist
    if masked:
        for filename in ['output_0_all', 'output_0_reg']:
            masked_file = os.path.join(args.in_dir, "netcdf", f"{filename}_masked.nc")
            if not os.path.exists(masked_file):
                save_masked_samples(
                    os.path.join(args.in_dir, "netcdf", f"{filename}.nc"), masked_file)

    ensure_directory_exists(args.out_dir)
    process_models(args.in_dir, args.out_dir, args.n_ensemble, masked)

    # Process Hydra config
    config = os.path.join(args.in_dir, "hydra", "overrides.yaml")
    yaml_to_tsv(config, os.path.join(args.out_dir, "generate_overrides.tsv"))

    # Process training loss
    for label in ["regression", "diffusion"]:
        read_training_loss_and_plot(args.in_dir, args.out_dir, label)


if __name__ == "__main__":
    main()
