import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import plot_helpers as ph
from score_samples_v2 import score_samples
from mask_samples import save_masked_samples


# General utility functions
def ensure_directory_exists(directory: Path, subdir: Optional[str] = None) -> Path:
    """
    Ensure the given directory exists.

    Parameters:
        directory (Path): The base directory.
        subdir (Optional[str]): A subdirectory to be created inside `directory`. Defaults to None.

    Returns:
        Path: The path of the created directory.
    """
    path = directory / subdir if subdir else directory
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_tensorboard_log(log_dir: Path, scalar_name: str = "training_loss",
                         max_duration: float = None) -> Tuple[List[datetime], List[float]]:
    """
    Read TensorBoard logs and extract scalar values.

    Parameters:
        log_dir (Path): Path to the TensorBoard log directory.
        scalar_name (str): The name of the scalar to extract. Defaults to "training_loss".
        max_duration (float): The maximum training duration (in steps) to include.
                              If None, includes all data.

    Returns:
        Tuple[List[datetime], List[float]]: Lists of timestamps and corresponding scalar values.
    """
    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()
    tags = event_acc.Tags().get("scalars", [])

    if scalar_name not in tags:
        raise ValueError(f"Scalar '{scalar_name}' not found. Available scalars: {tags}")

    scalar_events = event_acc.Scalars(scalar_name)
    wall_times = [datetime.fromtimestamp(event.wall_time) for event in scalar_events]
    values = [event.value for event in scalar_events]
    steps = [event.step for event in scalar_events]

    # Apply filtering based on max_duration
    if max_duration is not None:
        wall_times, values = zip(*[(t, v) for t, v, s in zip(wall_times, values, steps)
                                   if s <= max_duration])

    return wall_times, values


def read_training_loss_and_plot(in_dir: Path, out_dir: Path, label: str,
                                max_duration: float = None) -> None:
    """
    Read training loss from TensorBoard logs and generate a plot.

    Parameters:
        in_dir (Path): Input directory containing TensorBoard logs.
        out_dir (Path): Output directory to save the loss plot.
        label (str): Label for the model (e.g., "regression", "diffusion").
        max_duration (float): The maximum training duration (in steps) to include.
                              If None, includes all data.
    """
    log_dir = in_dir / f"tensorboard_{label}"
    wall_times, values = read_tensorboard_log(log_dir, max_duration=max_duration)
    ph.plot_training_loss(wall_times, values, out_dir / f"training_loss_{label}.png")


# YAML processing functions
def yaml_to_tsv(yaml_file_path: Path, tsv_filename: Path) -> None:
    """
    Convert a YAML file to a TSV file.

    Parameters:
        yaml_file_path (Path): Path to the YAML file.
        tsv_filename (Path): Path where the TSV file should be saved.
    """
    with yaml_file_path.open("r") as file:
        data = yaml.safe_load(file)

    # Flatten the YAML structure into key-value pairs for a table
    def flatten_dict(d, parent_key='', sep='.'):
        """
        Recursively flattens a nested dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_data = flatten_dict(data)
    df = pd.DataFrame([flat_data]).transpose()
    df.to_csv(tsv_filename, sep="\t", index=True)


def save_to_tsv(ds: pd.DataFrame, output_path: Path, number_format: str) -> None:
    """
    Save a dataset to a TSV file.

    Parameters:
        ds (pd.DataFrame): The dataset to save.
        output_path (Path): Path where the TSV file should be saved.
        number_format (str): Format string for floating-point numbers.
    """
    ds.to_dataframe().to_csv(output_path, sep="\t", float_format=f"%{number_format}")


def save_metric_table_and_plot(ds: pd.DataFrame, metric: str,
                               output_path: Path, number_format: str) -> None:
    """
    Save metric tables and generate plots.

    Parameters:
        ds (pd.DataFrame): Dataset containing metric values.
        metric (str): The metric to process (e.g., "MAE", "RMSE").
        output_path (Path): Path where the TSV and plot files should be saved.
        number_format (str): Format string for floating-point numbers.
    """
    ds_filtered = ds.sel(metric=metric).drop_vars("metric")
    filename = output_path / f"monthly_{metric.lower()}"
    save_to_tsv(ds_filtered, filename.with_suffix(".tsv"), number_format)
    ph.plot_monthly_metrics(ds_filtered, metric, filename.with_suffix(".png"), number_format)


def save_tables_and_plots(ds_mean: pd.DataFrame, ds_group_by_month: pd.DataFrame,
                          output_path: Path, number_format: str = ".2f") -> None:
    """
    Save summary tables and generate plots for metrics.

    Parameters:
        ds_mean (pd.DataFrame): Mean dataset of metrics.
        ds_group_by_month (pd.DataFrame): Monthly grouped dataset of metrics.
        output_path (Path): Output directory where files should be saved.
        number_format (str): Format string for floating-point numbers. Defaults to ".2f".
    """
    filename = output_path / "metrics_mean"
    save_to_tsv(ds_mean, filename.with_suffix(".tsv"), number_format)
    ph.plot_metrics(ds_mean, filename.with_suffix(".png"), number_format)

    for metric in ["MAE", "RMSE"]:
        save_metric_table_and_plot(ds_group_by_month, metric, output_path, number_format)


# Model processing functions
def process_model(in_dir: Path, out_dir: Path, label: str,
                  n_ensemble: int, masked: bool) -> pd.DataFrame:
    """
    Process a model, generate plots, and save metrics.

    Parameters:
        in_dir (Path): Input directory containing NetCDF files.
        out_dir (Path): Output directory to save results.
        label (str): Model label (e.g., "all", "reg").
        n_ensemble (int): Number of ensemble members.
        masked (bool): Whether to apply landmask.

    Returns:
        pd.DataFrame: Computed metrics dataset.
    """
    suffix = "_masked" if masked else ""
    nc_file = in_dir / "netcdf" / f"output_0_{label}{suffix}.nc"
    metrics, spatial_error, truth_flat, pred_flat = score_samples(nc_file, n_ensemble)
    output_path = ensure_directory_exists(out_dir, label)

    for var in spatial_error.data_vars.keys():
        ensure_directory_exists(output_path, var) # Create subdir for each variable
    ph.plot_monthly_error(spatial_error, output_path)
    ph.plot_pdf(truth_flat, pred_flat, output_path)

    metric_mean = metrics.mean(dim="time")
    metrics_grouped = metrics.groupby("time.month").mean(dim="time")
    save_tables_and_plots(
        metric_mean, metrics_grouped, ensure_directory_exists(output_path, "overview"))

    return metrics


def compare_models(metrics_all: pd.DataFrame, metrics_reg: pd.DataFrame,
                   output_path: Path) -> None:
    """
    Compare models and save results.

    Parameters:
        metrics_all (pd.DataFrame): Metrics dataset for the full model.
        metrics_reg (pd.DataFrame): Metrics dataset for the regression-only model.
        output_path (Path): Output directory where comparison results should be saved.
    """
    metrics_mean_diff = metrics_all.mean(dim="time") - metrics_reg.mean(dim="time")
    metrics_grouped_diff = (
        metrics_all.groupby("time.month").mean(dim="time")
        - metrics_reg.groupby("time.month").mean(dim="time")
    )
    save_tables_and_plots(metrics_mean_diff, metrics_grouped_diff,
                          output_path, number_format=".3f")


# Main workflow
def process_models(in_dir: Path, out_dir: Path, n_ensemble: int, masked: bool) -> None:
    """
    Process models and generate results.

    Parameters:
        in_dir (Path): Input directory containing model data.
        out_dir (Path): Output directory to save results.
        n_ensemble (int): Number of ensemble members.
        masked (bool): Whether to apply landmask.
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
    parser.add_argument("in_dir", type=Path, help="Folder to read the NetCDF files and config.")
    parser.add_argument("out_dir", type=Path, help="Folder to save the plots and tables.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    parser.add_argument("--masked", type=str, default="yes", help="Whether to apply landmask.")
    parser.add_argument("--max-duration", type=float, help="Max duration to plot training loss.")
    args = parser.parse_args()

    masked = args.masked.lower() == "yes"
    print(f"corrdiff_plotgen: in_dir={args.in_dir} out_dir={args.out_dir} masked={masked}")

    # Ensure masked NetCDF files exist
    if masked:
        for filename in ["output_0_all", "output_0_reg"]:
            masked_file = args.in_dir / "netcdf" / f"{filename}_masked.nc"
            if not masked_file.exists():
                save_masked_samples(args.in_dir / "netcdf" / f"{filename}.nc", masked_file)

    # Process models
    ensure_directory_exists(args.out_dir)
    process_models(args.in_dir, args.out_dir, args.n_ensemble, masked)

    # Process Hydra config
    for key, output_file in \
        [("hydra_generate", "generate_config.tsv"), ("hydra_train", "train_config.tsv")]:
        config_path = args.in_dir / key / "config.yaml"
        if config_path.exists():
            yaml_to_tsv(config_path, args.out_dir / output_file)

    # Process training loss
    for label in ["regression", "diffusion"]:
        read_training_loss_and_plot(args.in_dir, args.out_dir, label, args.max_duration)


if __name__ == "__main__":
    main()
