"""
Shared utilities for processing and exporting model evaluation results.

This module contains reusable helpers used by the CLI/runner code, including:
- filesystem helpers for creating output directories
- TensorBoard scalar extraction and training-loss plotting
- Hydra YAML flattening and YAML->TSV conversion
- TSV export for xarray datasets
- metric table export + plotting wrappers
- grouping time-indexed metric datasets into fixed N-year bins
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import xarray as xr
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plot_helper import (
    plot_training_loss,
    plot_metrics,
    plot_monthly_metrics,
    plot_nyear_metrics,
)


def ensure_directory_exists(directory: Path, subdir: Optional[str] = None) -> Path:
    """Create `directory[/subdir]` if needed and return the resulting path."""
    path = directory / subdir if subdir else directory
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# TensorBoard utilities
# -----------------------------------------------------------------------------
def read_tensorboard_log(log_dir: Path, scalar_name: str = "training_loss",
                         max_duration: float = None) -> Tuple[List[datetime], List[float]]:
    """
    Read TensorBoard event files in `log_dir` and extract (wall_time, value) pairs
    for `scalar_name`. Optionally filter by `step <= max_duration`.
    """
    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()
    tags = event_acc.Tags().get("scalars", [])

    if scalar_name not in tags:
        raise ValueError(f"Scalar '{scalar_name}' not found. Available scalars: {tags}")

    scalar_events = event_acc.Scalars(scalar_name)
    wall_times = [datetime.fromtimestamp(event.wall_time) for event in scalar_events]
    values = [e.value for e in scalar_events]
    steps = [e.step for e in scalar_events]

    # Apply filtering based on max_duration
    if max_duration is not None:
        wall_times, values = zip(*[(t, v) for t, v, s in zip(wall_times, values, steps)
                                   if s <= max_duration])

    return wall_times, values


def read_training_loss_and_plot(in_dir: Path, out_dir: Path, label: str,
                                max_duration: float = None) -> None:
    """Read `training_loss` from TensorBoard logs and save a plot."""
    log_dir = in_dir / f"tensorboard_{label}"
    wall_times, values = read_tensorboard_log(log_dir, max_duration=max_duration)
    plot_training_loss(wall_times, values, out_dir / f"training_loss_{label}.png")


# -----------------------------------------------------------------------------
# YAML / TSV utilities
# -----------------------------------------------------------------------------
def _flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dict into a single-level dict using dot-separated keys."""
    out = {}

    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key, sep=sep))
        else:
            out[key] = v

    return out


def yaml_to_tsv(yaml_file_path: Path, tsv_filename: Path) -> None:
    """Convert a Hydra YAML config into a TSV (flattened key/value table)."""
    with yaml_file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    flat = _flatten_dict(data)
    pd.DataFrame([flat]).T.to_csv(tsv_filename, sep="\t", index=True)


def save_to_tsv(ds: xr.Dataset, output_path: Path, number_format: str) -> None:
    """Save an xarray Dataset to TSV."""
    ds.to_dataframe().to_csv(output_path, sep="\t", float_format=f"%{number_format}")


def save_metric_table_and_plot(
    ds: xr.Dataset,
    metric: str,
    output_path: Path,
    number_format: str,
) -> None:
    """Save TSV + plot for `metric` for the given dataset."""
    ds_filtered = ds.sel(metric=metric).drop_vars("metric")
    filename = output_path / metric.lower()
    save_to_tsv(ds_filtered, filename.with_suffix(".tsv"), number_format)

    plot_fn = plot_monthly_metrics if output_path.name.startswith("monthly") \
                else plot_nyear_metrics
    plot_fn(ds_filtered, metric, filename.with_suffix(".png"), number_format)


def save_tables_and_plots(
    ds_mean: xr.Dataset,
    ds_group_by_month: xr.Dataset,
    ds_group_by_nyear: Optional[xr.Dataset],
    output_path: Path,
    number_format: str = ".2f",
) -> None:
    """Save summary TSVs and plots for mean / monthly / N-year grouped metrics."""
    filename = output_path / "metrics_mean"
    save_to_tsv(ds_mean, filename.with_suffix(".tsv"), number_format)
    plot_metrics(ds_mean, filename.with_suffix(".png"), number_format)

    for metric in ["MAE", "RMSE", "CORR", "CRPS", "STD_DEV"]:
        save_metric_table_and_plot(
            ds_group_by_month, metric,
            ensure_directory_exists(output_path, "monthly_metrics"), number_format
        )
        if ds_group_by_nyear is not None:
            save_metric_table_and_plot(
                ds_group_by_nyear, metric,
                ensure_directory_exists(output_path, "nyear_metrics"), number_format
            )


# -----------------------------------------------------------------------------
# Time grouping utilities
# -----------------------------------------------------------------------------
def group_by_nyear(metrics: xr.Dataset, n_years: int) -> Optional[xr.Dataset]:
    """
    Group `metrics` into fixed N-year bins starting at the first year present.
    Returns None if only a single year is present.
    """
    years = metrics["time"].dt.year
    base, max_year = int(years.min()), int(years.max())
    if base == max_year:
        return None     # Skip grouping if only single year data

    bin_start = base + ((years - base) // n_years) * n_years
    bin_end = (bin_start + (n_years - 1)).clip(max=max_year)
    # Make string labels like "2015-2020"
    year_bin = bin_start.astype(str) + "-" + bin_end.astype(str)

    return (
        metrics.assign_coords(year_bin=("time", year_bin.data))
        .groupby("year_bin")
        .mean(dim="time")
    )
