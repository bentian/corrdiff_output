"""
CLI entry point for generating evaluation plots and tables for CorrDiff models.

This script orchestrates:
- running scoring on truth/pred samples (via score_samples_v2)
- producing per-variable plots (PDFs, monthly errors, top samples, p90 maps)
- exporting summary tables/plots (mean, monthly, N-year)
- optionally computing metrics vs ensemble size
- exporting Hydra configs to TSV
- plotting training loss from TensorBoard
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import xarray as xr

from samples_handler import (
    get_timestamp,
    save_masked_samples,
    score_samples,
    score_samples_multi_ensemble,
    N_YEARS,
)
from analysis_utils import (
    ensure_directory_exists,
    yaml_to_tsv,
    read_training_loss_and_plot,
    save_tables_and_plots,
    group_by_nyear,
)
from plot_helper import (
    plot_pdf,
    plot_rank_histogram,
    plot_monthly_error,
    plot_p90_by_nyear,
    plot_metrics_cnt,
    plot_top_samples,
    plot_metrics_vs_ensembles,
)

OVERVIEW_METRIC_FMT = ".2f"
DIFF_METRIC_FMT = ".3f"


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _is_bcsd(in_dir: Path) -> bool:
    """Check if the input directory contains BCSD data."""
    return (in_dir / "bcsd_masked.nc").exists()


def _get_nc_path(in_dir: Path, label: str, masked: bool) -> Path:
    """Get the NetCDF path for the given model label and masking option."""
    return (
        in_dir / "bcsd_masked.nc"
        if _is_bcsd(in_dir)
        else in_dir / "netcdf" / f"output_0_{label}{'_masked' if masked else ''}.nc"
    )


def _plot_diagnostics(
    scored: tuple,
    output_path: Path,
) -> xr.Dataset:
    """Generate per-variable diagnostic plots."""
    metrics, rank_histograms, spatial_error, top_samples, flats, p90s = scored

    # Create per-variable output directories
    for var in spatial_error.data_vars.keys():
        ensure_directory_exists(output_path, var)

    # Plots per variable
    plot_pdf(*flats, output_path)
    plot_rank_histogram(rank_histograms, output_path)
    plot_monthly_error(spatial_error, output_path)
    plot_p90_by_nyear(*p90s, output_path)
    for metric in ["MAE", "RMSE"]:
        plot_metrics_cnt(metrics, metric, output_path)
        plot_top_samples(top_samples, metric, output_path)

    return metrics


def _plot_metrics_vs_ensembles(
    nc_path: Path, metrics: xr.Dataset, output_path: Path
) -> None:
    """Plot metrics vs. number of ensembles."""
    metrics_by_n = score_samples_multi_ensemble(nc_path, n_ensembles=(1, 4, 16))
    plot_metrics_vs_ensembles(
        [metrics_by_n[1], metrics_by_n[4], metrics_by_n[16], metrics],
        output_path,
    )


def _save_overview(metrics: xr.Dataset, output_path: Path) -> None:
    """Save overview plots and tables."""
    save_tables_and_plots(
        metrics.mean(dim="time"),
        metrics.groupby("time.month").mean(dim="time"),  # group by month
        group_by_nyear(metrics, N_YEARS),  # group by nyear
        ensure_directory_exists(output_path, "overview"),
        number_format=OVERVIEW_METRIC_FMT,
    )


def _mean_diff(
    a: xr.Dataset, b: xr.Dataset, groupby: Optional[str] = None
) -> xr.Dataset:
    """Compute the mean difference between two datasets, optionally grouped by a time dimension."""
    return (
        a.groupby(groupby).mean("time") - b.groupby(groupby).mean("time")
        if groupby
        else a.mean("time") - b.mean("time")
    )


# -----------------------------------------------------------------------------
# Model processing functions
# -----------------------------------------------------------------------------
def process_model(
    in_dir: Path, out_dir: Path, label: str, n_ensemble: int, masked: bool
) -> xr.Dataset:
    """
    Process a model, generate plots, and save metrics.

    Parameters:
        in_dir (Path): Input directory containing NetCDF files.
        out_dir (Path): Output directory to save results.
        label (str): Model label (e.g., "all", "reg").
        n_ensemble (int): Number of ensemble members.
        masked (bool): Whether to apply landmask.

    Returns:
        xr.Dataset: Computed metrics dataset.
    """
    nc_path = _get_nc_path(in_dir, label, masked)
    output_path = ensure_directory_exists(out_dir, label)

    # Score samples and generate plots
    metrics = _plot_diagnostics(
        score_samples(nc_path, n_ensemble, is_bcsd=_is_bcsd(in_dir)),
        output_path,
    )
    _save_overview(metrics, output_path)

    # Plot metrics vs. # ensembles
    if label == "all" and n_ensemble == 64:
        _plot_metrics_vs_ensembles(nc_path, metrics, output_path)

    return metrics


def compare_models(
    metrics_all: xr.Dataset, metrics_reg: xr.Dataset, output_path: Path
) -> None:
    """
    Compare models and save results.

    Parameters:
        metrics_all (xr.Dataset): Metrics dataset for the full model.
        metrics_reg (xr.Dataset): Metrics dataset for the regression-only model.
        output_path (Path): Output directory where comparison results should be saved.
    """
    save_tables_and_plots(
        _mean_diff(metrics_all, metrics_reg),
        _mean_diff(metrics_all, metrics_reg, "time.month"),
        group_by_nyear(metrics_all - metrics_reg, N_YEARS),
        output_path,
        number_format=DIFF_METRIC_FMT,
    )


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
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

    # Process regression-only model with n_ensemble=1
    metrics_reg = process_model(in_dir, out_dir, "reg", 1, masked)

    # Compare models
    compare_models(
        metrics_all, metrics_reg, ensure_directory_exists(out_dir, "minus_reg")
    )


def main():
    """
    Main function to process models and generate plots and tables.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_dir", type=Path, help="Folder to read the NetCDF files and config."
    )
    parser.add_argument(
        "out_dir", type=Path, help="Folder to save the plots and tables."
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=1, help="Number of ensemble members."
    )
    parser.add_argument(
        "--no-mask",
        dest="masked",
        action="store_false",
        help="Disable landmask.",
    )
    parser.set_defaults(masked=True)

    args = parser.parse_args()
    print(
        f"[{get_timestamp()}] corrdiff_plotgen: in_dir={args.in_dir} out_dir={args.out_dir} "
        f"n_ensemble={args.n_ensemble} masked={args.masked}"
    )

    # Handle BCSD input
    if _is_bcsd(args.in_dir):
        process_model(args.in_dir, args.out_dir, "all", n_ensemble=1, masked=True)
        return

    ### Non-BCSD input only

    # Ensure masked NetCDF files exist
    if args.masked:
        for filename in ["output_0_all", "output_0_reg"]:
            src = args.in_dir / "netcdf" / f"{filename}.nc"
            dst = args.in_dir / "netcdf" / f"{filename}_masked.nc"
            if not dst.exists():
                save_masked_samples(src, dst)

    # Process models
    ensure_directory_exists(args.out_dir)
    process_models(args.in_dir, args.out_dir, args.n_ensemble, args.masked)

    # Process Hydra config
    for key, output_file in [
        ("hydra_generate", "generate_config.tsv"),
        ("hydra_train", "train_config.tsv"),
    ]:
        config_path = args.in_dir / key / "config.yaml"
        if config_path.exists():
            yaml_to_tsv(config_path, args.out_dir / output_file)

    # Process training loss
    for label in ["regression", "diffusion"]:
        read_training_loss_and_plot(args.in_dir, args.out_dir, label)

    print(f"[{get_timestamp()}] corrdiff_plotgen completed")


if __name__ == "__main__":
    main()
