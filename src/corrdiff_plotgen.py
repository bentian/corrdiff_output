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

import xarray as xr

import plot_helper as ph
from samples_handler import (
    get_timestamp, save_masked_samples,
    score_samples, score_samples_multi_ensemble, N_YEARS
)
from analysis_utils import (
    ensure_directory_exists,
    yaml_to_tsv, read_training_loss_and_plot,
    save_tables_and_plots, group_by_nyear,
)

OVERVIEW_METRIC_FMT = ".2f"
DIFF_METRIC_FMT = ".3f"

# Model processing functions
def process_model(in_dir: Path, out_dir: Path, label: str,
                  n_ensemble: int, masked: bool) -> xr.Dataset:
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
    nc_path = in_dir / "netcdf" / f"output_0_{label}{'_masked' if masked else ''}.nc"

    # Score samples
    metrics, spatial_error, top_samples, flats, p90s = score_samples(nc_path, n_ensemble)

    # Create output directory and sub directories for each variable
    output_path = ensure_directory_exists(out_dir, label)
    for var in spatial_error.data_vars.keys():
        ensure_directory_exists(output_path, var)

    # Plots per variable
    ph.plot_pdf(*flats, output_path)
    ph.plot_monthly_error(spatial_error, output_path)
    ph.plot_p90_by_nyear(*p90s, output_path)
    for metric in ["MAE", "RMSE"]:
        ph.plot_metrics_cnt(metrics, metric, output_path)
        ph.plot_top_samples(top_samples, metric, output_path)

    # Plot metrics vs. # ensembles
    if label == "all" and n_ensemble == 64:
        metrics_by_n = score_samples_multi_ensemble(nc_path, n_ensembles=(1, 4, 16))
        ph.plot_metrics_vs_ensembles(
            [metrics_by_n[1], metrics_by_n[4], metrics_by_n[16]] + [metrics],
            output_path,
        )

    # Overview plots and tables
    save_tables_and_plots(
        metrics.mean(dim="time"),
        metrics.groupby("time.month").mean(dim="time"),   # group by month
        group_by_nyear(metrics, N_YEARS),                 # group by nyear
        ensure_directory_exists(output_path, "overview"),
        number_format=OVERVIEW_METRIC_FMT
    )

    return metrics


def compare_models(metrics_all: xr.Dataset, metrics_reg: xr.Dataset, output_path: Path) -> None:
    """
    Compare models and save results.

    Parameters:
        metrics_all (xr.Dataset): Metrics dataset for the full model.
        metrics_reg (xr.Dataset): Metrics dataset for the regression-only model.
        output_path (Path): Output directory where comparison results should be saved.
    """
    metrics_mean_diff = metrics_all.mean(dim="time") - metrics_reg.mean(dim="time")
    metrics_monthly_diff = (
        metrics_all.groupby("time.month").mean(dim="time")
        - metrics_reg.groupby("time.month").mean(dim="time")
    )
    save_tables_and_plots(
        metrics_mean_diff,
        metrics_monthly_diff,
        group_by_nyear(metrics_all - metrics_reg, N_YEARS),
        output_path,
        number_format=DIFF_METRIC_FMT
    )


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
    Main function to process models and generate plots and tables.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=Path, help="Folder to read the NetCDF files and config.")
    parser.add_argument("out_dir", type=Path, help="Folder to save the plots and tables.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    parser.add_argument("--masked", type=str, default="yes", help="Whether to apply landmask.")
    args = parser.parse_args()

    masked = args.masked.lower() == "yes"
    print(f"[{get_timestamp()}] corrdiff_plotgen: in_dir={args.in_dir} out_dir={args.out_dir} "
          f"n_ensemble={args.n_ensemble} masked={masked}")

    # Ensure masked NetCDF files exist
    if masked:
        for filename in ["output_0_all", "output_0_reg"]:
            src = args.in_dir / "netcdf" / f"{filename}.nc"
            dst = args.in_dir / "netcdf" / f"{filename}_masked.nc"
            if not dst.exists():
                save_masked_samples(src, dst)

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
        read_training_loss_and_plot(args.in_dir, args.out_dir, label)

    print(f"[{get_timestamp()}] corrdiff_plotgen completed")


if __name__ == "__main__":
    main()
