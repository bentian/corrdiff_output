"""
Plotting utilities for ensemble forecast diagnostics and calibration.

This module provides visualizations and summary diagnostics for evaluating
ensemble forecast performance, including:

- forecast metrics versus ensemble size
- rank histograms (Talagrand diagrams)
- monthly calibration summaries based on rank-histogram scores

Available diagnostics
---------------------
Ensemble-size sensitivity
    Shows how forecast metrics change as the number of ensemble members increases.

Rank histogram
    Evaluates ensemble calibration by comparing the rank of observations relative to
    ensemble forecasts. Histograms are plotted as relative frequencies with a uniform-
    reference line.

Monthly rank histogram
    Same as rank histogram, but for each month.

Notes
-----
- Functions operate on aggregated xarray datasets produced by the scoring pipeline.
- Figures are saved to variable-specific output directories.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from .samples import COLOR_MAPS


def _require_dims(ds: xr.Dataset, dims: set[str]) -> None:
    """Ensure the dataset contains the required dimensions."""
    missing = dims - set(ds.dims)
    if missing:
        raise ValueError(
            f"rank_histograms missing dimensions: {', '.join(sorted(missing))}"
        )


def _rank_labels(hist: xr.DataArray) -> list[str]:
    """Get rank labels from the rank histogram."""
    return [f"{r:g}" for r in hist["rank"].values]


def _rank_freq(hist: xr.DataArray) -> np.ndarray:
    """Get normalized rank frequencies from the rank histogram."""
    values = hist.values.astype(float)
    total = np.nansum(values)
    return values / total if total else values


def _plot_rank_bars(ax, hist: xr.DataArray, color) -> None:
    """Plot rank histogram bars on the given axes."""
    ax.bar(
        _rank_labels(hist),
        _rank_freq(hist),
        color=color,
        edgecolor="black",
        alpha=0.75,
    )
    ax.axhline(
        1 / hist.sizes["rank"],
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.6,
        label="Uniform reference",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_rank_histogram(rank_histograms: xr.Dataset, output_path: Path) -> None:
    """
    Plot rank histograms (Talagrand diagrams) for each variable.

    Parameters
    ----------
    rank_histograms : xr.Dataset
        Output from ``xskillscore.rank_histogram`` merged into a dataset, with one
        variable per forecast variable and a ``rank`` dimension.
    output_path : Path
        Base output directory. Each figure is saved to ``<output_path>/<var>/rank_histogram.png``.
    """
    _require_dims(rank_histograms, {"rank"})
    n_members = rank_histograms.sizes["rank"] - 1

    for i, (var, hist) in enumerate(rank_histograms.data_vars.items()):
        values = hist.values.astype(float)
        color = plt.get_cmap(COLOR_MAPS[i % len(COLOR_MAPS)])(0.6)

        _, ax = plt.subplots(figsize=(10, 6))
        _plot_rank_bars(ax, hist, color)

        ax.set_title(
            f"Rank histogram of {var}\n({int(np.nansum(values)):,} pts, {n_members} members)"
        )
        ax.set_xlabel("Truth rank among ensemble members")
        ax.set_ylabel("Relative frequency")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path / var / "rank_histogram.png")
        plt.close()


def plot_monthly_rank_histogram(rank_histograms: xr.Dataset, output_path: Path) -> None:
    """
    Plot monthly rank histograms as a 3x4 panel for each variable.

    Each subplot shows the rank histogram for one calendar month, normalized to relative frequency
    and compared against the uniform reference distribution. The resulting figure provides
    a compact view of seasonal changes in ensemble calibration.

    Parameters
    ----------
    rank_histograms : xr.Dataset
        Monthly aggregated rank histograms with dimensions ``month`` and ``rank``.
        Each data variable corresponds to a forecast variable.
    output_path : Path
        Base output directory. Figures are saved to
        ``<output_path>/<var>/monthly_rank_histogram.png``
        for each forecast variable.
    """

    _require_dims(rank_histograms, {"rank", "month"})
    n_members = rank_histograms.sizes["rank"] - 1

    for i, var in enumerate(rank_histograms.data_vars):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        color = plt.get_cmap(COLOR_MAPS[i % len(COLOR_MAPS)])(0.6)

        for month, ax in zip(range(1, 13), axes):
            hist = rank_histograms[var].sel(month=month)

            _plot_rank_bars(ax, hist, color)
            ax.set_title(f"Month {month}", fontsize=10)
            ax.set_ylim(
                0,
                max(
                    np.nanmax(_rank_freq(hist)) * 1.15,
                    1 / hist.sizes["rank"] * 1.5,
                ),
            )

        fig.suptitle(
            f"Monthly Rank Histogram of {var}\n({n_members} members)", fontsize=16
        )
        fig.supxlabel("Truth rank among ensemble members")
        fig.supylabel("Relative frequency")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(output_path / var / "monthly_rank_histogram.png")
        plt.close()
