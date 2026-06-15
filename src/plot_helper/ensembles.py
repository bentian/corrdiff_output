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
    Shows how forecast metrics change as the number of ensemble members
    increases.

Rank histogram
    Evaluates ensemble calibration by comparing the rank of observations
    relative to ensemble forecasts. Histograms are plotted as relative
    frequencies with a uniform-reference line.

BiasScore
    Derived from the mean rank.

    - BiasScore < 0 : forecast tends to be too large
    - BiasScore > 0 : forecast tends to be too small
    - BiasScore ≈ 0 : unbiased

DispersionScore
    Derived from the variance of the rank histogram relative to a
    uniform distribution.

    - DispersionScore > 1 : U-shaped (under-dispersed)
    - DispersionScore < 1 : dome-shaped (over-dispersed)
    - DispersionScore ≈ 1 : well dispersed

Monthly rank-score plot
    Displays monthly BiasScore and DispersionScore on a 2-D plane,
    allowing seasonal changes in ensemble calibration to be identified.

Notes
-----
- Functions operate on aggregated xarray datasets produced by the scoring pipeline.
- Figures are saved to variable-specific output directories.
- Rank-histogram scores can be computed from fully aggregated,
  monthly-aggregated, or other grouped rank histograms.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from .samples import COLOR_MAPS


def _score_rank_histogram(rank_histograms: xr.Dataset) -> xr.Dataset:
    """
    Compute bias and dispersion scores from daily rank histograms.

    Expected input:
        rank_histograms[var]: dims (time, rank)

    Returns:
        xr.Dataset with metric dimension:
            BiasScore:
                < 0 forecast positive bias
                > 0 forecast negative bias
                ~ 0 unbiased

            DispersionScore:
                > 1 U-shape / under-dispersed
                < 1 dome-shape / over-dispersed
                ~ 1 well dispersed
    """
    if "rank" not in rank_histograms.dims:
        raise ValueError("rank_histograms must include a 'rank' dimension")

    ranks = rank_histograms["rank"]
    n_rank = rank_histograms.sizes["rank"]

    center = ranks.mean()
    half_range = (ranks.max() - ranks.min()) / 2
    uniform_var = (n_rank**2 - 1) / 12

    out = {}
    for var, hist in rank_histograms.data_vars.items():
        total = hist.sum("rank")
        mean_rank = (hist * ranks).sum("rank") / total
        rank_var = (hist * (ranks - mean_rank) ** 2).sum("rank") / total

        out[var] = xr.concat(
            [(mean_rank - center) / half_range, rank_var / uniform_var],
            dim=xr.IndexVariable("score", ["BiasScore", "DispersionScore"]),
        )

    return xr.Dataset(out)


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
    if "rank" not in rank_histograms.dims:
        raise ValueError("rank_histograms must include a 'rank' dimension")

    scores = _score_rank_histogram(rank_histograms)
    n_members = rank_histograms.sizes["rank"] - 1

    for i, (var, hist) in enumerate(rank_histograms.data_vars.items()):
        ranks = hist["rank"].values
        counts = hist.values.astype(float)
        freq = counts / np.nansum(counts)

        # xskillscore returns rank labels as floats in examples; use compact
        # integer-like labels when possible.
        labels = [str(int(r)) if float(r).is_integer() else str(r) for r in ranks]
        total = int(np.nansum(counts))
        color = plt.get_cmap(COLOR_MAPS[i % len(COLOR_MAPS)])(0.6)

        plt.figure(figsize=(10, 6))
        plt.bar(labels, freq, color=color, edgecolor="black", alpha=0.75)
        plt.axhline(
            1 / len(ranks) if len(ranks) else 0,
            linestyle="--",
            linewidth=1,
            label="Uniform reference",
        )

        bias_score, dispersion_score = scores[var].values
        plt.title(
            f"Rank histogram of {var}\n({total:,} pts, {n_members} members; "
            f"BiasScore={bias_score:.2f}, DispersionScore={dispersion_score:.2f})"
        )
        plt.xlabel("Truth rank among ensemble members")
        plt.ylabel("Relative frequency")
        plt.legend()
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        plt.savefig(output_path / f"{var}" / "rank_histogram.png")
        plt.close()


def plot_monthly_rank_scores(rank_histograms: xr.Dataset, output_path: Path) -> None:
    """
    Plot monthly rank-histogram scores for each variable.
        X-axis: Bias Score
        Y-axis: Dispersion Score

    Each point represents one month and is labeled 1-12.
    """
    if "rank" not in rank_histograms.dims:
        raise ValueError("rank_histograms must include a 'rank' dimension")
    if "month" not in rank_histograms.dims:
        raise ValueError("rank_histograms must include a 'month' dimension")

    scores = _score_rank_histogram(rank_histograms)
    n_members = rank_histograms.sizes["rank"] - 1

    for i, var in enumerate(scores.data_vars):
        bias = scores[var].sel(score="BiasScore")
        dispersion = scores[var].sel(score="DispersionScore")
        color = plt.get_cmap(COLOR_MAPS[i % len(COLOR_MAPS)])(0.6)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            bias.values,
            dispersion.values,
            color=color,
            edgecolor="black",
            alpha=0.85,
            zorder=3,
        )

        for month, x, y in zip(scores["month"].values, bias.values, dispersion.values):
            plt.annotate(
                str(int(month)),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )

        plt.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        plt.axhline(1, color="black", linestyle="--", linewidth=1, alpha=0.6)

        plt.title(f"Monthly rank histogram scores of {var}\n({n_members} members)")
        plt.xlabel("Bias Score")
        plt.ylabel("Dispersion Score")
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()

        plt.savefig(output_path / f"{var}" / "monthly_rank_scores.png")
        plt.close()
