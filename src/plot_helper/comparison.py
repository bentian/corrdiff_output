"""
Plotting utilities for experiment metric comparison.

This module provides functions to visualize evaluation metrics across
experiments, variables, and time scales.

Features
--------
- 2x2 subplot layouts for multiple metrics
- Consistent experiment ordering (via `experiment_sort_key`)
- Line plots for mean metrics across experiment suffixes
- Line plots for decadal (year-bin) metrics
- Distinct visual encoding:
    - color   → experiment group (e.g. W1a, W1, W2)
    - linestyle / marker → evaluation label (all / reg)

Typical usage
-------------
1. Prepare a long-form DataFrame (via extraction utilities)
2. Call:
    - `plot_metrics_cmp` for mean metrics
    - `plot_nyear_metrics_cmp` for decadal metrics

Figures are saved to disk in the specified output folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from typing import Literal
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Y_LIMITS = {
    "prcp": {
        "RMSE": (9.0, None),
        "MAE": (4, None),
        "CRPS": (4, None),
        "CORR": (0.1, 0.5),
    },
    "t2m": {
        "RMSE": (0.7, None),
        "MAE": (0.5, None),
        "CRPS": (0.5, None),
        "CORR": (0.9, 1.0),
    },
    **{
        k: {
            "RMSE": (0.4, None),
            "MAE": (0.2, None),
            "CRPS": (0.2, None),
            "CORR": (0.4, 1.0),
        }
        for k in ["u10m", "v10m"]
    },
}
LabelMode = Literal["all", "reg", "both"]

SUFFIX_ORDER = ["-1", "-2", "-3", "-5"]
GROUP_COLORS = {"W1a": "tab:olive", "W1": "tab:blue", "W2": "tab:orange"}
LAB_STYLE = {
    "all": {"linestyle": "-", "marker": "o"},
    "reg": {"linestyle": "--", "marker": "^"},
}


@dataclass(frozen=True)
class MetricGridSpec:
    """
    Configuration for generating and saving metric comparison figures.

    Attributes
    ----------
    metrics : list[str]
        Metrics to plot (max 4 used per 2x2 grid, in display order).
    variables : list[str]
        Variables to generate figures for (one figure per variable).
    folder_path : Path
        Output directory where figures will be saved.
    filename_suffix : str
        Suffix appended to output filenames (e.g. "mean_cmp", "nyear_cmp").
    title_prefix : str
        Prefix used in the figure title (e.g. "Metric Mean Comparison").
    """

    metrics: list[str]
    variables: list[str]
    folder_path: Path
    filename_suffix: str
    title_prefix: str


def experiment_sort_key(exp: str) -> tuple[int, int, int]:
    """
    Generate a sortable key for experiment names.
    Ensures ordering such as:
        W1a-* before W1-*, then increasing experiment number

    Parameters
    ----------
    exp : str
        Experiment name (e.g. "W1a-1", "W1-2", "W2-1").

    Returns
    -------
    tuple[int, int, int]
        Sorting key: (group_number, group_variant_priority, experiment_number)
    """
    m = re.match(r"W(\d+)(a?)-(\d+)([a-z]?)$", exp)
    if not m:
        return (9999, 9999, 9999)
    return (
        int(m.group(1)),
        0 if m.group(2) == "a" else 1,
        int(m.group(3)),
    )


def _apply_ylim(ax: plt.Axes, variable: str, metric: str) -> None:
    """Apply y-axis limits based on variable and metric."""
    ylim = Y_LIMITS.get(variable, {}).get(metric)
    if ylim:
        ax.set_ylim(*ylim)


def _first_legend(axes: np.ndarray):
    """Get the first legend from a grid of axes."""
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            return handles, labels
    return [], []


def _sorted_subset(df: pd.DataFrame, *, metric: str, variable: str) -> pd.DataFrame:
    """
    Filter and sort a DataFrame for a given metric and variable.
    Sorting is performed using `experiment_sort_key`.
    """
    sub = df[(df["metric"] == metric) & (df["variable"] == variable)].copy()
    if sub.empty:
        return sub
    return (
        sub.assign(exp_sort=sub["experiment"].map(experiment_sort_key))
        .sort_values("exp_sort")
        .drop(columns="exp_sort")
    )


def _save_metric_grid(
    df: pd.DataFrame,
    spec: MetricGridSpec,
    plot_one: Callable,
) -> None:
    """Save a 2x2 grid of plots for the given metrics and variables."""
    if df.empty:
        print(f"No {spec.filename_suffix} data found.")
        return

    for var in spec.variables:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()

        for ax, metric in zip(axes, spec.metrics[:4]):
            plot_one(ax, df, metric=metric, variable=var)

        for ax in axes[len(spec.metrics) :]:
            ax.set_visible(False)

        handles, labels = _first_legend(axes)
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))

        fig.suptitle(f"{spec.title_prefix} ({var})", y=1.02)
        plt.tight_layout()

        out_path = spec.folder_path / f"{var}_{spec.filename_suffix}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def _plot_metric_all_groups(
    ax: plt.Axes, df: pd.DataFrame, *, metric: str, variable: str
) -> None:
    """
    Plot mean metric values across experiments in a single subplot.

    Each line represents a (group, label) pair:
        - color   → group (W1a, W1, W2)
        - linestyle / marker → label (all, reg)

    X-axis corresponds to experiment suffixes (e.g. -1, -2, -3, -5).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target subplot.
    df : pd.DataFrame
        Long-form metrics DataFrame.
    metric : str
        Metric to plot.
    variable : str
        Variable to plot.
    """
    sub = _sorted_subset(df, metric=metric, variable=variable)
    if sub.empty:
        ax.set_visible(False)
        return

    for (group, label), g in sub.groupby(["group", "label"], sort=False):
        y = (
            g.assign(suffix=g["experiment"].str.removeprefix(group))
            .set_index("suffix")["value"]
            .reindex(SUFFIX_ORDER)
            .to_numpy()
        )
        ax.plot(
            SUFFIX_ORDER,
            y,
            color=GROUP_COLORS.get(group),
            label=f"{group} ({label})",
            **LAB_STYLE[label],
        )

    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    _apply_ylim(ax, variable, metric)


def plot_metrics_cmp(
    df: pd.DataFrame, metrics: list[str], variables: list[str], folder_path: Path
) -> None:
    """
    Create and save 2x2 comparison plots for mean metrics.

    One figure is generated per variable, containing up to four metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form metrics DataFrame.
    metrics : list[str]
        Metrics to include (max 4 displayed).
    variables : list[str]
        Variables to plot (one figure per variable).
    folder_path : Path
        Output directory for saved figures.
    """
    spec = MetricGridSpec(
        metrics=metrics,
        variables=variables,
        folder_path=folder_path,
        filename_suffix="mean_cmp",
        title_prefix="Metric Mean Comparison",
    )
    _save_metric_grid(df, spec, _plot_metric_all_groups)


def _plot_nyear_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
    variable: str,
    label_mode: LabelMode,
) -> None:
    """
    Plot decadal (year-bin) metric curves.

    Each line represents an experiment and label combination.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target subplot.
    df : pd.DataFrame
        Long-form decadal metrics DataFrame.
    metric : str
        Metric to plot.
    variable : str
        Variable to plot.
    label_mode : {"all", "reg", "both"}
        Which evaluation labels to include.
    """
    sub = _sorted_subset(df, metric=metric, variable=variable)
    if label_mode != "both":
        sub = sub[sub["label"] == label_mode]
    if sub.empty:
        ax.set_visible(False)
        return

    exps = sub["experiment"].drop_duplicates().tolist()
    cmap = plt.get_cmap("tab10")
    exp_color = {exp: cmap(i % 10) for i, exp in enumerate(exps)}

    for (exp_name, label), g in sub.groupby(["experiment", "label"], sort=False):
        g = g.sort_values("year_bin")
        ax.plot(
            g["year_bin"].astype(str),
            g["value"].to_numpy(float),
            color=exp_color[exp_name],
            label=f"{exp_name} ({label})",
            linewidth=1.8,
            markersize=3.5,
            **LAB_STYLE[label],
        )

    ax.set_title(metric)
    ax.set_xlabel("Decade")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    _apply_ylim(ax, variable, metric)


def plot_nyear_metrics_cmp(
    df: pd.DataFrame,
    metrics: list[str],
    variables: list[str],
    folder_path: Path,
    label_mode: LabelMode = "both",
) -> None:
    """
    Create and save 2x2 comparison plots for decadal metrics.

    One figure is generated per variable.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form decadal metrics DataFrame.
    metrics : list[str]
        Metrics to include.
    variables : list[str]
        Variables to plot.
    folder_path : Path
        Output directory.
    label_mode : {"all", "reg", "both"}, optional
        Which evaluation labels to include.
    """
    spec = MetricGridSpec(
        metrics=metrics,
        variables=variables,
        folder_path=folder_path,
        filename_suffix="nyear_cmp",
        title_prefix="Decadal Metric Comparison",
    )

    _save_metric_grid(
        df,
        spec,
        lambda ax, data, metric, variable: _plot_nyear_metric(
            ax, data, metric=metric, variable=variable, label_mode=label_mode
        ),
    )
