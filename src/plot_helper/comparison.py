"""
Utilities for comparing experiment evaluation metrics across variables and time scales.

This module provides plotting helpers for:
  1) Mean metric comparisons across experiments and groups
  2) Decadal (year-bin) metric comparisons across experiments

Supported features:
  - Automatic grouping of experiments by prefix (e.g. W1, W2)
  - Consistent experiment ordering (e.g. W1-1a, W1-1, W1-2, ...)
  - Side-by-side comparison of multiple metrics in 2x2 subplot layouts
  - Support for multiple variables (e.g. precipitation and 2m temperature)
  - Optional comparison of evaluation labels (all / reg)
  - Metric- and variable-specific y-axis limits for clearer visualization

Typical usage:
  - Prepare a long-form metrics DataFrame (mean or decadal)
  - Call ``plot_metrics_cmp`` for mean metrics
  - Call ``plot_nyear_metrics_cmp`` for year-bin metrics

Output figures are saved to disk under the specified folder path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
Y_LIMITS = {
    "prcp": {
        "RMSE": (10.0, None),  # e.g. (0.0, 8.0)
        "MAE":  (4.5, None),   # e.g. (0.0, 4.0)
        "CRPS": (4.5, None),   # e.g. (0.0, 4.0)
        "CORR": (0.25, 0.5),   # correlations benefit most from tight limits
    },
    "t2m": {
        "RMSE": (0.8, None),
        "MAE":  (0.5, None),
        "CRPS": (0.5, None),
        "CORR": (0.95, 1.0),
    }
}
LabelMode = Literal["all", "reg", "both"]  # both = plot all+reg


# ----------------------------
# helpers
# ----------------------------
def experiment_sort_key(exp: str) -> tuple[int, int, int, str]:
    """
    Sort order:
      W1-1a, W1-1, W1-2, ...
      W2-1, W2-2, ...

    Returns a tuple:
      (group_number, experiment_number, letter_priority, letter)
    """
    m = re.match(r"W(\d+)-(\d+)([a-z]?)", exp)
    if not m:
        return (9999, 9999, 1, exp)

    group_num = int(m.group(1))
    exp_num = int(m.group(2))
    letter = m.group(3)

    # letter priority:
    #   'a' -> 0 (comes before plain number)
    #   ''  -> 1
    letter_priority = 0 if letter == "a" else 1

    return (group_num, exp_num, letter_priority, letter)


def _metric_grid(metrics: list[str]) -> list[list[str]]:
    """Arrange a flat list of metrics into a 2x2 grid layout."""
    return [metrics[i:i + 2] for i in range(0, len(metrics), 2)]


def _mean_wide(df: pd.DataFrame, metric: str, variable: str) -> pd.DataFrame:
    """Filter long df and pivot to wide (group/experiment index, label columns)."""
    sub = df.query("metric == @metric and variable == @variable")
    if sub.empty:
        return sub

    return (
        sub.pivot_table(index=["group", "experiment"], columns="label",
                        values="value", aggfunc="mean")
        .reset_index()
        .assign(_s=lambda d: d["experiment"].map(experiment_sort_key))
        .sort_values(["group", "_s"])
        .drop(columns="_s")
    )


def _bar_positions(wide: pd.DataFrame, group_gap: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute x positions, group start positions, and group sizes (in row order)."""
    sizes = wide.groupby("group", sort=False).size().to_numpy()
    starts = np.r_[0.0, np.cumsum(sizes[:-1] + group_gap)]
    x = np.concatenate([st + np.arange(n) for st, n in zip(starts, sizes)])
    return x, starts, sizes


def _draw_group_labels(ax: plt.Axes, wide: pd.DataFrame,
                       starts: np.ndarray, sizes: np.ndarray) -> None:
    """Draw vertical separators and group labels."""
    centers = starts + (sizes - 1) / 2
    groups = wide["group"].drop_duplicates().tolist()

    for sep in starts[1:]:
        ax.axvline(sep, linestyle="--", alpha=0.35)

    for grp, cx in zip(groups, centers):
        ax.text(
            cx,
            1.02,
            grp,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def _apply_ylim(ax: plt.Axes, variable: str, metric: str) -> None:
    """
    Apply variable- and metric-specific y-axis limits to a subplot.

    Limits are read from the global ``Y_LIMITS`` configuration.
    If no limits are defined, the axis is left unchanged.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target subplot axis.
    variable : str
        Variable name (e.g. ``"prcp"``, ``"t2m"``).
    metric : str
        Metric name (e.g. ``"RMSE"``, ``"CRPS"``).
    """
    ylim = Y_LIMITS.get(variable, {}).get(metric)
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])


def _first_legend(axes: np.ndarray):
    """Return the first non-empty (handles, labels) found in a grid of axes."""
    for one_ax in axes.ravel():
        axis_handles, axis_labels = one_ax.get_legend_handles_labels()
        if axis_handles:
            return axis_handles, axis_labels
    return [], []


# ----------------------------
# mean
# ----------------------------
def _plot_metric_all_groups(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
    variable: str,
    group_gap: float = 1.0,
) -> None:
    """
    Plot mean metric values for all experiment groups in a single subplot.
    Bars compare ``all`` vs ``reg``; groups are separated by a visual gap.
    """
    mean_wide = _mean_wide(df, metric, variable)
    if mean_wide.empty:
        ax.set_visible(False)
        return

    label_list = [lab for lab in ("all", "reg") if lab in mean_wide.columns]
    if not label_list:
        ax.set_visible(False)
        return

    x_pos, group_starts, group_sizes = _bar_positions(mean_wide, group_gap)
    bar_w = 0.8 / len(label_list)
    color_map = {"all": "tab:blue", "reg": "tab:olive"}

    # bars
    for i, lab in enumerate(label_list):
        ax.bar(
            x_pos + i * bar_w,
            mean_wide[lab].to_numpy(float),
            width=bar_w,
            label=lab,
            color=color_map.get(lab),
            edgecolor="black",
        )

    # xtick labels (e.g., "W1-1a")
    ax.set_xticks(x_pos + bar_w * ((len(label_list) / 2) - 0.5))
    ax.set_xticklabels(mean_wide["experiment"], rotation=45, ha="right")

    # separators + group labels (e.g., "W1")
    _draw_group_labels(ax, mean_wide, group_starts, group_sizes)

    # title, labels, and grid
    ax.set(title=metric, ylabel=metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # ylimits
    _apply_ylim(ax, variable, metric)


def plot_metrics_cmp(df: pd.DataFrame, metrics: list[str],
                     variables: list[str], folder_path: Path) -> None:
    """Create and save 2x2 mean-metric comparison figures (one per variable)."""
    if df.empty:
        print("No metrics found.")
        return

    grid = _metric_grid(metrics)

    for var in variables:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        for r in range(2):
            for c in range(2):
                _plot_metric_all_groups(axes[r, c], df, metric=grid[r][c], variable=var)

        handles, labels = _first_legend(axes)
        if handles:
            fig.legend(handles, labels, title="Label",
                       loc="center left", bbox_to_anchor=(1.01, 0.5))

        fig.suptitle(f"Metric Mean Comparison ({var})", y=1.02)
        plt.tight_layout()

        out_path = folder_path / f"{var}_mean_cmp.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ----------------------------
# nyear
# ----------------------------
def _plot_nyear_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
    variable: str,
    label_mode: LabelMode,
) -> None:
    """Plot year-bin metric curves for one metric/variable (optionally all/reg/both)."""
    sub_df = df.query("metric == @metric and variable == @variable")
    if label_mode != "both":
        sub_df = sub_df.query("label == @label_mode")
    if sub_df.empty:
        ax.set_visible(False)
        return

    exp_df = (
        sub_df[["group", "experiment"]]
        .drop_duplicates()
        .assign(exp_sort=lambda d: d["experiment"].map(experiment_sort_key))
        .sort_values(["group", "exp_sort"])
    )

    cmap = plt.get_cmap("tab10")
    exp_color = {e: cmap(i % 10) for i, e in enumerate(exp_df["experiment"].tolist())}

    for keys, part_df in sub_df.groupby(["experiment", "label"], sort=False):
        exp_name, lab = keys

        ordered = part_df.sort_values("year_bin")
        ax.plot(
            ordered["year_bin"].astype(str),
            ordered["value"].to_numpy(float),
            linestyle="--" if lab == "reg" else "-",
            label=f"{exp_name} ({lab})",
            color=exp_color.get(exp_name),
            linewidth=1.8,
            marker="o",
            markersize=3.5,
        )

    ax.set(title=metric, xlabel="Decade", ylabel=metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    _apply_ylim(ax, variable, metric)


def plot_nyear_metrics_cmp(
    df: pd.DataFrame,
    metrics: list[str],
    variables: list[str],
    folder_path: Path,
    label_mode: LabelMode = "all",
) -> None:
    """Create and save 2x2 year-bin comparison figures (one per variable)."""
    if df.empty:
        print("No nyear metrics found.")
        return

    grid = _metric_grid(metrics)

    for var in variables:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        for r in range(2):
            for c in range(2):
                _plot_nyear_metric(axes[r, c], df, metric=grid[r][c],
                                   variable=var, label_mode=label_mode)

        handles, labels = _first_legend(axes)
        if handles:
            fig.legend(handles, labels, title="Experiment (label)",
                       loc="center left", bbox_to_anchor=(1.01, 0.5))

        fig.suptitle(f"Decadal Metric Comparison ({var})", y=1.02)
        plt.tight_layout()

        out_path = folder_path / f"{var}_nyear_cmp.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
