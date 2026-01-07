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
    """
    Arrange a flat list of metrics into a 2x2 grid layout.

    Parameters
    ----------
    metrics : list[str]
        List of metric names (expected length = 4).

    Returns
    -------
    list[list[str]]
        Metrics arranged row-wise for 2x2 subplot indexing.
    """
    return [metrics[i:i + 2] for i in range(0, len(metrics), 2)]


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


# ----------------------------
# mean
# ----------------------------
def _plot_metric_all_groups(ax: plt.Axes, df: pd.DataFrame, *,
                            metric: str, variable: str, group_gap: float = 1.0) -> None:
    """
    Plot mean metric values for all experiment groups in a single subplot.

    The x-axis shows experiments ordered within each group, with visual gaps
    separating groups. Bars compare ``all`` vs ``reg`` labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target subplot axis.
    df : pd.DataFrame
        Long-form DataFrame produced by ``extract_metrics``.
    metric : str
        Metric to plot.
    variable : str
        Variable to plot.
    group_gap : float, optional
        Horizontal spacing between experiment groups (default: 1.0).
    """
    sub = df.query("metric == @metric and variable == @variable")
    if sub.empty:
        ax.set_visible(False)
        return

    wide = (
        sub.pivot_table(index=["group", "experiment"], columns="label",
                        values="value", aggfunc="mean")
          .reset_index()
          .assign(exp_sort=lambda d: d["experiment"].map(experiment_sort_key))
          .sort_values(["group", "exp_sort"])
          .drop(columns="exp_sort")
    )

    labels = [lab for lab in ("all", "reg") if lab in wide.columns]
    if not labels:
        ax.set_visible(False)
        return

    sizes = wide.groupby("group", sort=False).size().to_numpy()
    starts = np.r_[0.0, np.cumsum(sizes[:-1] + group_gap)]
    x = np.concatenate([s + np.arange(n) for s, n in zip(starts, sizes)])

    bar_w = 0.8 / len(labels)
    colors = {"all": "tab:blue", "reg": "tab:olive"}

    for i, lab in enumerate(labels):
        ax.bar(x + i * bar_w, wide[lab].to_numpy(float), width=bar_w,
               label=lab, color=colors.get(lab), edgecolor="black")

    ax.set_xticks(x + bar_w * ((len(labels) / 2) - 0.5))
    ax.set_xticklabels(wide["experiment"], rotation=45, ha="right")

    # separators + group labels
    for b in (starts[1:] - group_gap / 2):
        ax.axvline(b, linestyle="--", alpha=0.35)

    centers = starts + (sizes - 1) / 2
    for grp, cx in zip(wide["group"].drop_duplicates().tolist(), centers):
        ax.text(cx, 1.02, grp, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set(title=metric, ylabel=metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    _apply_ylim(ax, variable, metric)


def plot_metrics_cmp(df: pd.DataFrame, metrics: list[str],
                     variables: list[str], folder_path: str) -> None:
    """
    Generate 2x2 mean-metric comparison plots for each variable.

    For each variable, creates a figure with one subplot per metric and
    saves it to disk. All experiment groups are shown in each subplot.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``extract_metrics``.
    metrics : list[str]
        Metrics to plot (expected length = 4).
    variables : list[str]
        Variables to plot (e.g. ``["prcp", "t2m"]``).
    folder_path : str
        Output directory for saved figures.
    """
    if df.empty:
        print("No metrics found.")
        return

    folder = Path(folder_path)

    for var in variables:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        grid = _metric_grid(metrics)

        for r in range(2):
            for c in range(2):
                _plot_metric_all_groups(
                    axes[r, c],
                    df,
                    metric=grid[r][c],
                    variable=var,
                )

        # ---- shared legend (from first axis that has handles) ----
        handles, labels = [], []
        for ax in axes.ravel():
            axis_handles, axis_labels = ax.get_legend_handles_labels()
            if axis_handles:
                handles, labels = axis_handles, axis_labels
                break

        if handles:
            fig.legend(
                handles,
                labels,
                title="Label",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
            )

        fig.suptitle(f"Metric Mean Comparison ({var})", y=1.02)
        plt.tight_layout()

        out = folder / f"{var}_mean_cmp.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ----------------------------
# nyear
# ----------------------------
def _plot_nyear_metric(ax: plt.Axes, df: pd.DataFrame, *,
                       metric: str, variable: str, label_mode: LabelMode = "all") -> None:
    """
    Plot decadal (year-bin) metric values for a single metric.

    The x-axis shows ``year_bin`` and the y-axis shows metric values.
    Lines compare experiments; ``all`` and ``reg`` may be distinguished
    by linestyle depending on ``label_mode``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target subplot axis.
    df : pd.DataFrame
        Output of ``extract_nyear_metrics``.
    metric : str
        Metric to plot.
    variable : str
        Variable to plot.
    label_mode : {"all", "reg", "both"}, optional
        Which evaluation labels to include (default: ``"all"``).
    """
    sub = df.query("metric == @metric and variable == @variable")
    if label_mode != "both":
        sub = sub.query("label == @label_mode")
    if sub.empty:
        ax.set_visible(False)
        return

    exps = (
        sub[["group", "experiment"]]
        .drop_duplicates()
        .assign(exp_sort=lambda d: d["experiment"].map(experiment_sort_key))
        .sort_values(["group", "exp_sort"])
    )["experiment"].tolist()

    cmap = plt.get_cmap("tab10")
    exp2color = {e: cmap(i % 10) for i, e in enumerate(exps)}
    ls_map = {"all": "-", "reg": "--"}

    group_cols = ["experiment", "label"] if label_mode == "both" else ["experiment"]
    for keys, g in sub.groupby(group_cols, sort=False):
        exp, lab = keys if label_mode == "both" else \
            (keys if isinstance(keys, str) else keys[0], label_mode)
        ax.plot(
            g.sort_values("year_bin")["year_bin"].astype(str),
            g.sort_values("year_bin")["value"].to_numpy(float),
            linestyle=ls_map.get(lab, "-"),
            label=f"{exp} ({lab})" if label_mode == "both" else exp,
            color=exp2color.get(exp),
            linewidth=1.8,
            marker="o",
            markersize=3.5,
        )

    ax.set(title=metric, xlabel="year_bin", ylabel=metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    _apply_ylim(ax, variable, metric)


def plot_nyear_metrics_cmp(df: pd.DataFrame, metrics: list[str], variables: list[str],
                           folder_path: str, label_mode: LabelMode = "all") -> None:
    """
    Generate 2x2 decadal-metric comparison plots for each variable.

    For each variable, creates a figure with one subplot per metric,
    plotting metric values across year bins and experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``extract_nyear_metrics``.
    metrics : list[str]
        Metrics to plot (expected length = 4).
    variables : list[str]
        Variables to plot.
    folder_path : str
        Output directory for saved figures.
    label_mode : {"all", "reg", "both"}, optional
        Which evaluation labels to include in the plots.
    """
    if df.empty:
        print("No nyear metrics found.")
        return

    folder = Path(folder_path)
    for var in variables:
        grid = _metric_grid(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        for r in range(2):
            for c in range(2):
                _plot_nyear_metric(axes[r, c], df, metric=grid[r][c],
                                   variable=var, label_mode=label_mode)

        # one shared legend (pull from first visible axis)
        handles, labels = [], []
        for ax in axes.ravel():
            axis_handles, axis_labels = ax.get_legend_handles_labels()
            if axis_handles:
                handles, labels = axis_handles, axis_labels
                break

        if handles:
            fig.legend(handles, labels,
                       title="Experiment (label)" if label_mode == "both" else "Experiment",
                       loc="center left", bbox_to_anchor=(1.01, 0.5))

        fig.suptitle(f"Decadal Metric Comparison ({var})", y=1.02)
        plt.tight_layout()
        out = folder / f"{var}_nyear_cmp.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
