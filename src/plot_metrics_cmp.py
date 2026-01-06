"""
Module for extracting and plotting metric values for precipitation (prcp)
and 2m temperature (t2m) across experiments, grouped automatically by prefix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Literal
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
VARS = ["prcp", "t2m"]
METRICS = ["RMSE", "CORR", "MAE", "CRPS"]  # STD_DEV skipped
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
def parse_experiment_name(name: str) -> tuple[str, str]:
    """
    Examples
    --------
    W1-1a        -> group=W1, exp=W1-1a
    W1-1a_2M     -> group=W1, exp=W1-1a
    W2-3_4M      -> group=W2, exp=W2-3
    """
    base = name.split("_", 1)[0]
    group = base.split("-", 1)[0]
    return group, base


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


# ----------------------------
# extraction
# ----------------------------
def extract_metrics(folder_path: str) -> pd.DataFrame:
    """
    Extract evaluation metrics for selected variables from experiment folders.

    This function iterates over experiment subdirectories under ``folder_path``,
    reads ``metrics_mean.tsv`` files from both ``all`` and ``reg`` evaluation
    splits, and collects metric values into a tidy (long-form) DataFrame.

    Folder structure is assumed to follow:
        <folder_path>/
            <experiment_name>/
                all/overview/metrics_mean.tsv
                reg/overview/metrics_mean.tsv

    Experiments whose directory names start with ``"BL"`` are skipped.

    For each valid experiment, the function:
      - Infers the experiment group and base name (e.g. ``W1`` and ``W1-1a``)
      - Extracts values for variables listed in ``VARS``
      - Extracts metrics listed in ``METRICS`` (if present)
      - Records results separately for ``all`` and ``reg`` labels

    The returned DataFrame is sorted by:
      1. ``group`` (e.g. ``W1`` before ``W2``)
      2. experiment order using ``experiment_sort_key``
      3. ``label`` (``all`` before ``reg``)

    Parameters
    ----------
    folder_path : str
        Path to the root directory containing experiment subfolders.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
            - ``group``      : experiment group prefix (e.g. ``W1``, ``W2``)
            - ``experiment`` : experiment base name (e.g. ``W1-1a``)
            - ``label``      : evaluation split (``all`` or ``reg``)
            - ``metric``     : metric name (e.g. ``RMSE``, ``MAE``, ``CRPS``)
            - ``variable``   : variable name (e.g. ``prcp``, ``t2m``)
            - ``value``      : metric value (float)

        If no valid metrics are found, an empty DataFrame is returned.
    """
    rows: list[dict] = []
    folder = Path(folder_path)

    for exp_dir in folder.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("BL"):
            continue

        group, exp_base = parse_experiment_name(exp_dir.name)

        for label in ("all", "reg"):
            metrics_file = exp_dir / label / "overview" / "metrics_mean.tsv"
            if not metrics_file.exists():
                continue

            df = pd.read_csv(metrics_file, sep="\t", index_col=0)

            common_metrics = df.index.intersection(METRICS)
            common_vars = df.columns.intersection(VARS)

            for metric in common_metrics:
                for var in common_vars:
                    rows.append(
                        {
                            "group": group,         # W1 / W2
                            "experiment": exp_base, # W1-1a / W1-1 / ...
                            "label": label,         # all / reg
                            "metric": metric,       # RMSE / MAE / ...
                            "variable": var,        # prcp / t2m
                            "value": float(df.loc[metric, var]),
                        }
                    )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["exp_sort"] = out["experiment"].apply(experiment_sort_key)

    return out.sort_values(["group", "exp_sort", "label"]).drop(columns="exp_sort")


# ----------------------------
# plotting
# ----------------------------
def extract_nyear_metrics(folder_path: str,
                          labels: Sequence[str] = ("all", "reg")) -> pd.DataFrame:
    """
    Extract per-year-bin metrics from overview/nyear_metrics/<metric>.tsv for each experiment.

    Expected file location (per experiment and label):
        <experiment>/<label>/overview/nyear_metrics/<metric>.tsv

    Returns a long-form DataFrame with columns:
        group, experiment, label, metric, year_bin, variable, value
    """
    rows: list[dict] = []
    folder = Path(folder_path)

    for exp_dir in folder.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("BL"):
            continue

        group, exp_base = parse_experiment_name(exp_dir.name)

        for label in labels:
            nyear_dir = exp_dir / label / "overview" / "nyear_metrics"
            if not nyear_dir.exists():
                continue

            for metric in METRICS:
                metric_file = nyear_dir / f"{metric.lower()}.tsv"
                if not metric_file.exists():
                    continue

                df = pd.read_csv(metric_file, sep="\t")
                if "year_bin" not in df.columns:
                    continue

                common_vars = df.columns.intersection(VARS).tolist()
                if not common_vars:
                    continue

                long = df.melt(
                    id_vars="year_bin",
                    value_vars=common_vars,
                    var_name="variable",
                    value_name="value",
                )
                long["value"] = long["value"].astype(float)

                long["group"] = group
                long["experiment"] = exp_base
                long["label"] = label
                long["metric"] = metric

                rows.extend(
                    long[["group", "experiment", "label", "metric",
                          "year_bin", "variable", "value"]]
                    .to_dict("records")
                )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # sort experiments
    out["exp_sort"] = out["experiment"].map(experiment_sort_key)

    # keep year_bin in first-seen order
    year_order = pd.unique(out["year_bin"])
    out["year_bin"] = pd.Categorical(out["year_bin"], categories=year_order, ordered=True)

    return (
        out.sort_values(["group", "exp_sort", "experiment", "label",
                         "metric", "variable", "year_bin"])
           .drop(columns="exp_sort")
    )


def plot_nyear_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
    variable: str,
    label_mode: LabelMode = "all",
) -> None:
    """
    One subplot for a single metric:
      x-axis: year_bin
      y-axis: metric value
      lines: one per experiment (color); optionally separate all/reg by linestyle
    """
    sub = df.query("metric == @metric and variable == @variable")
    if sub.empty:
        ax.set_visible(False)
        return

    if label_mode != "both":
        sub = sub.query("label == @label_mode")
        if sub.empty:
            ax.set_visible(False)
            return

    # stable ordering for experiments
    exps = (
        sub[["group", "experiment"]]
        .drop_duplicates()
        .assign(exp_sort=lambda d: d["experiment"].map(experiment_sort_key))
        .sort_values(["group", "exp_sort"])
    )["experiment"].tolist()

    cmap = plt.get_cmap("tab10")
    exp2color = {e: cmap(i % 20) for i, e in enumerate(exps)}
    ls_map = {"all": "-", "reg": "--"}

    group_cols = ["experiment", "label"] if label_mode == "both" else ["experiment"]

    for keys, g in sub.groupby(group_cols, sort=False):
        if label_mode == "both":
            exp, lab = keys
            ls = ls_map.get(lab, "-")
            line_label = f"{exp} ({lab})"
        else:
            exp = keys if isinstance(keys, str) else keys[0]
            ls = "-"
            line_label = exp

        g = g.sort_values("year_bin")
        ax.plot(
            g["year_bin"].astype(str),
            g["value"].to_numpy(dtype=float),
            linestyle=ls,
            label=line_label,
            color=exp2color.get(exp),
            linewidth=1.8,
            marker="o",
            markersize=3.5,
        )

    ax.set_title(metric)
    ax.set_xlabel("year_bin")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Experiment", loc="upper right")

    ylim = Y_LIMITS.get(variable, {}).get(metric)
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])


def plot_nyear_metrics(folder_path: str) -> None:
    """
    For EACH variable (prcp, t2m), create ONE figure with 2x2 subplots:
      - each subplot is one metric (from METRICS)
      - x-axis is year_bin, y-axis is metric value
      - lines compare experiments (and all/reg splits via linestyle)

    Saves:
      {folder_path}/{var}_nyear_cmp.png
    """
    df = extract_nyear_metrics(folder_path)
    if df.empty:
        print("No nyear metrics found.")
        return

    metric_grid = [METRICS[i:i + 2] for i in range(0, len(METRICS), 2)]

    for var in VARS:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False, sharey=False)

        for r in range(2):
            for c in range(2):
                metric = metric_grid[r][c]
                plot_nyear_metric(axes[r, c], df, metric=metric, variable=var)

        # Put one shared legend outside (avoid 4 repeated legends)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="Experiment (label)",
                       loc="center left", bbox_to_anchor=(1.01, 0.5))

        fig.suptitle(f"Decadal Metric Comparison ({var})", y=1.02)
        plt.tight_layout()

        out = Path(folder_path) / f"{var}_nyear_cmp.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_metric_all_groups(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
    variable: str,
    group_gap: float = 1.0,
) -> None:
    """
    One subplot (one metric) that includes ALL groups on the same axis.
    X-axis is experiments ordered within each group, with a gap between groups.
    Bars are "all" vs "reg".
    """
    sub = df.query("metric == @metric and variable == @variable")
    if sub.empty:
        ax.set_visible(False)
        return

    # wide table with stable ordering
    wide = (
        sub.pivot_table(
            index=["group", "experiment"], columns="label", values="value", aggfunc="mean"
        ).reset_index()
    )
    wide["exp_sort"] = wide["experiment"].apply(experiment_sort_key)
    wide = wide.sort_values(["group", "exp_sort"]).drop(columns="exp_sort")

    labels = [c for c in ("all", "reg") if c in wide.columns]
    if not labels:
        ax.set_visible(False)
        return

    # x positions with group gaps
    group_sizes = wide.groupby("group", sort=False).size().to_numpy()
    starts = np.r_[0.0, np.cumsum(group_sizes[:-1] + group_gap)]
    x_positions = np.concatenate([s + np.arange(n) for s, n in zip(starts, group_sizes)])
    xticklabels = wide["experiment"].tolist()

    bar_w = 0.8 / len(labels)
    colors = {"all": "tab:blue", "reg": "tab:olive"}

    for i, lab in enumerate(labels):
        ax.bar(
            x_positions + i * bar_w,
            wide[lab].to_numpy(dtype=float),
            width=bar_w,
            label=lab,
            color=colors.get(lab),
            edgecolor="black",
        )

    ax.set_xticks(x_positions + bar_w * ((len(labels) / 2) - 0.5))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    # group separators + titles
    centers = starts + (group_sizes - 1) / 2
    for b in (starts[1:] - group_gap / 2):
        ax.axvline(b, linestyle="--", alpha=0.35)
    for grp, cx in zip(wide["group"].drop_duplicates().tolist(), centers):
        ax.text(cx, 1.02, grp, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Label", loc="upper right")

    # y-limits (if configured)
    ylim = Y_LIMITS.get(variable, {}).get(metric)
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])


def plot_metrics(folder_path: str) -> None:
    """
    For EACH variable (prcp, t2m), create ONE figure with 2x2 subplots:
      - each subplot is one metric (RMSE, CORR, MAE, CRPS)
      - each subplot includes ALL groups automatically (W1/W2/...)
    Saves:
      {folder_path}/prcp_mean_cmp.png
      {folder_path}/t2m_mean_cmp.png
    """
    df = extract_metrics(folder_path)
    if df.empty:
        print("No metrics found.")
        return

    metric_grid = [METRICS[i:i + 2] for i in range(0, len(METRICS), 2)]

    for var in VARS:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False, sharey=False)

        for r in range(2):
            for c in range(2):
                metric = metric_grid[r][c]
                plot_metric_all_groups(
                    axes[r, c],
                    df,
                    metric=metric,
                    variable=var,
                )

        fig.suptitle(f"Metric Mean Comparison ({var})", y=1.02)
        plt.tight_layout()

        out = Path(folder_path) / f"{var}_mean_cmp.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# Example usage
plot_metrics("../docs/experiments")
plot_nyear_metrics("../docs/experiments")
