"""
Extraction utilities for experiment evaluation metrics.

This module scans experiment directories, extracts evaluation metrics for
precipitation (prcp) and 2 m temperature (t2m), and prepares tidy
(long-form) DataFrames for downstream plotting.

Two types of metrics are supported:
  1) Mean metrics from ``metrics_mean.tsv`` (all / reg splits)
  2) Decadal (year-bin) metrics from ``overview/nyear_metrics/<metric>.tsv``

Key features:
  - Automatic discovery of experiment folders
  - Grouping by experiment prefix (e.g. W1, W2)
  - Stable experiment ordering via ``experiment_sort_key``
  - Support for multiple variables and metrics
  - Consistent long-form output for plotting functions

Typical usage:
  - Call ``extract_metrics`` to obtain mean metrics
  - Call ``extract_nyear_metrics`` to obtain decadal metrics
  - Pass the resulting DataFrames to plotting helpers
    (e.g. ``plot_metrics_cmp`` and ``plot_nyear_metrics_cmp``)

This module does not perform plotting itself; it focuses on data extraction
and ordering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import pandas as pd

from plot_helper import plot_metrics_cmp, plot_nyear_metrics_cmp, experiment_sort_key

# --- Config ---
VARS = ["prcp", "t2m"]
METRICS = ["RMSE", "CORR", "MAE", "CRPS"]  # STD_DEV skipped

# ----------------------------
# helpers
# ----------------------------
def _parse_experiment_name(name: str) -> tuple[str, str]:
    """
    Examples
    --------
    W1-1a        -> group=W1, exp=W1-1a
    W1-1a_2M     -> group=W1, exp=W1-1a
    W2-3_4M      -> group=W2, exp=W2-3
    """
    base = name.split("_", 1)[0]
    return base.split("-", 1)[0], base


def _iter_experiments(folder_path: str):
    """
    Iterate over valid experiment directories under a root folder.

    Parameters
    ----------
    folder_path : str
        Path to the root directory containing experiment subfolders.

    Yields
    ------
    pathlib.Path
        Path object for each valid experiment directory.
    """
    for exp_dir in Path(folder_path).iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith("BL"):
            yield exp_dir


def _finalize_sort(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    """
    Apply a consistent experiment ordering to a metrics DataFrame.

    Adds a temporary experiment sort key using ``experiment_sort_key``,
    sorts by ``group`` and experiment order followed by ``sort_cols``,
    then removes the temporary column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``group`` and ``experiment`` columns.
    sort_cols : list[str]
        Additional columns to include in the sort order after experiment sorting.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame. If the input is empty, it is returned unchanged.
    """
    if df.empty:
        return df
    df = df.copy()
    df["exp_sort"] = df["experiment"].map(experiment_sort_key)
    df = df.sort_values(["group", "exp_sort", *sort_cols]).drop(columns="exp_sort")
    return df


# ----------------------------
# extraction
# ----------------------------
def extract_metrics(folder_path: str) -> pd.DataFrame:
    """
    Extract mean evaluation metrics from experiment folders.

    Reads ``metrics_mean.tsv`` files from both ``all`` and ``reg`` splits
    under each experiment directory and returns the results in long form.

    Expected file structure:
        <experiment>/<label>/overview/metrics_mean.tsv

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

        The DataFrame is sorted by group, experiment order, and label.
        If no metrics are found, an empty DataFrame is returned.
    """
    rows: list[dict] = []

    for exp_dir in _iter_experiments(folder_path):
        group, exp_base = _parse_experiment_name(exp_dir.name)

        for label in ("all", "reg"):
            f = exp_dir / label / "overview" / "metrics_mean.tsv"
            if not f.exists():
                continue

            df = pd.read_csv(f, sep="\t", index_col=0)
            for metric in df.index.intersection(METRICS):
                for var in df.columns.intersection(VARS):
                    rows.append(
                        dict(
                            group=group,
                            experiment=exp_base,
                            label=label,
                            metric=metric,
                            variable=var,
                            value=float(df.loc[metric, var]),
                        )
                    )

    return _finalize_sort(pd.DataFrame(rows), ["label"])


def extract_nyear_metrics(folder_path: str,
                          labels: Sequence[str] = ("all", "reg")) -> pd.DataFrame:
    """
    Extract per-year-bin (decadal) metrics from experiment folders.

    Reads ``<metric>.tsv`` files from ``overview/nyear_metrics`` under the
    specified evaluation labels and returns results in long form.

    Expected file structure:
        <experiment>/<label>/overview/nyear_metrics/<metric>.tsv

    Parameters
    ----------
    folder_path : str
        Path to the root directory containing experiment subfolders.
    labels : Sequence[str], optional
        Evaluation splits to include (default: ``("all", "reg")``).

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:\
            - ``group``      : experiment group prefix (e.g. ``W1``, ``W2``)
            - ``experiment`` : experiment base name (e.g. ``W1-1a``)
            - ``label``      : evaluation split (``all`` or ``reg``)
            - ``metric``     : metric name (e.g. ``RMSE``, ``MAE``, ``CRPS``)
            - ``year_bin``   : year bin label (e.g. ``2015-2020``, ``2021-2026``)
            - ``variable``   : variable name (e.g. ``prcp``, ``t2m``)
            - ``value``      : metric value (float)

        The DataFrame is sorted by group, experiment order, and year_bin.
        If no metrics are found, an empty DataFrame is returned.
    """
    rows: list[dict] = []

    for exp_dir in _iter_experiments(folder_path):
        group, exp_base = _parse_experiment_name(exp_dir.name)

        for label in labels:
            nyear_dir = exp_dir / label / "overview" / "nyear_metrics"
            if not nyear_dir.exists():
                continue

            for metric in METRICS:
                f = nyear_dir / f"{metric.lower()}.tsv"
                if not f.exists():
                    continue

                df = pd.read_csv(f, sep="\t")
                if "year_bin" not in df.columns:
                    continue

                common_vars = df.columns.intersection(VARS)
                if common_vars.empty:
                    continue

                long = df.melt("year_bin", common_vars, "variable", "value").assign(
                    group=group, experiment=exp_base, label=label, metric=metric
                )
                long["value"] = long["value"].astype(float)

                rows.extend(
                    long[["group", "experiment", "label", "metric",
                          "year_bin", "variable", "value"]]
                    .to_dict("records")
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # keep year_bin in first-seen order
    year_order = pd.unique(out["year_bin"])
    out["year_bin"] = pd.Categorical(out["year_bin"], categories=year_order, ordered=True)

    return _finalize_sort(out, ["experiment", "label", "metric", "variable", "year_bin"])


# Example usage
root = "../docs/experiments"
plot_metrics_cmp(extract_metrics(root), METRICS, VARS, root)
plot_nyear_metrics_cmp(extract_nyear_metrics(root), METRICS, VARS, root)
