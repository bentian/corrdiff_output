"""
Extraction utilities for experiment evaluation metrics.

This module scans experiment directories and extracts evaluation metrics
into long-form DataFrames suitable for plotting.

Supported data
--------------
1. Mean metrics (metrics_mean.tsv)
2. Decadal metrics (overview/nyear_metrics/*.tsv)

Features
--------
- Automatic experiment discovery
- Group parsing from experiment names
- Consistent experiment ordering via `experiment_sort_key`
- Output in tidy (long-form) structure

Typical workflow
----------------
1. Call:
    - `extract_metrics` for mean metrics
    - `extract_nyear_metrics` for decadal metrics
2. Pass results to plotting utilities (e.g. `plot_metrics_cmp`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import re
import pandas as pd

from plot_helper import plot_metrics_cmp, plot_nyear_metrics_cmp

EXP_FOLDER_PATH = Path("../docs/experiments")
CMP_FOLDER_PATH = Path("../docs/comparisons")

EXP_GROUP = "CropW"
GROUPS = (EXP_GROUP,)  # (EXP_GROUP, "BCSD")

VARS = ["prcp", "t2m"] if "BCSD" in GROUPS else ["prcp", "t2m", "u10m", "v10m"]
LABEL_MODE = "both" if len(GROUPS) == 1 else "all"
METRICS = ["RMSE", "CORR", "MAE", "CRPS"]


def _parse_experiment_name(name: str) -> tuple[str, str]:
    """
    Extract group and experiment base name from a directory name.

    Examples
    --------
    W1a-1        -> ("W1a", "W1a-1")
    W1-1a_2M     -> ("W1", "W1-1a")
    W2-3_4M      -> ("W2", "W2-3")

    Parameters
    ----------
    name : str
        Directory name.

    Returns
    -------
    tuple[str, str]
        (group, experiment_base)
    """
    base = name.split("_", 1)[0]
    return base.split("-", 1)[0], base


def _iter_experiments(folder_path: str):
    """Iterate over valid experiment directories."""
    for exp_dir in Path(folder_path).iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith(GROUPS):
            yield exp_dir


def _experiment_sort_key(exp: str) -> tuple[int, int, int]:
    """
    Sort order:
        W1a-*, then W1-*
        W2...
        then CropW1a-*, CropW1-*, CropW2...

    Within each:
        1a, 1, 2, ...

    Returns:
        (prefix_priority, group_number, group_variant_priority,
         exp_number, letter_priority, letter)
    """
    m = re.match(r"(Crop)?W(\d+)(a?)-(\d+)([a-z]?)", exp)
    if not m:
        return (9999, 9999, 9999, 9999, 1, exp)

    is_crop, group_num, group_variant, exp_num, letter = m.groups()

    return (
        is_crop is not None,  # W before CropW
        int(group_num),
        group_variant != "a",  # W1a before W1
        int(exp_num),
        letter != "a",  # 1a before 1
        letter,
    )


def _sort_df(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """
    Sort a DataFrame by experiment order and additional columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    *cols : str
        Additional columns for sorting.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame.
    """
    if df.empty:
        return df

    if LABEL_MODE != "both":
        df = df[df["label"] == LABEL_MODE]

    return (
        df.assign(
            exp_sort=df["experiment"].map(_experiment_sort_key),
            label_sort=df["label"].map({"all": 0, "reg": 1}),
        )
        .sort_values(["exp_sort", "label_sort", *cols])
        .drop(columns=["exp_sort", "label_sort"])
        .reset_index(drop=True)
    )


def _read_mean_metrics(
    path: Path, *, group: str, experiment: str, label: str
) -> pd.DataFrame:
    """
    Read and reshape mean metrics from a TSV file.

    Parameters
    ----------
    path : Path
        Path to metrics_mean.tsv.
    group : str
        Experiment group (e.g. W1).
    experiment : str
        Experiment name.
    label : str
        Evaluation label ("all" or "reg").

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame (may be empty if file missing).
    """
    if not path.exists():
        return pd.DataFrame()

    wide = pd.read_csv(path, sep="\t", index_col=0)
    wide = wide.loc[wide.index.intersection(METRICS), wide.columns.intersection(VARS)]
    if wide.empty:
        return pd.DataFrame()

    return (
        wide.rename_axis("metric")
        .reset_index()
        .melt(id_vars="metric", var_name="variable", value_name="value")
        .assign(group=group, experiment=experiment, label=label)[
            ["group", "experiment", "label", "metric", "variable", "value"]
        ]
    )


def extract_metrics(folder_path: str) -> pd.DataFrame:
    """
    Extract mean metrics from experiment directories.

    Parameters
    ----------
    folder_path : str
        Root directory containing experiment folders.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        ["group", "experiment", "label", "metric", "variable", "value"]
    """
    frames = []

    for exp_dir in _iter_experiments(folder_path):
        group, exp_base = _parse_experiment_name(exp_dir.name)
        for label in ("all", "reg"):
            frames.append(
                _read_mean_metrics(
                    exp_dir / label / "overview" / "metrics_mean.tsv",
                    group=group,
                    experiment=exp_base,
                    label=label,
                )
            )

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _sort_df(out, "metric", "variable")


def extract_nyear_metrics(
    folder_path: str, labels: Sequence[str] = ("all", "reg")
) -> pd.DataFrame:
    """
    Extract decadal (year-bin) metrics from experiment directories.

    Parameters
    ----------
    folder_path : str
        Root directory containing experiment folders.
    labels : Sequence[str], optional
        Labels to include.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        ["group", "experiment", "label", "metric", "year_bin", "variable", "value"]
    """
    frames = []

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

                frames.append(
                    df.melt("year_bin", common_vars, "variable", "value").assign(
                        group=group,
                        experiment=exp_base,
                        label=label,
                        metric=metric,
                        value=lambda d: d["value"].astype(float),
                    )[
                        [
                            "group",
                            "experiment",
                            "label",
                            "metric",
                            "year_bin",
                            "variable",
                            "value",
                        ]
                    ]
                )

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    year_order = pd.unique(out["year_bin"])
    out["year_bin"] = pd.Categorical(
        out["year_bin"], categories=year_order, ordered=True
    )
    return _sort_df(out, "metric", "variable", "year_bin")


def main():
    """
    Run full extraction and plotting pipeline.

    - Extract mean metrics
    - Extract decadal metrics
    - Generate and save comparison plots
    """
    plot_metrics_cmp(
        extract_metrics(EXP_FOLDER_PATH), METRICS, VARS, CMP_FOLDER_PATH / EXP_GROUP
    )
    plot_nyear_metrics_cmp(
        extract_nyear_metrics(EXP_FOLDER_PATH),
        METRICS,
        VARS,
        CMP_FOLDER_PATH / EXP_GROUP,
    )


if __name__ == "__main__":
    main()
