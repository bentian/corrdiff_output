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
from typing import Iterable, Sequence, Optional, Union
import re
import pandas as pd

from plot_helper import plot_metrics_cmp, plot_nyear_metrics_cmp

EXP_FOLDER_PATH = Path("../docs/experiments")
CMP_FOLDER_PATH = Path("../docs/comparisons")

EXP_GROUP = "CropW"
GROUPS = (EXP_GROUP,)  # e.g., (EXP_GROUP, "BCSD")
FILENAME_SUFFIX = ""  # e.g., "w1_" to differentiate

VARS = ["prcp", "t2m"] if "BCSD" in GROUPS else ["prcp", "t2m", "u10m", "v10m"]
LABELS = (
    ("all",) if len(GROUPS) >= 1 else ("all", "reg")  # Only plot both for single group
)
METRICS = ["RMSE", "CORR", "MAE", "CRPS"]

BASE_COLS = ["group", "experiment", "label", "metric", "variable", "value"]
NYEAR_COLS = ["group", "experiment", "label", "metric", "year_bin", "variable", "value"]


def _experiment_info(exp_dir: Path) -> tuple[str, str]:
    """
    Extract group and experiment name from a directory name.

    Examples
    --------
    W1a-1        -> ("W1a", "W1a-1")
    W1-1a_2M     -> ("W1", "W1-1a")
    W2-3_4M      -> ("W2", "W2-3")

    Parameters
    ----------
    exp_dir : Path
        Experiment directory.

    Returns
    -------
    tuple[str, str]
        (group, experiment)
    """
    experiment = exp_dir.name.split("_", 1)[0]
    group = experiment.split("-", 1)[0]
    return group, experiment


def _experiments(root: Path) -> Iterable[Path]:
    """Yield experiment directories matching GROUPS."""
    return (p for p in root.iterdir() if p.is_dir() and p.name.startswith(GROUPS))


def _sort_key(exp: str) -> tuple[bool, int, bool, int, bool, str]:
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
    match = re.match(r"(Crop)?W(\d+)(a?)-(\d+)([a-z]?)", exp)
    if not match:
        return (True, 9999, True, 9999, True, exp)

    crop, group_num, group_variant, exp_num, letter = match.groups()
    return (
        crop is not None,  # W before CropW
        int(group_num),
        group_variant != "a",  # W1a before W1
        int(exp_num),
        letter != "a",  # 1a before 1
        letter,
    )


def _finish(df: pd.DataFrame, sort_cols: Sequence[str]) -> pd.DataFrame:
    """Apply label filtering and stable experiment sorting."""
    if df.empty:
        return df

    if len(LABELS) == 1:
        df = df[df["label"] == LABELS[0]]

    return (
        df.assign(
            exp_sort=df["experiment"].map(_sort_key),
            label_sort=df["label"].map({"all": 0, "reg": 1}),
        )
        .sort_values(["exp_sort", "label_sort", *sort_cols])
        .drop(columns=["exp_sort", "label_sort"])
        .reset_index(drop=True)
    )


def _melt_wide_metrics(
    df: pd.DataFrame,
    *,
    id_vars: Union[str, Sequence[str]],
    metric: Optional[str] = None,
    group: str,
    experiment: str,
    label: str,
) -> pd.DataFrame:
    """Convert a wide metrics table to the shared long-form schema."""
    vars_found = df.columns.intersection(VARS)
    if vars_found.empty:
        return pd.DataFrame()

    out = df.melt(
        id_vars=id_vars, value_vars=vars_found, var_name="variable", value_name="value"
    )
    if metric is not None:
        out["metric"] = metric

    return out.assign(
        group=group,
        experiment=experiment,
        label=label,
        value=lambda d: pd.to_numeric(d["value"], errors="coerce"),
    )


def extract_metrics(folder_path: Union[str, Path]) -> pd.DataFrame:
    """Extract metrics_mean.tsv files into long form."""
    frames: list[pd.DataFrame] = []

    for exp_dir in _experiments(Path(folder_path)):
        group, experiment = _experiment_info(exp_dir)

        for label in LABELS:
            path = exp_dir / label / "overview" / "metrics_mean.tsv"
            if not path.exists():
                continue

            df = pd.read_csv(path, sep="\t", index_col=0)
            df = df.loc[df.index.intersection(METRICS)]
            if df.empty:
                continue

            frames.append(
                _melt_wide_metrics(
                    df.rename_axis("metric").reset_index(),
                    id_vars="metric",
                    group=group,
                    experiment=experiment,
                    label=label,
                )[BASE_COLS]
            )

    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=BASE_COLS)
    )
    return _finish(out, ["metric", "variable"])


def extract_nyear_metrics(folder_path: Union[str, Path]) -> pd.DataFrame:
    """Extract overview/nyear_metrics/*.tsv files into long form."""
    frames: list[pd.DataFrame] = []

    for exp_dir in _experiments(Path(folder_path)):
        group, experiment = _experiment_info(exp_dir)

        for label in LABELS:
            metric_dir = exp_dir / label / "overview" / "nyear_metrics"
            if not metric_dir.exists():
                continue

            for metric in METRICS:
                path = metric_dir / f"{metric.lower()}.tsv"
                if not path.exists():
                    continue

                df = pd.read_csv(path, sep="\t")
                if "year_bin" not in df:
                    continue

                frames.append(
                    _melt_wide_metrics(
                        df,
                        id_vars="year_bin",
                        metric=metric,
                        group=group,
                        experiment=experiment,
                        label=label,
                    )[NYEAR_COLS]
                )

    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=NYEAR_COLS)
    )
    if out.empty:
        return out

    out["year_bin"] = pd.Categorical(
        out["year_bin"], categories=pd.unique(out["year_bin"]), ordered=True
    )
    return _finish(out, ["metric", "variable", "year_bin"])


def main() -> None:
    """
    Run full extraction and plotting pipeline.

    - Extract mean metrics
    - Extract decadal metrics
    - Generate and save comparison plots
    """
    out_dir = CMP_FOLDER_PATH / EXP_GROUP

    plot_metrics_cmp(
        extract_metrics(EXP_FOLDER_PATH),
        METRICS,
        VARS,
        out_dir,
        FILENAME_SUFFIX,
    )

    if EXP_GROUP not in ["DM", "LR"]:
        plot_nyear_metrics_cmp(
            extract_nyear_metrics(EXP_FOLDER_PATH),
            METRICS,
            VARS,
            out_dir,
            FILENAME_SUFFIX,
        )


if __name__ == "__main__":
    main()
