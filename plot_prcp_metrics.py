"""
Module for extracting and plotting RMSE and MAE values for precipitation (prcp)
across multiple experiments.

Experiments are grouped by suffix (e.g., "2M", "4M") and further categorized
by prefix ("BL" or "D1") and dataset type ("all" or "reg"). The script generates
a grouped bar chart comparing these experiments.

Output:
- A bar chart (saved as "data/prcp_cmp.png") showing RMSE and MAE comparisons.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_prcp_metrics(folder_path: str) -> pd.DataFrame:
    """
    Extract RMSE and MAE values for 'prcp' from 'metrics_mean.tsv' files in multiple experiments.

    Parameters:
        folder_path (str): Path to the folder containing experiment subdirectories.

    Returns:
        pd.DataFrame: A DataFrame containing extracted metrics sorted by suffix, label, and prefix.
    """
    metrics_list = []

    folder = Path(folder_path)
    for exp_dir in folder.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.endswith("_ds4"):
            continue

        for label in ["all", "reg"]:  # Include both dataset types
            metrics_file = exp_dir / label / "overview" / "metrics_mean.tsv"

            if metrics_file.exists():
                df = pd.read_csv(metrics_file, sep="\t", index_col=0)
                if "prcp" in df.columns:
                    # Extract prefix ("BL" or "D1") and suffix (e.g., "2M", "4M_1322")
                    parts = exp_dir.name.split("_")
                    exp_prefix, exp_suffix = \
                        parts[0], "_".join(parts[1:]) if len(parts) > 1 else "base"

                    metrics_list.append({
                        "experiment": exp_dir.name,
                        "label": label,  # "all" or "reg"
                        "prefix": exp_prefix,  # "BL" or "D1"
                        "suffix": exp_suffix,  # e.g., "2M", "4M_1322"
                        "RMSE": df.loc["RMSE", "prcp"],
                        "MAE": df.loc["MAE", "prcp"],
                    })

    return pd.DataFrame(metrics_list).sort_values(by=["suffix", "label", "prefix"])


def plot_grouped_bars(ax: plt.axes, pivot_df: pd.DataFrame, metric_name: str, ymin: float) -> None:
    """
    Plot grouped bar charts for RMSE and MAE comparisons.

    Parameters:
        ax (plt.Axes): The subplot axis to draw on.
        pivot_df (pd.DataFrame): Pivoted DataFrame containing the metric values.
        metric_name (str): Name of the metric being plotted (RMSE or MAE).
        ymin (float): Minimum y-axis value for better scaling.
    """
    n_groups = len(pivot_df)
    index = np.arange(n_groups)  # X-axis positions
    bar_width = 0.2  # Bar width to align BL and D1 pairs

    # Define colors for BL and D1 in "all" and "reg" datasets
    colors = {"BL-all": "tab:blue", "D1-all": "tab:green",
              "BL-reg": "tab:orange", "D1-reg": "tab:red"}

    for i, (label, prefix) in enumerate(pivot_df.columns):
        tag = f"{prefix}-{label}"
        ax.bar(index + i * bar_width, pivot_df[(label, prefix)], width=bar_width,
               label=tag, color=colors[tag], edgecolor="black")

        # Add value labels
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
        #                 xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
        #                 fontsize=10, color="black")

    ax.set_xticks(index + bar_width * ((len(pivot_df.columns) / 2) - 0.5))
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    ax.set_xlabel("Experiment Suffix")
    ax.set_ylabel(metric_name)
    ax.set_ylim(ymin=ymin)  # Ensure minimum y-axis threshold
    ax.set_title(f"{metric_name} Comparison for PRCP")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="Group", loc="upper left", bbox_to_anchor=(1, 1))


def plot_prcp_metrics(folder_path: str)  -> None:
    """
    Extracts RMSE and MAE values, groups them by suffix, and generates a grouped bar chart.

    Parameters:
        folder_path (str): Path to the experiment data directory.

    Output:
        - Saves a bar chart as "data/prcp_cmp.png" showing RMSE and MAE comparisons.
    """
    metrics_df = extract_prcp_metrics(folder_path)

    # Pivot data for plotting
    rmse_pivot = metrics_df.pivot(index="suffix", columns=["label", "prefix"], values="RMSE")
    mae_pivot = metrics_df.pivot(index="suffix", columns=["label", "prefix"], values="MAE")

    # Create figure
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_grouped_bars(axes[0], rmse_pivot, "RMSE", ymin=7.5)
    plot_grouped_bars(axes[1], mae_pivot, "MAE", ymin=3.5)

    plt.tight_layout()
    plt.savefig("data/prcp_cmp.png")
    plt.close()


# Example Usage:
plot_prcp_metrics("docs/experiments")
