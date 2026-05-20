"""
Training loss visualization utilities.

This module provides helper functions for plotting training loss curves
extracted from experiment logs (e.g., TensorBoard). It focuses on simple,
clear visualizations with optional smoothing to highlight long-term
training trends.

Typical usage:
    >>> plot_training_loss(wall_times, loss_values, Path("training_loss.png"))
"""

from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt


def plot_training_loss(
    x_values: Union[List[datetime], List[int]], loss_values: List[float], output_file: Path
) -> None:
    """
    Create a training loss plot with time on the x-axis and save it to a PNG file.

    Parameters:
        x_values (Union[List[datetime], List[int]]): Wall times / Training duration (x-axis values).
        loss_values (List[float]): Loss values (y-axis values).
        output_file (Path): File path to save the output plot.
    """
    window_size = min(20, len(loss_values))
    smoothed_loss_values = np.convolve(
        loss_values, np.ones(window_size) / window_size, mode="valid"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, loss_values, alpha=0.5, label="Raw Loss", color="gray", linewidth=1)
    plt.plot(
        x_values[: len(smoothed_loss_values)],
        smoothed_loss_values,
        label="Smoothed Loss",
        linestyle="--",
        linewidth=2,
    )

    # Set x-axis label for Wall times / Training duration
    if x_values and isinstance(x_values[0], datetime):
        plt.xlabel("Time")
        plt.gcf().autofmt_xdate()  # Format x-axis for better readability
    else:
        plt.xlabel("Training Duration")

    plt.yscale("log")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
