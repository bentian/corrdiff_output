"""
Training loss visualization utilities.

This module provides helper functions for plotting training loss curves
extracted from experiment logs (e.g., TensorBoard). It focuses on simple,
clear visualizations with optional smoothing to highlight long-term
training trends.

Typical usage:
    >>> plot_training_loss(wall_times, loss_values, Path("training_loss.png"))
"""
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def plot_training_loss(wall_times: List[float], values: List[float], output_file: Path) -> None:
    """
    Create a training loss plot with time on the x-axis and save it to a PNG file.

    Parameters:
        wall_times (List[float]): Wall times (x-axis values).
        values (List[float]): Loss values (y-axis values).
        output_file (Path): File path to save the output plot.
    """
    window_size = min(20, len(values))
    smoothed_values = np.convolve(values, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(wall_times, values, alpha=0.5, label="Raw Loss", color="gray", linewidth=1)
    plt.plot(wall_times[:len(smoothed_values)], smoothed_values,
             label="Smoothed Loss", linestyle="--", linewidth=2)

    plt.xlabel("Time")
    plt.yscale("log")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()  # Format x-axis for better readability
    plt.savefig(output_file)
    plt.close()
