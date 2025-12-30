"""
Plotting utilities for CorDiff model evaluation.

This package provides modular plotting helpers used throughout the
evaluation and analysis pipeline. Functions are organized by purpose
(e.g., metrics, distributions, samples, training) but re-exported here
to present a clean public API.

Typical usage:
    >>> import plot_helpers as ph
    >>> ph.plot_metrics(metrics_ds, output_path)

Submodules
----------
- metrics       : Metric curves, tables, and ensemble comparisons
- distributions : PDFs and value distributions
- samples       : Spatial sample plots and error maps
- training      : Training loss visualization
"""
from .metrics import plot_metrics, plot_monthly_metrics, \
                     plot_nyear_metrics, plot_metrics_vs_ensembles
from .distributions import plot_metrics_cnt, plot_pdf
from .samples import plot_top_samples, plot_p90_by_nyear, plot_monthly_error
from .training import plot_training_loss

__all__ = [
    "plot_metrics",         "plot_monthly_metrics",
    "plot_nyear_metrics",   "plot_metrics_vs_ensembles",
    "plot_metrics_cnt",     "plot_pdf",
    "plot_top_samples",     "plot_p90_by_nyear",
    "plot_monthly_error",   "plot_training_loss",
]
