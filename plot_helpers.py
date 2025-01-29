from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(ds, output_path: Path, number_format: str):
    metrics = ds["metric"].values
    variables = list(ds.data_vars.keys())
    data_array = np.array([ds[var] for var in variables])

    x = np.arange(len(metrics))
    width = 0.2

    _, ax = plt.subplots(figsize=(10, 6))
    for i, var in enumerate(variables):
        bars = ax.bar(x + i * width, data_array[i], width, label=var)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:{number_format}}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title("Metrics Mean", fontsize=14)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.set_xticks(x + width * (len(variables) - 1) / 2)
    ax.set_xticklabels([metric.upper() for metric in metrics])
    ax.legend(title="Variables")
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)


def plot_monthly_metrics(ds, metric, output_path: Path, number_format: str):
    _, ax = plt.subplots(figsize=(10, 6))
    df_grouped = ds.to_dataframe()

    for variable in df_grouped.columns:
        ax.plot(df_grouped.index, df_grouped[variable], marker="o", label=variable)
        for x, y in zip(df_grouped.index, df_grouped[variable]):
            ax.annotate(f"{y:{number_format}}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    ax.set_title(f"Monthly Mean for {metric.upper()}", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(f"{metric.upper()} Value", fontsize=12)
    ax.set_xticks(np.arange(1, 13))
    ax.legend(title="Variables")
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)


def get_bin_count_n_note(ds, bin_width=1):
    min_val, max_val = ds.min().item(), ds.max().item()
    bin_count = int((max_val - min_val) / bin_width)
    return bin_count, f"({len(ds):,} pts in [{min_val:.1f}, {max_val:.1f}])"


def plot_pdf(truth, pred, output_path: Path):
    """
    Plot PDFs for all variables in the truth dataset, comparing with prediction.
    """
    for var in truth.data_vars:
        if var != 'prcp':
            continue  # Only plot 'prcp' for now

        if var in pred:
            truth_flat = truth[var].values.flatten()
            pred_flat = pred[var].mean("ensemble").values.flatten() \
                        if "ensemble" in pred.dims else pred[var].values.flatten()

            truth_bin_count, truth_note = get_bin_count_n_note(truth_flat)
            pred_bin_count, pred_note = get_bin_count_n_note(pred_flat)
            print(f"Variable: {var} | PDF bin count: {truth_bin_count} (truth) / {pred_bin_count} (pred)")
            title_suffix = ' (zoomed)' if var == 'prcp' else ''

            plt.figure(figsize=(10, 6))
            plt.hist(truth_flat, bins=truth_bin_count, alpha=0.5, label="Truth", density=True)
            plt.hist(pred_flat, bins=pred_bin_count, alpha=0.5, label="Prediction", density=True)
            plt.title(f"PDF of {var}{title_suffix}:\nTruth {truth_note} /\nPrediction {pred_note}")
            plt.xlabel(f"{var} (units)")
            plt.ylabel("Density")
            plt.legend()
            plt.grid()

            plt.xlim([-5, 15]) if var == 'prcp' else None  # Adjust x-limits for precipitation

            plt.savefig(output_path / f"pdf_{var}.png")
            plt.close()


def plot_monthly_error(ds, output_path: Path):
    """
    Compute monthly mean error and plot results.
    """
    monthly_mean = ds.groupby("time.month").mean(dim="time")
    variables = list(ds.data_vars.keys())
    colormaps = ["Blues", "Oranges", "Greens", "Reds"]

    for index, var in enumerate(variables):
        if var != 'prcp':
            continue  # Only plot 'prcp' for now

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for month in range(1, 13):
            data = monthly_mean[var].sel(month=month).mean(dim="ensemble")
            im = axes[month - 1].imshow(data, cmap=colormaps[index], origin="lower")
            axes[month - 1].set_title(f"Month {month}", fontsize=10)
            axes[month - 1].set_axis_off()
            fig.colorbar(im, ax=axes[month - 1], shrink=0.8)

        fig.suptitle(f"Monthly Mean Error of {var}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path / f"monthly_error_{var}.png")


def plot_training_loss(wall_times, values, output_file: Path):
    """
    Create a training loss plot with time on the x-axis and save it to a PNG file.
    """
    window_size = 20
    smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(wall_times, values, alpha=0.5, label="Raw Loss", color="gray", linewidth=1)
    plt.plot(wall_times[:len(smoothed_values)], smoothed_values, label="Smoothed Loss", linestyle="--", linewidth=2)

    plt.xlabel("Time")
    plt.yscale("log")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()  # Format x-axis for better readability
    plt.savefig(output_file)
