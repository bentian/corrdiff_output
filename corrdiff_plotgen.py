import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from score_samples_n_plot import score_samples_n_plot

def plot_metrics(data, output_path):
    metrics = data["metric"].values
    variables = list(data.data_vars.keys())
    data_array = np.array([data[var] for var in variables])  # Shape: (4, 4)

    # Bar chart
    x = np.arange(len(metrics))  # Metric indices
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, var in enumerate(variables):
        bars = ax.bar(x + i * width, data_array[i], width, label=var)
        # Add value annotations on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center", va="bottom", fontsize=9
            )

    ax.set_title("Metrics for TReAD channels", fontsize=14)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.set_xticks(x + width * (len(variables) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(title="Variables")
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)

def plot_monthly_metrics(ds, metrics, output_path):
    fig, ax = plt.subplots(len(metrics), 1, figsize=(10, 6 * len(metrics)))

    if len(metrics) == 1:
        ax = [ax]

    for i, metric in enumerate(metrics):
        df_grouped = (
            ds.groupby("time.month").mean(dim="time")
            .sel(metric=metric)
            .to_dataframe()
            .round(2)
        )

        for variable in df_grouped.columns[:-1]:
            ax[i].plot(df_grouped.index, df_grouped[variable], marker="o", label=variable)
            for x, y in zip(df_grouped.index, df_grouped[variable]):
                ax[i].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

        ax[i].set_title(f"Monthly Mean for {metric.upper()}", fontsize=14)
        ax[i].set_xlabel("Month", fontsize=12)
        ax[i].set_ylabel(f"{metric.upper()} Value", fontsize=12)
        ax[i].set_xticks(np.arange(1, 13))
        ax[i].legend(title="Variables")
        ax[i].grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)

def aggregate_metrics(input_file, output_path_prefix):
    ds = xr.open_dataset(input_file, engine='netcdf4').rename({
        "precipitation": "prcp",
        "temperature_2m": "t2m",
        "eastward_wind_10m": "u10m",
        "northward_wind_10m": "v10m",
    })

    # Compute mean of all metrics and save results
    metric_mean = ds.mean(dim="time")
    metric_mean.to_dataframe().round(2).to_csv(
        f"{output_path_prefix}_metrics.csv", float_format="%.2f"
    )
    plot_metrics(metric_mean, f"{output_path_prefix}_metrics.png")

    # Plot monthly metrics and save results
    plot_monthly_metrics(ds, ["mae", "rmse"], f"{output_path_prefix}_monthly_metrics.png")
    grouped = ds.groupby("time.month").mean(dim="time")

    grouped.sel(metric="mae").to_dataframe().to_csv(
        f"{output_path_prefix}_monthly_mae.csv", float_format="%.2f")
    grouped.sel(metric="rmse").to_dataframe().to_csv(
        f"{output_path_prefix}_monthly_rmse.csv", mode="a", float_format="%.2f")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Path for the output file.")
    parser.add_argument("plotpath", type=str, help="Folder to save the plots.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    args = parser.parse_args()

    plot_prefix = f"{args.plotpath}/{args.output[:-3]}"
    score_filename = score_samples_n_plot(args.output, plot_prefix, args.n_ensemble)

    aggregate_metrics(score_filename, plot_prefix)
