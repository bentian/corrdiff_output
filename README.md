# 📌 Overview
This project provides a set of tools to generate CorrDiff experiment results by evaluating, processing, and visualizing weather model data. It includes:

- Masking datasets using landmasks
- Computing metrics for regression and diffusion models
- Generating visual plots for model evaluation
- Exporting processed data into tables

![image](graphic/workflow.png)

More information can be found in [intro deck](graphic/intro_deck.pdf).

# 📦 Installation

Before using corrdiff_output, install the required dependencies:

`conda env create -f corrdiff.yml`

If you haven't installed `xskillscore`, you may need:

`pip install xskillscore`

# 🚀 Usage

## Process and Generate Outputs
   Run the main script to generate plots and metrics:

   `python corrdiff_plotgen.py <input_directory> <output_directory>`

   - `<input_directory>`: Directory containing
     - NetCDF (`netcdf/`)
     - TensorBoard logs (`tensorboard_regression/` and `tensorboard_diffusion/`)
     - Hydra configs (`hydra_train/` and `hydra_generate/`)
   - `<output_directory>`: Directory where outputs will be saved.
   - `--n-ensemble`: Number of ensemble members (default: 1).
   - `--no-mask`: Disable landmask (default: apply landmask).

## 📖 Example Run

`python corrdiff_plotgen.py data/Baseline output/ --n-ensemble=1 --no-mask`

This will:
- ✅ Process NetCDF datasets
- ✅ Compute model evaluation metrics
- ✅ Generate PNG plots & TSV tables
- ✅ Disable landmask if specified

# 📊 Generated Outputs

The script produces:

- Plots (`.png`): Metrics, monthly errors, PDFs
- Tables (`.tsv`): Mean metrics, monthly grouped metrics
- Processed NetCDF files (`.nc`): Masked datasets

# 📂 Project Structure

```
📦 corrdiff_output
 ┣ 📂 data/                   # Example input data (NetCDF files)
 ┣ 📂 docs/                   # Web pages to display generated plots and tables
   ┣ 📂 experiments/          # Generated plots and tables per experiment
   ┣ 📂 css/                  # Style of web pages
   ┣ 📂 js/                   # JS files to display web pages dynamically per experiment
   ┣ 📜 render.html           # Web page to display experiment plots and tables
   ┗ 📜 index.html            # Portal web page to select experiment for display
 ┣ 📂 src/
   ┣ 📂 nc_helpers/           # Helpers to process BCSD data & split NetCDF files
   ┣ 📂 plot_helper/          # Helpers to generate plots
   ┣ 📂 samples_handler/      # Helpers to landmask / score / process samples
   ┣ 📜 analysis_utils.py     # Utilities for analysis
   ┣ 📜 corrdiff_plotgen.py   # Main script to process models
   ┗ 📜 cmp_plotgen.py        # Script to plot metrics comparison among experiments
 ┗ 📜 README.md               # Project documentation
```

# 📜 Script Descriptions

## 🔹 `corrdiff_plotgen.py` - Main Pipeline
  - Reads NetCDF weather model data.
  - Computes model evaluation metrics.
  - Generates plots and tables.

## 🔹 `analysis_utils.py` - Utilities for analysis
  - Metric table export + plotting wrappers.
  - Grouping time-indexed metric datasets into fixed N-year bins.
  - TensorBoard scalar extraction and training-loss plotting.
  - Hydra YAML flattening and YAML -> TSV conversion.

## 🔹 `samples_handler/*` - Compute Metrics
  - Computes `RMSE`, `MAE`, `CORR`, `CRPS`, and `STD_DEV`.
  - Processes truth vs. predicted values.
  - Saves error maps and flattened datasets.
  - Select overall top dates and create p90 grids per N-year bin.

## 🔹 `plot_helper/*` - Generates Plots
  - Bar charts for metrics.
  - PDF distributions for variables.
  - Monthly and decadal error visualizations.

## 🔹 `nc_helpers/*` - Process NetCDF files
  - Splits NetCDF files by year into folders `truth`, `prediction_all`, and `prediction_reg`.
  - Regrids BCSD data to 128x96 grid (TaiESM 3.5km) and merges with truth for evaluation.

## 🔹 `cmp_plotgen.py` - Compares Metrics among Experiments
  - Aggregates metrics mean and decadal trends across experiments.
  - Visualizes as line charts for comparison among experiments.
