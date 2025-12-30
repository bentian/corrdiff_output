# 📌 Overview
This project provides a set of tools to generate CorrDiff experiment results by evaluating, processing, and visualizing weather model data. It includes:

- Masking datasets using landmasks
- Computing metrics for regression and diffusion models
- Generating visual plots for model evaluation
- Exporting processed data into tables

![image](graphic/infograph.png)

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
   - `--masked`: Apply landmask (yes or no. default: yes).

## 📖 Example Run

`python corrdiff_plotgen.py data/Baseline output/ --n-ensemble=1 --masked=no`

This will:
- ✅ Process NetCDF datasets
- ✅ Compute model evaluation metrics
- ✅ Generate PNG plots & TSV tables
- ✅ Apply landmask if specified

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
 ┣ 📜 corrdiff_plotgen.py     # Main script to process models
 ┣ 📜 mask_samples.py         # Applies landmask to datasets
 ┣ 📜 score_samples_v2.py     # Computes model evaluation metrics
 ┣ 📜 plot_helpers.py         # Contains helper functions for visualization
 ┣ 📜 plot_prcp_metrics.py    # Create PRCP metrics plot among experiments
 ┣ 📜 refresh.zsh             # Zsh script for refreshing plots and tables under docs/experiments/
 ┗ 📜 README.md               # Project documentation
```

# 📜 Script Descriptions

## 🔹 corrdiff_plotgen.py - Main Pipeline
  - Reads NetCDF weather model data.
  - Computes model evaluation metrics.
  - Generates plots and tables.
  - Saves processed Hydra configuration.

## 🔹 mask_samples.py - Applies Landmask
  - Loads truth and prediction datasets.
  - Masks areas outside of land.
  - Saves processed NetCDF files.

## 🔹 score_samples_v2.py - Compute Metrics
  - Computes RMSE, CRPS, standard deviation.
  - Processes truth vs. predicted values.
  - Saves error maps and flattened datasets.

## 🔹 plot_helpers.py - Generates Plots
  - Bar charts for metrics.
  - PDF distributions for variables.
  - Monthly error visualizations.

## 🔹 plot_prcp_metrics.py - Created PRCP Metrics Plot
  - Bar chart for metrics among experiments.
  - Groups experiments by name prefix and suffix.

## 🔹 refresh.zsh - Zsh Automation Script
  - Refreshes the plots and tables.
  - Runs plot generation scripts in a batch.
