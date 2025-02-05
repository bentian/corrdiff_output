# Experiment Viewer
Experiment Viewer is a web-based tool for visualizing and comparing experiment results, including statistical metrics, probability density functions (PDFs), and error analyses.

# Features
- 📊 Compare two experiments with interactive tables and plots.
- 📈 Summarize a single experiment with key performance metrics.
- 📂 View various experiment results including MAE, RMSE, and PDFs for multiple variables.
- 🌍 User-friendly interface with collapsible sections for efficient navigation.
- 🔍 Lightbox feature for enlarged visualization of plots.

# Installation & Usage

## 1️⃣ Clone the Repository
```
git clone https://github.com/bentian/corrdiff_output.git
cd corrdiff_output/docs
```

## 2️⃣ Open the Web Interface
Simply open `index.html` in a browser.

Alternatively, start a simple HTTP server:

`python3 -m http.server 8000`

Then, access `http://localhost:8000` in your browser.

# Project Structure

```
📂 docs/
│── 📂 css/               # Stylesheets
│   ├── render.css        # Styling for experiment results page
│   ├── style.css         # Global styles
│
│── 📂 js/                # JavaScript logic
│   ├── main.js           # Handles experiment selection forms
│   ├── render.js         # Dynamically loads and displays experiment results
│
│── 📂 experiments/       # Contains experiment results (JSON, images, TSV, etc.)
│   ├── list.json         # List of available experiments
│
│── index.html            # Homepage for selecting experiments
│── render.html           # Displays selected experiment results
│── README.md             # Project documentation
```

# How It Works

## 🔹 Experiment Selection
- The main page (`index.html`) allows users to:
  - Select and summarize one experiment.
  - Compare two experiments.

## 🔹 Dynamic Rendering (`render.html`)
- Loads selected experiment data dynamically.
- Displays collapsible sections for metrics, PDFs, error plots.
- Supports lightbox feature for enlarged image viewing.

## 🔹 JavaScript Components
`main.js` - Handles experiment selection
  - Fetches available experiments from `experiments/list.json`.
  - Populates dropdowns dynamically.
  - Redirects to `render.html` with selected experiment parameters.

`render.js` - Displays experiment results
  - Reads URL parameters to load selected experiment data.
  - Generates collapsible sections for each metric.
  - Adds hash-based navigation for easy linking to specific results.
  - Implements lightbox image viewer for enlarged plots.

# Customization

## 🖌 Modify Styling
Edit the CSS files inside `css/`:
- `render.css`: Styles for `render.html` (tables, collapsibles, etc.).
- `style.css`: General site-wide styles.

## 📜 Update Experiment Data
- Store experiment results in `experiments/` as `.png` or `.tsv`.
- Update `experiments/list.json` to add new experiments.
