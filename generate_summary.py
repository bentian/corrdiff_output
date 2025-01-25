import os
import argparse
import pandas as pd
from fpdf import FPDF
from glob import glob

FONT = "Times"

def add_config_page(pdf, config_path):
    """
    Add a configuration page to the PDF from a YAML file, formatted as a table in landscape orientation.
    """
    import yaml

    # Load the YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Add a new landscape page for the config table
    pdf.add_page(orientation="L")  # Landscape orientation
    pdf.set_font(FONT, size=14, style="B")
    pdf.cell(0, 10, txt="Configuration", ln=True, align="C")
    pdf.ln(10)  # Line break

    # Table column setup
    pdf.set_font(FONT, size=10)
    col_widths = [80, 200]  # Adjust column widths for landscape layout
    row_height = 8

    # Add table headers
    pdf.set_font(FONT, size=12, style="B")
    pdf.cell(col_widths[0], row_height, "Parameter", border=1, align="C")
    pdf.cell(col_widths[1], row_height, "Value", border=1, align="C")
    pdf.ln(row_height)

    # Add table rows (key-value pairs)
    pdf.set_font(FONT, size=10)
    if isinstance(config, list):  # If the YAML is a list of key-value pairs
        for item in config:
            # Split key-value pair
            if '=' in item:
                key, value = item.split('=', 1)
            else:
                key, value = item, ''

            # Convert lists to comma-separated strings for display
            if value.startswith('[') and value.endswith(']'):  # Handle list-like strings
                value = value.strip('[]').replace(',', ', ')

            # Add table cells (no multi-line wrapping)
            pdf.cell(col_widths[0], row_height, str(key), border=1, align="L")
            pdf.cell(col_widths[1], row_height, str(value), border=1, align="L")
            pdf.ln(row_height)
    else:
        raise ValueError("The YAML configuration is not in the expected list format.")


def add_table(pdf, file, filename, idx):
    df = pd.read_csv(file)

    pdf.add_page()
    pdf.set_font(FONT, size=14, style="B")
    pdf.cell(200, 10, txt=f"Table {idx}. {filename[:-4]}", ln=True, align="C")
    pdf.ln(10)  # Line break

    # Add the table headers
    col_width = pdf.w / (len(df.columns) + 1)
    row_height = 8
    for col in df.columns:
        pdf.set_font(FONT, size=10, style="B")
        pdf.cell(col_width, row_height, col, border=1, align="C")
    pdf.ln(row_height)

    # Add the table rows
    pdf.set_font(FONT, size=10)
    for _, row in df.iterrows():
        for value in row:
            pdf.cell(col_width, row_height, str(value), border=1, align="C")
        pdf.ln(row_height)


def add_figure(pdf, file, filename, idx):
    pdf.add_page()
    pdf.set_font(FONT, size=14, style="B")
    pdf.cell(200, 10, txt=f"Figure {idx}. {filename[:-4]}", ln=True, align="C")
    pdf.image(file, x=10, y=30, w=220)


def generate_summary_pdf(files_path, output_pdf, file_suffix_order, config_path=None):
    """
    Generate a summary PDF from PNG images and CSV files in the specified suffix order.

    Parameters:
        files_path (str): Path to the folder containing PNG images and CSV files.
        output_pdf (str): Output PDF file path.
        file_suffix_order (list of str): List of file suffixes specifying the desired order.
    """
    # Initialize FPDF
    pdf = FPDF(orientation='L')
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add configuration page
    if config_path:
        add_config_page(pdf, config_path)

    # Get all files and filter by suffix order
    all_files = glob(os.path.join(files_path, "*"))
    filtered_files = [
        file for suffix in file_suffix_order
        for file in all_files if os.path.basename(file).endswith(suffix)
    ]

    indices = {"csv": 1, "png": 1}
    handlers = {
        "csv": add_table,
        "png": add_figure
    }

    # Add files into PDF
    for file in filtered_files:
        filename = os.path.basename(file)
        ext = filename.split(".")[-1]
        if ext in handlers:
            handlers[ext](pdf, file, filename, indices[ext])
            indices[ext] += 1

    # Save the PDF
    pdf.output(output_pdf)
    print(f"PDF generated => {output_pdf}")


def generate_summary(folder, output_pdf, config_path=None):
    file_suffix_order = [
        # regression + diffusion model
        "all-metrics_mean.csv", "all-metrics_mean.png",
        "all-monthly_mae.csv", "all-monthly_mae.png",
        "all-monthly_rmse.csv", "all-monthly_rmse.png",
        "pdf_prcp.png", "pdf_t2m.png",
        "pdf_u10m.png", "pdf_v10m.png",
        "monthly_error_prcp.png", "monthly_error_t2m.png",
        "monthly_error_u10m.png", "monthly_error_v10m.png",

        # regression + diffusion model minus regression model only
        "minus_reg-metrics_mean.csv", "minus_reg-metrics_mean.png",
        "minus_reg-monthly_mae.csv", "minus_reg-monthly_mae.png",
        "minus_reg-monthly_rmse.csv", "minus_reg-monthly_rmse.png",
    ]

    generate_summary_pdf(folder, output_pdf, file_suffix_order, config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str, help="Folder to read the NetCDF files and config")
    parser.add_argument("--config-path", type=int, help="Path to YAML file of config")
    args = parser.parse_args()

    generate_summary(args.in_dir, "summary.pdf")
