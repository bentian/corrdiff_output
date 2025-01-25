import os
import pandas as pd
from fpdf import FPDF
from glob import glob

FONT = "Times"

def add_table(pdf, file, filename, idx):
    df = pd.read_csv(file)

    pdf.add_page()
    pdf.set_font(FONT, size=12, style="B")
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
    pdf.set_font(FONT, size=12, style="B")
    pdf.cell(200, 10, txt=f"Figure {idx}. {filename[:-4]}", ln=True, align="C")
    pdf.image(file, x=10, y=30, w=220)

def generate_summary_pdf(files_path, output_pdf, file_suffix_order):
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

def generate_summary(folder, prefix=''):
    file_suffix_order = [
        # regression + diffusion model
        "all-metrics_mean.csv", "all-metrics_mean.png",
        "all-monthly_mae.csv", "all-monthly_mae.png",
        "all-monthly_rmse.csv", "all-monthly_rmse.png",
        # regression + diffusion model minus regression model only
        "minus_reg-metrics_mean.csv", "minus_reg-metrics_mean.png",
        "minus_reg-monthly_mae.csv", "minus_reg-monthly_mae.png",
        "minus_reg-monthly_rmse.csv", "minus_reg-monthly_rmse.png",
        # regression + diffusion model
        "_pdf.png", "_pdf_clipped.png",
        "all-monthly_error_prcp.png", "all-monthly_error_t2m.png",
        "all-monthly_error_u10m.png", "all-monthly_error_v10m.png"
    ]

    generate_summary_pdf(folder, f"{prefix}_summary.pdf", file_suffix_order)

if __name__ == "__main__":
    generate_summary("./plots")
