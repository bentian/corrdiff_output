import os
import pandas as pd
from fpdf import FPDF
from glob import glob

def generate_summary_pdf(files_path, output_pdf, file_suffix_order):
    """
    Generate a summary PDF from PNG images and CSV files in the specified suffix order.

    Parameters:
        files_path (str): Path to the folder containing PNG images and CSV files.
        output_pdf (str): Output PDF file path.
        file_suffix_order (list of str): List of file suffixes specifying the desired order.
    """
    # Initialize FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Get all files and filter by suffix order
    all_files = glob(os.path.join(files_path, "*"))
    filtered_files = [
        file for suffix in file_suffix_order
        for file in all_files if os.path.basename(file).endswith(suffix)
    ]

    # Process files in the specified order
    for file in filtered_files:
        suffix = os.path.basename(file)
        if suffix.endswith(".csv"):
            # Process CSV file
            df = pd.read_csv(file)

            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Table: {suffix}", ln=True, align="C")
            pdf.ln(10)  # Line break

            # Add the table headers
            col_width = pdf.w / (len(df.columns) + 1)
            row_height = 8
            for col in df.columns:
                pdf.set_font("Arial", size=10, style="B")
                pdf.cell(col_width, row_height, col, border=1, align="C")
            pdf.ln(row_height)

            # Add the table rows
            pdf.set_font("Arial", size=10)
            for _, row in df.iterrows():
                for value in row:
                    pdf.cell(col_width, row_height, str(value), border=1, align="C")
                pdf.ln(row_height)

        elif suffix.endswith(".png"):
            # Process PNG file
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Image: {suffix}", ln=True, align="C")
            pdf.image(file, x=10, y=30, w=180)

    # Save the PDF
    pdf.output(output_pdf)
    print(f"PDF generated: {output_pdf}")


# Example usage
file_suffix_order = [
    "metrics_mean.csv", "metrics_mean.png",
    "monthly_mae.csv", "monthly_rmse.csv", "monthly_metrics.png",
    "pdf.png", "cdf.png",
    "monthly_mean_prcp.png", "monthly_mean_t2m.png",
    "monthly_mean_u10m.png", "monthly_mean_v10m.png"
]
generate_summary_pdf("./plots", "summary.pdf", file_suffix_order)
