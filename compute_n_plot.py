import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from glob import glob

from aggregate_metrics import aggregate_metrics
from create_prcp_pdf import create_prcp_pdf
from score_samples import main as score_samples

def generate_summary_pdf(images_path, csv_files_path, output_pdf):
    """
    Generate a summary PDF from PNG images and CSV files.

    Parameters:
        images_path (str): Path to the folder containing PNG images.
        csv_files_path (str): Path to the folder containing CSV files.
        output_pdf (str): Output PDF file path.
    """
    # Initialize FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Process CSV files
    csv_files = glob(os.path.join(csv_files_path, "*.csv"))
    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Add a new page for each CSV table
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Table: {os.path.basename(csv_file)}", ln=True, align="C")
        pdf.ln(10)  # Line break

        # Add the table headers
        col_width = pdf.w / (len(df.columns) + 1)  # Dynamically calculate column width
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

    # Process PNG images
    png_files = glob(os.path.join(images_path, "*.png"))
    for png_file in png_files:
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Image: {os.path.basename(png_file)}", ln=True, align="C")
        pdf.image(png_file, x=10, y=30, w=180)

    # Save the PDF
    pdf.output(output_pdf)
    print(f"PDF generated: {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Path for the output file.")
    parser.add_argument("score", type=str, help="Path to the score file.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    args = parser.parse_args()

    # # CorrDiff scores
    score_samples(args.output, args.score, args.n_ensemble)

    # # Aggregate metrics
    aggregate_metrics(args.score, args.score[:-3])

    # # Create PRCP PDF
    create_prcp_pdf(args.output, args.output[:-3], args.n_ensemble)

    generate_summary_pdf("./plots", "./plots", args.output[:-3] + "_summary.pdf")
