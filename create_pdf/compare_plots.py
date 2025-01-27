import os
import sys
import pandas as pd
from fpdf import FPDF
from glob import glob

from generate_summary import SUFFIX_ORDER

def add_comparison_page(pdf, file1, file2, filename):
    """
    Add a comparison page to the PDF for the given files (CSV or PNG).
    """

    pdf.add_page(orientation="P")  # Portrait orientation
    pdf.set_font("Times", size=14, style="B")
    pdf.cell(0, 10, txt=f"Comparison: {filename}", ln=True, align="C")
    pdf.ln(10)  # Line break

    if file1.endswith(".csv") and file2.endswith(".csv"):
        # Load CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Add folder 1 table
        pdf.set_font("Times", size=12, style="U")
        pdf.cell(0, 10, txt=f"{file1}", ln=True, align="L")
        pdf.ln(5)
        add_table_to_pdf(pdf, df1, pdf.w - 20)

        pdf.ln(10)  # Add space after the table

        # Add folder 2 table
        pdf.set_font("Times", size=12, style="U")
        pdf.cell(0, 10, txt=f"{file2}", ln=True, align="L")
        pdf.ln(5)
        add_table_to_pdf(pdf, df2, pdf.w - 20)
    elif file1.endswith(".png") and file2.endswith(".png"):
        # Add images scaled to fit the page height
        pdf.set_font("Times", size=12, style="U")
        pdf.cell(0, 10, txt=f"{file1}", ln=True, align="L")
        pdf.image(file1, x=30, y=pdf.get_y() + 5, h=90)  # Fixed height of 90
        pdf.ln(100)  # Add space after the image
        pdf.cell(0, 10, txt=f"{file2}", ln=True, align="L")
        pdf.image(file2, x=30, y=pdf.get_y() + 5, h=90)  # Fixed height of 90
    else:
        pdf.cell(0, 10, txt="File type not supported for comparison.", ln=True, align="L")


def add_table_to_pdf(pdf, df, max_width):
    """
    Add a Pandas DataFrame as a table to the PDF, adjusted to fit the page width.
    """
    col_widths = [max_width / len(df.columns)] * len(df.columns)
    row_height = 8

    # Add table header
    pdf.set_font("Times", size=10, style="B")
    for col, col_width in zip(df.columns, col_widths):
        pdf.cell(col_width, row_height, str(col), border=1, align="C")
    pdf.ln(row_height)

    # Add table rows
    pdf.set_font("Times", size=10)
    for _, row in df.iterrows():
        for value, col_width in zip(row, col_widths):
            pdf.cell(col_width, row_height, str(value), border=1, align="C")
        pdf.ln(row_height)


def generate_comparison_pdf(folder1, folder2, output_pdf):
    """
    Generate a portrait PDF comparing plots (PNG/CSV) in two folders.
    Each page lists the same file from both folders for comparison.
    """
    # Initialize PDF
    pdf = FPDF(orientation="P")
    pdf.set_auto_page_break(auto=True, margin=15)

    # Get all files from both folders
    files1 = glob(os.path.join(folder1, "*"))
    files2 = glob(os.path.join(folder2, "*"))

    # Match files by name (common files)
    common_files = set(os.path.basename(f) for f in files1).intersection(
        os.path.basename(f) for f in files2
    )
    if not common_files:
        print("No common files found for comparison.")
        return

    # Filter commaon files by suffix order
    filtered_files = [
        file for suffix in SUFFIX_ORDER
        for file in common_files if os.path.basename(file).endswith(suffix)
    ]

    # Add each common file to the PDF
    for filename in filtered_files:
        file1 = os.path.join(folder1, filename)
        file2 = os.path.join(folder2, filename)
        add_comparison_page(pdf, file1, file2, filename)

    # Save the PDF
    pdf.output(output_pdf)
    print(f"Comparison PDF generated => {output_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_plots.py <folder2> <folder2>")
        print("  e.g., $python compare_plots.py Baseline/ D1/")
        sys.exit(1)

    generate_comparison_pdf(sys.argv[1], sys.argv[2], "comparison.pdf")
