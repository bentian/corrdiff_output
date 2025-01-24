import os
import pandas as pd
from fpdf import FPDF
from glob import glob

class PDF(FPDF):
    def __init__(self):
        super().__init__(orientation='L')
        self.alias_nb_pages()

    def footer(self):
        # Add a page number at the bottom of each page
        self.set_y(-15)  # Position the footer 15mm from the bottom
        self.set_font("Arial", size=10)
        self.cell(0, 10, f"Page {self.page_no()}/{self.alias_nb_pages()}", align="C")

def generate_summary_pdf(files_path, output_pdf, file_suffix_order):
    """
    Generate a summary PDF from PNG images and CSV files in the specified suffix order.

    Parameters:
        files_path (str): Path to the folder containing PNG images and CSV files.
        output_pdf (str): Output PDF file path.
        file_suffix_order (list of str): List of file suffixes specifying the desired order.
    """
    # Initialize FPDF
    pdf = PDF()
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
            pdf.cell(200, 10, txt=f"Table - {suffix[:-4]}", ln=True, align="C")
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
            pdf.cell(200, 10, txt=f"Image - {suffix[:-4]}", ln=True, align="C")
            pdf.image(file, x=10, y=30, w=220)

    # Save the PDF
    pdf.output(output_pdf)
    print(f"PDF generated: {output_pdf}")

def generate_summary(folder, prefix=''):
    file_suffix_order = [
        # regression + diffusion model
        "all_metrics_mean.csv", "all_metrics_mean.png",
        "all_monthly_mae.csv", "all_monthly_mae.png",
        "all_monthly_rmse.csv", "all_monthly_rmse.png",
        # regression + diffusion model minus regression model only
        "minus_reg_metrics_mean.csv", "minus_reg_metrics_mean.png",
        "minus_reg_monthly_mae.csv", "minus_reg_monthly_mae.png",
        "minus_reg_monthly_rmse.csv", "minus_reg_monthly_rmse.png",
        # regression + diffusion model
        "_pdf.png", "all_monthly_mean_prcp.png", "all_monthly_mean_t2m.png",
        "all_monthly_mean_u10m.png", "all_monthly_mean_v10m.png"
    ]

    generate_summary_pdf(folder, f"{prefix}_summary.pdf", file_suffix_order)

if __name__ == "__main__":
    generate_summary("./plots")
