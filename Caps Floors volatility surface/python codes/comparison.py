import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_four_shift_vols(file_name="./advanced interest rate project 2/inputs 2.xlsx",
                         sheet_name="comparison"):
    """
    Reads an Excel sheet where:
      - Each row corresponds to a different shift (e.g. 0.03, 0.02, 0.01, 0.075, 0.005).
      - Each column is a strike (e.g. -0.015, -0.0125, -0.01, ...).
      - Each cell is the vol for that (shift, strike).
    Then plots four of these shift-vol curves on the same chart.
    """

    # 1) Read the Excel file; assume first row has column headers (strikes).
    #    Each row is then a shift's vol series across strikes.
    df = pd.read_excel(file_name, sheet_name=sheet_name, header=0)

    # By default, pandas will label the rows 0,1,2,... 
    # We'll define a list of shift labels in the order they appear:
    shift_labels = [0.03, 0.02, 0.01, 0.075, 0.005]  # Adjust if needed

    # 2) Convert the column headers (strikes) to float if they're not already.
    df.columns = df.columns.astype(float)

    # 3) Decide which four rows (shifts) to plot (by index). 
    #    For example, we'll plot the first 4 rows: 0 -> shift=0.03, 1 -> 0.02, 2 -> 0.01, 3 -> 0.075
    rows_to_plot = [0, 1, 2, 3, 4]  # You can change these indices as desired

    # 4) Prepare the plot
    plt.figure(figsize=(8,5))

    # 'df.columns' are the strike values
    strikes = df.columns.values

    for row_idx in rows_to_plot:
        if row_idx >= len(df):
            print(f"Row index {row_idx} is out of range for this sheet.")
            continue

        # Extract the vol series for this row
        vols = df.iloc[row_idx].values

        # Plot
        plt.plot(strikes, vols, marker='', label=f"Shift={shift_labels[row_idx]}")

    plt.title("impl ln-vols (maturity fixed at 20Y) for different shifts")
    plt.xlabel("Strike")
    plt.ylabel("ln-vol")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_four_shift_vols(
        file_name="./advanced interest rate project 2/inputs 2.xlsx",
        sheet_name="comparison"
    )
