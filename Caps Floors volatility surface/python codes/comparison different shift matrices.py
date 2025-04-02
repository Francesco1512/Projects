import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_matrices_from_sheet(file_name, sheet_name):
    """
    Reads the sheet 'sheet_name' from 'file_name' and slices out 5 blocks of data.
    Each block is one matrix of implied vol for a given shift.
    
    Adjust the row/column boundaries below to match your actual Sheet5 layout.
    """
    # Read the entire sheet
    df_full = pd.read_excel(file_name, sheet_name=sheet_name, header=None)
    
    # df_full is a big DataFrame with all rows/cols from the sheet. We'll slice out blocks manually.
    # Example:
    #   * Block 1: rows 1..10, columns 0..6
    #   * Block 2: rows 14..23, columns 0..6
    #   * etc...
    # In your actual code, you must adjust these row/column ranges to match your layout in Sheet5.

    # Example row/column slices (dummy placeholders):
    block1 = df_full.iloc[3:18, 0:15].copy()    # 10 rows, 8 columns
    block2 = df_full.iloc[24:39, 0:15].copy()   # next 10 rows
    block3 = df_full.iloc[45:60, 0:15].copy()
    block4 = df_full.iloc[66:81, 0:15].copy()
    block5 = df_full.iloc[87:102, 0:15].copy()
    block6 = df_full.iloc[108:123, 0:15].copy()    # 10 rows, 8 columns
    block7 = df_full.iloc[129:144, 0:15].copy()   # next 10 rows
    block8 = df_full.iloc[150:165, 0:15].copy()
    block9 = df_full.iloc[171:186, 0:15].copy()
    block10 = df_full.iloc[192:207, 0:15].copy()

    # Suppose we label the shifts
    shifts = [0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.0075, 0.0075, 0.005, 0.005]
    blocks = [block1, block2, block3, block4, block5, block6, block7, block8, block9, block10]

    # Now, let's convert each block to a "clean" matrix
    matrices = []
    for shift, blk in zip(shifts, blocks):
        # The first column might be 'maturity'
        # The remaining columns might be vol data for various strikes
        # We'll rename columns so that the first column is "maturity" and the rest are numeric strikes.
        # If your sheet has a header row inside each block, you'll need to handle that differently.
        
        # Example: treat row 0 of each block as a header for the columns
        # If you have a row of strikes, you might do something like:
        # col_header = blk.iloc[0].values  # the row with strike labels
        # Then the actual data is blk.iloc[1:]...
        # But if your block doesn't have a separate header row, skip this step.

        # For simplicity, let's assume the first column is maturity, the rest are vol columns.
        blk.columns = ["maturity", "atm ", "-1.5", "-1.25", "-1", "-0.5", "-0.25", "0", "0.25", "0.5", "1", "1.5", "2", "5", "10"]
        # Drop any fully-NaN rows if needed
        blk.dropna(how="all", inplace=True)

        # Set maturity as index
        blk.set_index("maturity", inplace=True)

        # Store shift in a new column or separate attribute
        blk["Shift"] = shift

        matrices.append(blk)

    return matrices

def compare_vol_matrices(matrices):
    """
    Plots a bar chart of the number of computed vols (non-NaN) for each matrix.
    Odd blocks => Brent method
    Even blocks => Newton-Raphson method
    Different colors are assigned depending on the method.
    """
    # Count non-NaN in each matrix
    counts = [m.drop(columns="Shift").notna().sum().sum() for m in matrices]
    # Shifts for labeling
    shifts = [m["Shift"].iloc[0] for m in matrices]  # assuming each matrix has a consistent shift

    # We'll label odd-indexed blocks as Brent, even-indexed blocks as Newton
    # (Adjust if you want the opposite).
    method_labels = []
    colors = []
    for i in range(len(matrices)):
        if i % 2 == 0:
            method_labels.append("Brent")
            colors.append("skyblue")  # e.g. a shade of blue
        else:
            method_labels.append("Newton")
            colors.append("orange")   # e.g. a shade of orange

    # Build an x-axis label that combines shift + method
    # e.g. "0.03 (Brent)" or "0.03 (Newton)"
    x_labels = [f"{shifts[i]} ({method_labels[i]})" for i in range(len(matrices))]

    # Plot the bar chart
    plt.figure(figsize=(10, 5))
    x_positions = range(len(matrices))

    plt.bar(x_positions, counts, color=colors, edgecolor="black")

    plt.xlabel("Shift and Method")
    plt.ylabel("Count of Computed Vols")
    plt.title("Completeness: Number of Computed Vols per Shift & Method")

    # Replace numeric x-ticks with the combined labels
    plt.xticks(ticks=x_positions, labels=x_labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


    
def main():
    file_name = "./advanced interest rate project 2/inputs 2.xlsx"
    sheet_name = "Sheet6"  # Adjust to your actual sheet name

    matrices = load_matrices_from_sheet(file_name, sheet_name)
    compare_vol_matrices(matrices)

if __name__ == "__main__":
    main()
