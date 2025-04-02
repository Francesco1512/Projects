import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

def fill_missing_strikes_pchip(
    file_name="./advanced interest rate project 2/inputs_2.xlsx",
    sheet_name="interpolation",
    output_file="./advanced interest rate project 2/pchip_filled_vols_by_strike.xlsx"
):
    """
    Reads a volatility matrix (maturity in rows, strike in columns),
    then for each row (fixed maturity) interpolates across strike using
    a PCHIP interpolator to fill missing cells. This method produces
    smooth, shape-preserving vol smiles.
    
    Parameters
    ----------
    file_name : str
        Path to the input Excel file.
    sheet_name : str
        Sheet containing the data, where:
          - Column A: 'maturity'
          - Row 0: strike values (headers)
          - The rest: vol matrix (with missing values as NaN)
    output_file : str
        Path to save the filled matrix.
    """
    # 1) Read the Excel file with header=0 (first row as column headers)
    df = pd.read_excel(file_name, sheet_name=sheet_name, header=0)
    # We assume the first column is 'maturity', and the remaining columns are numeric strikes.
    df.set_index('maturity', inplace=True)
    
    # Convert columns to float (strike values) if needed.
    df.columns = df.columns.astype(float)
    
    # 2) For each row (fixed maturity), use PCHIP interpolation across strike.
    for maturity in df.index.unique():
        # In case of duplicate maturities, take the first row.
        row_series = df.loc[maturity]
        if isinstance(row_series, pd.DataFrame):
            row_series = row_series.iloc[0]
            
        known_mask = row_series.notna()
        if int(known_mask.sum()) < 2:
            # Not enough points to interpolate (need at least two points)
            continue
        
        # Extract known strike values and corresponding vols.
        x_known = row_series.index[known_mask].values
        y_known = row_series[known_mask].values
        
        # Build a PCHIP interpolator.
        pchip = PchipInterpolator(x_known, y_known)
        
        # Identify the strikes where data is missing.
        missing_mask = row_series.isna()
        if int(missing_mask.sum()) > 0:
            x_missing = row_series.index[missing_mask].values
            # Evaluate the PCHIP interpolator at these strike values.
            y_interp = pchip(x_missing)
            # Fill the missing cells.
            df.loc[maturity, x_missing] = y_interp

    # 3) Save the completed matrix to a new Excel file.
    df.to_excel(output_file, sheet_name="PCHIPFilledByStrike")
    print(f"PCHIP interpolation by strike complete. Results saved to '{output_file}'.")

if __name__ == "__main__":
    fill_missing_strikes_pchip(
        file_name="./advanced interest rate project 2/inputs 2.xlsx",
        sheet_name="interpolation",
        output_file="./advanced interest rate project 2/pchip_filled_vols_by_strike.xlsx"
    )
 
