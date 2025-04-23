import numpy as np
import pandas as pd

# Load the input data
input_file = "./advanced interest rate project/updated inputs.xlsx"
sheet_name = " SABR sigma SLN approx"
data = pd.read_excel(input_file, sheet_name=sheet_name)

# Define the formula for Shifted Log-Normal Volatility Approximation
def compute_shifted_ln_vol(forward, strike, f_shifted, k_shifted, sigma_n, sigma_n_atm, year_fraction):
    # Avoid log issues if shifted values are non-positive
    if f_shifted <= 0 or k_shifted <= 0:
        return np.nan

    # Check for the ATM case: when forward and strike are nearly equal
    if np.isclose(forward, strike):
        # Use the limit: 1/(f_shifted) instead of the log-term
        return sigma_n / f_shifted

    # For non-ATM cases, compute normally
    log_term = np.log(f_shifted / k_shifted) / (forward - strike)
    correction_factor = 1 + ((sigma_n_atm ** 2 * year_fraction) / (24 * f_shifted * k_shifted))
    return sigma_n * log_term * correction_factor



# Add a new column for computed Shifted Log-Normal Volatility
data["Shifted Log-Normal Vol"] = np.nan

# Compute the volatility for each row
for index, row in data.iterrows():
    try:
        f = row["forward swap rate"]
        k = row["strikes "]
        f_shifted = row["shifted forward swap rate"]
        k_shifted = row["shifted strikes"]
        sigma_n = row["shifted Bachelier mkt-implied norm vol"]
        year_fraction = row["tenor"]
        tenor = row["tenor"]
        expiry = row["expiry"]  # Assuming 'tenor' column exists
        
        # Find ATM sigma_n within the correct market smile (same expiry and tenor)
        atm_row = data[(data["expiry"] == expiry) & (data["tenor"] == year_fraction) & (data["strikes "] == data["forward swap rate"])]
        sigma_n_atm = atm_row["shifted Bachelier mkt-implied norm vol"].values[0] if not atm_row.empty else sigma_n
        
        # Compute the shifted log-normal volatility
        shifted_ln_vol = compute_shifted_ln_vol(f, k, f_shifted, k_shifted, sigma_n, sigma_n_atm, year_fraction)
        data.loc[index, "Shifted Log-Normal Vol"] = shifted_ln_vol
        
    except Exception as e:
        print(f"Error at row {index}: {e}")
        data.loc[index, "Shifted Log-Normal Vol"] = np.nan

# Save results to a new Excel file
output_file = "./advanced interest rate project/shifted_ln_vol_approximation.xlsx"
data.to_excel(output_file, index=False)
print(f"Computed Shifted Log-Normal Volatilities saved to {output_file}")
