import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

# Load the Excel file
file_path = "./advanced interest rate project/updated inputs.xlsx"  # Update with the actual file path
df_inputs = pd.read_excel(file_path, sheet_name="Inputs")

# Define the SABR normal volatility functions
def sabr_shifted_ln_normal_vol(F, K, T, alpha, beta, rho, nu, shift=0.05):
    if abs(K - F) < 1e-12:
        return sabr_shifted_ln_normal_vol_atm(F, T, alpha, beta, rho, nu, shift)
    Fb = F + shift
    Kb = K + shift
    if Fb <= 0 or Kb <= 0 or T <= 0:
        return np.nan
    
    try:
        d = math.log(Fb / Kb)
        z = (nu / alpha) * (Fb * Kb)**((1 - beta) / 2) * d
        num_y = math.sqrt(1 - 2 * rho * z + z**2) + z - rho
        den_y = 1 - rho
        Y_z = math.log(num_y / den_y)
        B_p = 1 + (1/24) * ((1 - beta) * math.log(Fb / Kb))**2 + (1/1920) * ((1 - beta) * math.log(Fb / Kb))**4
        A_p = 1 + (((alpha**2 * (1-beta)**2) / (24 * (Fb*Kb)**(1-beta))) + 
                   ((alpha * beta * rho * nu) / (4 * (Fb*Kb)**(0.5*(1-beta)))) + 
                   ((2-3*rho*rho)*nu*nu)/24) * T
        sigma_shifted_LN = (nu * d) / (Y_z) * (A_p / B_p)
    except Exception:
        return np.nan
    return abs(sigma_shifted_LN)


def sabr_shifted_ln_normal_vol_atm(F, T, alpha, beta, rho, nu, shift=0.05):
    Fb = F + shift
    if Fb <= 0 or T <= 0:
        return np.nan
    try:
        approx_vol = (alpha / Fb**(1-beta)) * (
            1 + (((1-beta)**2 * alpha**2) / (24 * Fb**(2-2*beta)) +
                 (0.25 * rho * beta * nu * alpha) / (Fb**(1-beta)) +
                 ((2-3*rho**2) * nu**2) / 24) * T
        )
    except Exception:
        return np.nan
    return approx_vol


def sabr_rmse(params, F, strikes, T, market_vols):
    alpha, beta, rho, nu = params
    sabr_vols = np.array([sabr_shifted_ln_normal_vol(F, K, T, alpha, beta, rho, nu, shift=0.05) for K in strikes])
    return math.sqrt(np.mean((sabr_vols - market_vols) ** 2))


# Extract unique expiry and forward swap rate combinations (Smile Market Sections)
smile_sections = df_inputs.groupby(["expiry", "forward swap rate"])

# Store results
sabr_parameters = []

# Initial guesses for SABR parameters and bounds
initial_guess = [0.09, 1, 0.09, 0.5]  # (alpha, beta, rho, nu)
bounds = [(0.001, 1.0), (0.0, 1.0), (-0.999, 0.999), (0.001, 2.0)]  # Constraints on SABR parameters

# Loop through each smile market section and calibrate SABR parameters
for (expiry, F), section in smile_sections:
    strikes = section["strikes "].values
    market_vols = section["Shifted mkt-implid ln-vol from approximatin formula"].values

    # --- Filter out rows with missing market vol data ---
    valid_mask = ~np.isnan(market_vols)
    if np.sum(valid_mask) < 3:
        # Skip calibration for this group if fewer than 3 valid market vols are available
        continue
    strikes = strikes[valid_mask]
    market_vols = market_vols[valid_mask]
    
    # Optimize SABR parameters using L-BFGS-B
    result = minimize(
        sabr_rmse, initial_guess, args=(F, strikes, expiry, market_vols),
        method="L-BFGS-B", bounds=bounds, options={"ftol":1e-12, "gtol":1e-12, "maxiter":1000}
    )
    
    # Store the calibrated parameters and additional data
    sabr_parameters.append({
        "expiry": expiry,
        "forward swap rate": F,
        "alpha": result.x[0],
        "beta": result.x[1],
        "rho": result.x[2],
        "nu": result.x[3],
        "error (RMSE)": result.fun,  # Calibration error
        "iterations": result.nit  # Number of iterations
    })

# Convert calibrated parameters to DataFrame
df_sabr = pd.DataFrame(sabr_parameters)

# Merge back with original dataframe
df_inputs = df_inputs.merge(df_sabr, on=["expiry", "forward swap rate"], how="left")

# Save the updated Excel file with calibration results
updated_file_path = "updated_corrected_inputs_with_sabr_errors.xlsx"
df_inputs.to_excel(updated_file_path, sheet_name="inputs", index=False)
print(f"SABR calibration completed with filtering out groups missing market vol data. Results are saved in '{updated_file_path}'")

# Compute SABR volatilities for each row using the calibrated parameters
sabr_vols = []
for _, row in df_inputs.iterrows():
    try:
        expiry = row["expiry"]
        F = row["forward swap rate"]
        K = row["strikes "]
        alpha = row["alpha"]
        beta = row["beta"]
        rho = row["rho"]
        nu = row["nu"]
        vol = sabr_shifted_ln_normal_vol(F, K, expiry, alpha, beta, rho, nu)
        sabr_vols.append(vol)
    except Exception as e:
        sabr_vols.append(np.nan)

df_inputs["SABR Volatility"] = sabr_vols

# Save the updated DataFrame with computed volatilities
final_output_file = "updated_corrected_inputs_with_sabr_vols.xlsx"
df_inputs.to_excel(final_output_file, sheet_name="Inputs", index=False)
print(f"SABR volatilities have been computed and saved to '{final_output_file}'")
