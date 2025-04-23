import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

# Load the Excel file
file_path = "./advanced interest rate project/updated inputs.xlsx"  # Update with the actual file path
df_inputs = pd.read_excel(file_path, sheet_name="Inputs")

# Define the SABR normal volatility functions
def sabr_normal_vol(F, K, T, alpha, beta, rho, nu, shift=0.043):
    if abs(K - F) == 0:
        return sabr_normal_vol_atm(F, T, alpha, beta, rho, nu, shift)

    Fb = F + shift
    Kb = K + shift
    if Fb <= 0 or Kb <= 0 or T <= 0:
        return np.nan

    alpha_bar = alpha * (1.0 + 0.25 * alpha * beta * rho * nu * (Fb**(1.0 - beta)) * T / 365)

    if abs(beta - 1.0) > 1e-12:
        num = (Kb**(1.0 - beta) - Fb**(1.0 - beta))
        den = (1.0 - beta)
        z = (nu / alpha_bar) * (num / den)
    else:
        z = (nu / alpha_bar) * math.log(Kb / Fb)

    E_z = math.sqrt(1.0 + 2.0 * rho * z + z * z)
    if E_z < 0:
        return np.nan

    yz_num = (z + rho + E_z)
    yz_den = (1.0 + rho)
    if yz_num <= 0 or abs(yz_den) < 1e-15:
        return np.nan
    Y_z = math.log(yz_num / yz_den)

    delta_num = beta * (2-beta)
    delta_den = 8 * Fb**(2-2*beta)
    delta = delta_num / delta_den

    term1 = (nu ** 2 / 24) * (-1 + 3 * ((z + rho - (rho * E_z) )/ (Y_z * E_z)))
    term2 = (alpha_bar ** 2 * delta / 6) * ((1 - rho ** 2) + ((z + rho) * E_z - rho) / (Y_z * E_z))
    O_z = term1 + term2

    if O_z >= 1e-12:
        Z_z = 1.0 + O_z * T / 365
    else:
        Z_z = (1.0 + O_z * T / 365)**(-1.0)

    sigmaN = nu * (Kb - Fb) * (Z_z / Y_z)
    return abs(sigmaN)

def sabr_normal_vol_atm(F, T, alpha, beta, rho, nu, shift=0.043):
    Fb = F + shift
    if Fb <= 0 or T <= 0:
        return np.nan

    alpha_bar = alpha * (1.0 + 0.25 * alpha * beta * rho * nu * (Fb**(1.0 - beta)) * (T / 365))
    approx_vol = alpha_bar * (Fb**beta) * (1 + ((nu**2) / 24) * (alpha_bar**2) / (Fb**(2 * beta)) + ((rho * nu * alpha_bar) / (4 * (Fb**beta))))

    return abs(approx_vol)

# Objective function: RMSE between SABR vols and market vols
def sabr_rmse(params, F, strikes, T, market_vols):
    alpha, beta, rho, nu = params
    sabr_vols = np.array([sabr_normal_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
    return math.sqrt(np.mean((sabr_vols - market_vols) ** 2))

# Extract unique expiry and forward swap rate combinations (Smile Market Sections)
smile_sections = df_inputs.groupby(["expiry", "forward swap rate"])

# Store results
sabr_parameters = []

# Initial guesses for SABR parameters
initial_guess = [0.02, 0.5, -0.5, 0.3]  # (alpha, beta, rho, nu)
bounds = [(0.001, 1.0), (0.0, 1.0), (-0.999, 0.999), (0.001, 2.0)]  # Constraints on SABR parameters

# Loop through each smile market section and calibrate SABR parameters
for (expiry, F), section in smile_sections:
    strikes = section["strikes "].values
    market_vols = section["new bach mkt impl vol"].values

    if len(strikes) < 3:  # Avoid ill-conditioned cases
        continue

    # Optimize SABR parameters
    result = minimize(
        sabr_rmse, initial_guess, args=(F, strikes, expiry, market_vols),
        method="L-BFGS-B", bounds=bounds
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

# Convert to DataFrame
df_sabr = pd.DataFrame(sabr_parameters)

# Merge back with original dataframe
df_inputs = df_inputs.merge(df_sabr, on=["expiry", "forward swap rate"], how="left")

# Save the updated Excel file
updated_file_path = "updated_corrected_inputs_with_sabr_errors.xlsx"
df_inputs.to_excel(updated_file_path, sheet_name="Inputs", index=False)

print(f"SABR calibration completed with errors and iterations added. The results are saved in '{updated_file_path}'")

# Compute SABR volatilities for each row
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
        
        sabr_vol = sabr_normal_vol(F, K, expiry, alpha, beta, rho, nu)
        sabr_vols.append(sabr_vol)
    except Exception as e:
        sabr_vols.append(np.nan)

# Add the SABR volatilities to the DataFrame
df_inputs["SABR Volatility"] = sabr_vols

# Save the updated DataFrame back to the Excel file
updated_file_path = "updated_corrected_inputs_with_sabr_vols.xlsx"
df_inputs.to_excel(updated_file_path, sheet_name="Inputs", index=False)

print(f"SABR volatilities have been computed and saved to '{updated_file_path}'")




