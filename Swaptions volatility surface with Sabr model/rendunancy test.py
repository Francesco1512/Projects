import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

# Load the Excel file
file_path = "./advanced interest rate project/updated inputs.xlsx"  # Update with the actual file path
df_inputs = pd.read_excel(file_path, sheet_name="comparison inputs")

# Define the SABR normal volatility functions
def sabr_normal_vol(F, K, T, alpha, beta, rho, nu, shift=0.043):
    if abs(K - F) < 1e-12:  # Handle ATM case
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

    delta_num = beta * (2 - beta)
    delta_den = 8 * Fb**(2 - 2 * beta)
    delta = delta_num / delta_den

    term1 = (nu ** 2 / 24) * (-1 + 3 * ((z + rho - (rho * E_z)) / (Y_z * E_z)))
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
def sabr_rmse(params, F, strikes, T, market_vols, fixed_param=None, fixed_value=None):
    alpha, beta, rho, nu = params
    if fixed_param == "alpha":
        alpha = fixed_value
    elif fixed_param == "beta":
        beta = fixed_value
    elif fixed_param == "rho":
        rho = fixed_value
    elif fixed_param == "nu":
        nu = fixed_value

    sabr_vols = np.array([sabr_normal_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
    return math.sqrt(np.mean((sabr_vols - market_vols) ** 2))

# Fixed values for each parameter
fixed_values = {
    "alpha": [0.01, 0.02, 0.03, 0.04],
    "beta": [0.0, 0.25, 0.5, 0.75],
    "rho": [-0.999, -0.5, 0.0, 0.5],
    "nu": [0.1, 0.2, 0.3, 0.4],
}

# Extract unique expiry and forward swap rate combinations (Smile Market Sections)
smile_sections = df_inputs.groupby(["expiry", "forward swap rate"])

# Store results
sabr_parameters = []

# Initial guesses for SABR parameters
initial_guess = [0.02, 0.5, -0.5, 0.3]  # (alpha, beta, rho, nu)
bounds = [(0.001, 1.0), (0.0, 1.0), (-0.999, 0.999), (0.001, 2.0)]  # Constraints on SABR parameters

# Loop through each parameter to test fixing it at different values
for fixed_param, values in fixed_values.items():
    for fixed_value in values:
        for (expiry, F), section in smile_sections:
            strikes = section["strikes "].values
            market_vols = section["new bach mkt impl vol"].values

            if len(strikes) < 3:  # Avoid ill-conditioned cases
                continue

            # Optimize remaining parameters
            result = minimize(
                sabr_rmse, initial_guess, args=(F, strikes, expiry, market_vols, fixed_param, fixed_value),
                method="L-BFGS-B", bounds=bounds
            )

            # Compute SABR volatilities for all strikes
            sabr_vols = [
                sabr_normal_vol(F, K, expiry, 
                                result.x[0] if fixed_param != "alpha" else fixed_value,
                                result.x[1] if fixed_param != "beta" else fixed_value,
                                result.x[2] if fixed_param != "rho" else fixed_value,
                                result.x[3] if fixed_param != "nu" else fixed_value)
                for K in strikes
            ]

            # Store the calibrated parameters, SABR volatilities, and additional data
            for K, sabr_vol in zip(strikes, sabr_vols):
                sabr_parameters.append({
                    "expiry": expiry,
                    "forward swap rate": F,
                    "strike": K,
                    "fixed_param": fixed_param,
                    "fixed_value": fixed_value,
                    "alpha": result.x[0] if fixed_param != "alpha" else fixed_value,
                    "beta": result.x[1] if fixed_param != "beta" else fixed_value,
                    "rho": result.x[2] if fixed_param != "rho" else fixed_value,
                    "nu": result.x[3] if fixed_param != "nu" else fixed_value,
                    "SABR Volatility": sabr_vol,
                    "error (RMSE)": result.fun,
                    "iterations": result.nit
                })

# Convert to DataFrame
df_sabr = pd.DataFrame(sabr_parameters)

# Save the results to an Excel file
output_file_path = "./advanced interest rate project/updated_corrected_inputs_with_fixed_sabr_parameters_and_vols.xlsx"
df_sabr.to_excel(output_file_path, index=False)

print(f"SABR calibration with fixed parameters completed. Results saved to '{output_file_path}'")
