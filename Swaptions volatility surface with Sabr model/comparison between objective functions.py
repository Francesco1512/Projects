import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

# Load the Excel file
file_path = "./advanced interest rate project/updated inputs.xlsx"  # Update with the actual file path
df_inputs = pd.read_excel(file_path, sheet_name="comparison inputs")

# Define SABR normal volatility
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

    delta_num = beta * (2 - beta)
    delta_den = 8 * Fb**(2 - 2 * beta)
    delta = delta_num / delta_den

    term1 = (nu ** 2 / 24) * (-1 + 3 * ((z + rho - (rho * E_z)) / (Y_z * E_z)))
    term2 = (alpha_bar ** 2 * delta / 6) * ((1 - rho ** 2) + ((z + rho) * E_z - rho) / (Y_z * E_z))
    O_z = term1 + term2

    if O_z >= 1e-12:
        Z_z = 1.0 + O_z * T / 365
    else:
        Z_z = (1.0 + O_z * T / 365) ** (-1.0)

    sigmaN = nu * (Kb - Fb) * (Z_z / Y_z)
    return abs(sigmaN)

def sabr_normal_vol_atm(F, T, alpha, beta, rho, nu, shift=0.043):
    Fb = F + shift
    if Fb <= 0 or T <= 0:
        return np.nan

    alpha_bar = alpha * (1.0 + 0.25 * alpha * beta * rho * nu * (Fb**(1.0 - beta)) * (T / 365))
    approx_vol = alpha_bar * (Fb**beta) * (1 + ((nu**2) / 24) * (alpha_bar**2) / (Fb**(2 * beta)) + ((rho * nu * alpha_bar) / (4 * (Fb**beta))))

    return abs(approx_vol)

# Objective functions
def relative_price_diff(params, F, strikes, T, market_vols):
    alpha, beta, rho, nu = params
    model_vols = np.array([sabr_normal_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
    return np.mean(np.abs((model_vols - market_vols) / market_vols))

def rmse_volatility(params, F, strikes, T, market_vols):
    alpha, beta, rho, nu = params
    model_vols = np.array([sabr_normal_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
    return np.sqrt(np.mean((model_vols - market_vols) ** 2))

def vega_weighted_rmse(params, F, strikes, T, market_vols, vegas):
    alpha, beta, rho, nu = params
    model_vols = np.array([sabr_normal_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
    return np.sqrt(np.sum(vegas * (model_vols - market_vols) ** 2) / np.sum(vegas))

# Extract unique expiry and forward swap rate combinations (Smile Market Sections)
smile_sections = df_inputs.groupby(["expiry", "forward swap rate"])

# Store results
sabr_parameters = []

# Initial guesses and bounds
initial_guess = [0.02, 0.5, -0.5, 0.3]  # (alpha, beta, rho, nu)
bounds = [(0.001, 1.0), (0.0, 1.0), (-0.999, 0.999), (0.001, 2.0)]  # Parameter constraints

# Loop through each smile market section and calibrate SABR parameters
for (expiry, F), section in smile_sections:
    strikes = section["strikes "].values
    market_vols = section["new bach mkt impl vol"].values
    vegas = section["new vega"].values 

    if len(strikes) < 3:  # Avoid ill-conditioned cases
        continue

    # Calibrate using each objective function
    for objective_func, obj_name in [(relative_price_diff, "RPD"), (rmse_volatility, "RMSE"), (vega_weighted_rmse, "VW RMSE")]:
        args = (F, strikes, expiry, market_vols) if obj_name != "VW RMSE" else (F, strikes, expiry, market_vols, vegas)
        
        result = minimize(
            objective_func, initial_guess, args=args, bounds=bounds, method="L-BFGS-B"
        )

        # Store the results
        sabr_parameters.append({
            "expiry": expiry,
            "forward swap rate": F,
            "objective": obj_name,
            "alpha": result.x[0],
            "beta": result.x[1],
            "rho": result.x[2],
            "nu": result.x[3],
            "error (objective)": result.fun,
            "iterations": result.nit
        })

# Convert to DataFrame and merge with inputs
df_sabr = pd.DataFrame(sabr_parameters)
df_inputs = df_inputs.merge(df_sabr, on=["expiry", "forward swap rate"], how="left")

# Save the updated DataFrame
updated_file_path = "./advanced interest rate project/task 1/updated_corrected_inputs_with_multiple_objectives.xlsx"
df_inputs.to_excel(updated_file_path, sheet_name="Inputs", index=False)

print(f"SABR calibration completed with multiple objective functions. Results saved to '{updated_file_path}'")
