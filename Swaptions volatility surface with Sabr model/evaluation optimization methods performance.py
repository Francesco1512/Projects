import pandas as pd
import numpy as np
import math
import time
import cma
from scipy.optimize import minimize

# Load the Excel file
file_path = "./advanced interest rate project/updated inputs.xlsx"
df_inputs = pd.read_excel(file_path, sheet_name="comparison inputs")

def sabr_normal_vol(F, K, T, alpha, beta, rho, nu, shift=0.05):
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

    # Constrain z to avoid extreme values
    z = max(min(z, 1e3), -1e3)

    # Ensure square root term is non-negative
    E_z_term = 1.0 + 2.0 * rho * z + z**2
    if E_z_term < 0:
        return np.nan  # Gracefully handle invalid square root
    E_z = math.sqrt(E_z_term)

    yz_num = (z + rho + E_z)
    yz_den = (1.0 + rho)
    if yz_num <= 0 or abs(yz_den) < 1e-15:
        return np.nan
    Y_z = math.log(yz_num / yz_den)

    delta_num = beta * (2 - beta)
    delta_den = 8 * Fb**(2 - 2 * beta)
    delta = delta_num / delta_den

    term1 = (nu**2 / 24) * (-1 + 3 * ((z + rho - (rho * E_z)) / (Y_z * E_z)))
    term2 = (alpha_bar**2 * delta / 6) * ((1 - rho**2) + ((z + rho) * E_z - rho) / (Y_z * E_z))
    O_z = term1 + term2

    if O_z >= 1e-12:
        Z_z = 1.0 + O_z * T / 365
    else:
        Z_z = (1.0 + O_z * T / 365)**(-1.0)

    sigmaN = nu * (Kb - Fb) * (Z_z / Y_z)
    return abs(sigmaN)

def sabr_normal_vol_atm(F, T, alpha, beta, rho, nu, shift=0.05):
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

# Store results for each optimization method
results = []

# Optimization methods to evaluate
methods = ["L-BFGS-B", "Nelder-Mead", "Powell", "CMA-ES"]

# Initial guesses for SABR parameters
initial_guess = [0.02, 0.5, -0.5, 0.3]  # (alpha, beta, rho, nu)
bounds = [(0.001, 1.0), (0.0, 1.0), (-0.98, 0.98), (0.001, 2.0)]  # Constraints on SABR parameters

# Loop through each smile market section and calibrate SABR parameters using all methods
for method in methods:
    for (expiry, F), section in smile_sections:
        strikes = section["strikes "].values
        market_vols = section["new bach mkt impl vol"].values

        if len(strikes) < 3:  # Avoid ill-conditioned cases
            continue

        start_time = time.time()

        if method == "CMA-ES":
            # CMA-ES optimization
            es = cma.CMAEvolutionStrategy(initial_guess, 0.1, {'bounds': [list(zip(*bounds))[0], list(zip(*bounds))[1]]})
            es.optimize(lambda x: sabr_rmse(x, F, strikes, expiry, market_vols))
            result_params = es.result.xbest
            result_rmse = es.result.fbest
            iterations = es.result.iterations
            converged = es.stop()

        else:
            # Gradient-based optimization
            result = minimize(
                sabr_rmse, initial_guess, args=(F, strikes, expiry, market_vols),
                method=method, bounds=bounds if method in ["L-BFGS-B"] else None
            )
            result_params = result.x if result.success else [np.nan] * 4
            result_rmse = result.fun if result.success else np.nan
            iterations = result.nit if result.success else np.nan
            converged = result.success

        end_time = time.time()

        # Store the calibrated parameters and additional data
        results.append({
            "expiry": expiry,
            "forward swap rate": F,
            "method": method,
            "alpha": result_params[0],
            "beta": result_params[1],
            "rho": result_params[2],
            "nu": result_params[3],
            "error (RMSE)": result_rmse,
            "iterations": iterations,
            "time (s)": end_time - start_time,
            "converged": converged
        })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save the results to an Excel file
output_file = "./advanced interest rate project/task 1/sabr_calibration_results_cma_es.xlsx"
df_results.to_excel(output_file, index=False)

# Provide the download link
output_file

