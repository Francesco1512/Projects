import numpy as np
import pandas as pd
from scipy.stats import norm
import math

SHIFT = 0.05  # Define shift for the Shifted Black model

def shifted_black_swaption_price(omega, f, K, T, sigma_B):
    """
    Shifted Black swaption price.
    omega = +1 for call/payer, -1 for put/receiver
    """
    f_shifted = f + SHIFT
    K_shifted = K + SHIFT
    
    if K_shifted <= 0:  # Prevent division errors
        raise ValueError("Shifted strike is non-positive, check inputs.")
    
    d1 = (math.log(f_shifted / K_shifted) + 0.5 * sigma_B**2 * T) / (sigma_B * math.sqrt(T))
    d2 = d1 - sigma_B * math.sqrt(T)
    
    return omega * (f_shifted * norm.cdf(omega * d1) - K_shifted * norm.cdf(omega * d2))

def shifted_black_vega(omega, f, K, T, sigma_B, df_data):
    """
    Computes the Shifted Black vega incorporating the discount factor.
    """
    f_shifted = f + SHIFT
    K_shifted = K + SHIFT
    
    if K_shifted <= 0:
        raise ValueError("Shifted strike is non-positive, check inputs.")
    
    d1 = (math.log(f_shifted / K_shifted) + 0.5 * sigma_B**2 * T) / (sigma_B * math.sqrt(T))
    closest_maturity_index = (df_data["maturity in years"] - T).abs().idxmin()
    discount_factor = df_data.loc[closest_maturity_index, "discount factors OIS"]
    
    return discount_factor * f_shifted * math.sqrt(T) * norm.pdf(d1)

def find_implied_vol_newton(omega, f, K, T, market_price, df_data, initial_guess=0.05, tol=1e-8, max_iter=100): 
    """
    Finds the implied volatility using the Newton-Raphson method for Shifted Black model.
    """
    sigma = initial_guess
    for _ in range(max_iter):
        price = shifted_black_swaption_price(omega, f, K, T, sigma)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = shifted_black_vega(omega, f, K, T, sigma, df_data)
        if abs(vega) < 1e-12:
            raise ValueError("Vega is too small; numerical trouble.")

        sigma_next = sigma - diff / vega
        if sigma_next < 1e-8:
            sigma_next = 1e-8

        sigma = sigma_next

    raise ValueError("Newton-Raphson did not converge.")

# Read Excel files for inputs and discount factors
corrected_inputs_path = "./advanced interest rate project/updated inputs.xlsx"
discount_factors_path = corrected_inputs_path

corrected_inputs_df = pd.read_excel(corrected_inputs_path, sheet_name='Inputs')
discount_factors_df = pd.read_excel(corrected_inputs_path, sheet_name='eur ois discount factor curve')

# Add columns to store the computed implied volatilities and vegas
corrected_inputs_df["Shifted Black mkt-implied vol"] = np.nan
corrected_inputs_df["Shifted Black vega"] = np.nan

# Loop over rows to compute implied volatilities and vegas
for index, row in corrected_inputs_df.iterrows():
    try:
        # Extract inputs
        omega = row["omega"]
        f = row["forward swap rate"]
        K = row["strikes "]
        T = row["expiry"]
        market_price = row["forward premium in dec."]

        # Compute implied volatility
        implied_vol = find_implied_vol_newton(omega, f, K, T, market_price, discount_factors_df)
        corrected_inputs_df.loc[index, "Shifted Black mkt-implied vol"] = implied_vol

        # Compute Vega
        vega = shifted_black_vega(omega, f, K, T, implied_vol, discount_factors_df)
        corrected_inputs_df.loc[index, "Shifted Black vega"] = vega

    except ValueError as e:
        corrected_inputs_df.loc[index, "Shifted Black mkt-implied vol"] = None
        corrected_inputs_df.loc[index, "Shifted Black vega"] = None
        print(f"Row {index}: Error computing values: {e}")

# Save results to a new Excel file
output_file = "./advanced interest rate project/implied_volatilities_with_vegas_shifted_black.xlsx"
corrected_inputs_df.to_excel(output_file, index=False)
print(f"Implied volatilities and vegas saved to {output_file}")