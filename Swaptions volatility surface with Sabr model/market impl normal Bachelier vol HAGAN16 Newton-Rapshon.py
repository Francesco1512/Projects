import numpy as np
import pandas as pd
from scipy.stats import norm
import math

def bachelier_swaption_price(omega, f, K, T, sigma_N):
    """
    Bachelier (normal) swaption price.
    omega = +1 for call/payer, -1 for put/receiver
    """
    d = (f - K) / (sigma_N * math.sqrt(T))
    return omega * (f - K) * norm.cdf(omega*d) + sigma_N * math.sqrt(T) * norm.pdf(d)

def bachelier_vega(omega, f, K, T, sigma_N, df_data):
    """
    Computes the corrected Bachelier vega incorporating the discount factor.
    """

    d = omega * (f - K) / (sigma_N * math.sqrt(T))
    closest_maturity_index = (df_data["maturity in years"] - T).abs().idxmin()
    discount_factor = df_data.loc[closest_maturity_index, "discount factors OIS"]
    return discount_factor * math.sqrt(T) * norm.pdf(d)

def find_implied_vol_newton(omega, f, K, T, market_price, df_data, initial_guess=0.05, tol=1e-8, max_iter=100):
    """
    Finds the implied volatility using the Newton-Raphson method.
    """
    sigma = initial_guess
    for _ in range(max_iter):
        price = bachelier_swaption_price(omega, f, K, T, sigma)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = bachelier_vega(omega, f, K, T, sigma, df_data)
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
corrected_inputs_df["   Bachelier mkt-implied norm vol "] = np.nan
corrected_inputs_df["shifted Bachelier vega"] = np.nan

# Loop over rows to compute implied volatilities and vegas
for index, row in corrected_inputs_df.iterrows():
    try:
        # Extract inputs
        omega = row["omega"]
        f = row["forward swap rate"]
        K = row["strikes "]  # Ensure correct column name
        T = row["expiry"]
        market_price = row["forward premium in dec."]

        # Compute implied volatility
        implied_vol = find_implied_vol_newton(omega, f, K, T, market_price, discount_factors_df)
        corrected_inputs_df.loc[index, "  Bachelier mkt-implied norm vol"] = implied_vol

        # Compute Vega
        vega = bachelier_vega(omega, f, K, T, implied_vol, discount_factors_df)
        corrected_inputs_df.loc[index, "Bachelier vega"] = vega

    except ValueError as e:
        corrected_inputs_df.loc[index, "  Bachelier mkt-implied norm vol"] = None
        corrected_inputs_df.loc[index, "Bachelier vega"] = None
        print(f"Row {index}: Error computing values: {e}")

# Save results to a new Excel file
output_file = "./advanced interest rate project/implied_volatilities_with_vegas.xlsx"
corrected_inputs_df.to_excel(output_file, index=False)
print(f"Implied volatilities and vegas saved to {output_file}")
