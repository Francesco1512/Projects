import numpy as np
import pandas as pd
from math import sqrt, log
from scipy.stats import norm

##############################################
# 1) Define the shifted Black option pricing function
##############################################
def shifted_black_option_price(forward, strike, vol, time_to_expiry, w, shift=0.06):
    """
    Computes the shifted Black price for a single caplet/floorlet.
    
    The forward and strike are shifted by 'shift' to ensure positivity.
    w = +1 for a call (caplet) and w = -1 for a put (floorlet).
    """
    F_shifted = forward + shift
    K_shifted = strike + shift

    if vol < 1e-14:
        intrinsic = max(w * (F_shifted - K_shifted), 0)
        return intrinsic

    try:
        d1 = (np.log(F_shifted / K_shifted) + 0.5 * vol**2 * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
    except ValueError:
        return 0.0
    d2 = d1 - vol * np.sqrt(time_to_expiry)

    if w == +1:
        price = F_shifted * norm.cdf(d1) - K_shifted * norm.cdf(d2)
    else:
        price = K_shifted * norm.cdf(-d2) - F_shifted * norm.cdf(-d1)
    return price

##############################################
# 2) Define cap/floor price as sum of caplets/floorlets
##############################################
def price_cap_floor(discount_curve, forward_curve, maturity_Tn, strike, w, vol, shift=0.06):
    """
    Computes the theoretical price of a cap/floor by summing the prices of individual caplets/floorlets.
    
    Uses 3M forwards (i=2..8) if maturity_Tn <= 2 years, otherwise 6M forwards (i=2..60).
    """
    is_3m = (maturity_Tn <= 2.0)
    i_max = 8 if is_3m else 60
    total_price = 0.0

    for i in range(2, i_max + 1):
        F_i = forward_curve(i, is_3m)
        if np.isnan(F_i):
            break
        if is_3m:
            T_i = i * 0.25
            tau_i = 0.25
        else:
            T_i = i * 0.5
            tau_i = 0.5
        if T_i > maturity_Tn:
            break
        Pd_i = discount_curve(T_i)
        caplet_price = shifted_black_option_price(F_i, strike, vol, T_i, w, shift=shift)
        total_price += Pd_i * tau_i * caplet_price
    return total_price

##############################################
# 3) Compute the vega for one caplet using shifted Black
##############################################
def shifted_black_vega(forward, strike, vol, time_to_expiry, w, shift=0.5):
    """
    Computes the sensitivity (vega) of a caplet/floorlet price to volatility,
    under the shifted Black model.
    """
    F_shifted = forward + shift
    K_shifted = strike + shift
    if vol < 1e-14:
        return 0.0
    try:
        d1 = (np.log(F_shifted / K_shifted) + 0.5 * vol**2 * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
    except ValueError:
        return 0.0
    # For Black, vega is typically F_shifted * sqrt(T) * norm.pdf(d1)
    # This holds for both calls and puts (since norm.pdf is symmetric).
    return F_shifted * norm.pdf(d1) * np.sqrt(time_to_expiry)

##############################################
# 4) Sum caplet vegas to get cap/floor vega
##############################################
def price_cap_floor_vega(discount_curve, forward_curve, maturity_Tn, strike, w, vol, shift=0.06):
    is_3m = (maturity_Tn <= 2.0)
    i_max = 8 if is_3m else 60
    total_vega = 0.0
    for i in range(2, i_max + 1):
        F_i = forward_curve(i, is_3m)
        if np.isnan(F_i):
            break
        if is_3m:
            T_i = i * 0.25
            tau_i = 0.25
        else:
            T_i = i * 0.5
            tau_i = 0.5
        if T_i > maturity_Tn:
            break
        Pd_i = discount_curve(T_i)
        vega_caplet = shifted_black_vega(F_i, strike, vol, T_i, w, shift=shift)
        total_vega += Pd_i * tau_i * vega_caplet
    return total_vega

##############################################
# 5) Newton–Raphson inversion for implied vol
##############################################
def newton_implied_vol_cap_floor(mkt_price, discount_curve, forward_curve, maturity_Tn, strike, w, shift=0.06):
    """
    Inverts the shifted Black price to find the implied lognormal volatility using Newton–Raphson.
    
    If the market premium is near zero, returns 0.0. If convergence fails, returns np.nan.
    """
    if mkt_price < 1e-14:
        return 0.0

    vol_guess = 0.1  # initial guess (you might adjust this based on ATM approximations)
    tol = 1e-8
    max_iter = 100

    for _ in range(max_iter):
        price = price_cap_floor(discount_curve, forward_curve, maturity_Tn, strike, w, vol_guess, shift=shift)
        diff = price - mkt_price
        if abs(diff) < tol:
            return vol_guess
        vega = price_cap_floor_vega(discount_curve, forward_curve, maturity_Tn, strike, w, vol_guess, shift=shift)
        if abs(vega) < 1e-12:
            # Avoid division by nearly zero derivative.
            break
        vol_guess = vol_guess - diff / vega
    return np.nan

##############################################
# 6) Main script: read data, compute implied vols, pivot, and save to Excel
##############################################
def main():
    file_name = "./advanced interest rate project 2/inputs 2.xlsx"
    discount_df = pd.read_excel(file_name, sheet_name="discount curve ois")
    fwd_df = pd.read_excel(file_name, sheet_name="fwd rates euribor")
    inputs_df = pd.read_excel(file_name, sheet_name="inputs")

    # Build discount curve interpolation.
    discount_maturities = discount_df["Maturity"].values
    discount_factors = discount_df["Discount factor ois"].values
    def discount_curve(T):
        if T <= discount_maturities[0]:
            return discount_factors[0]
        if T >= discount_maturities[-1]:
            return discount_factors[-1]
        return np.interp(T, discount_maturities, discount_factors)

    # Build forward dictionaries.
    fwd_6m_dict = {}
    fwd_3m_dict = {}
    for idx, row in fwd_df.iterrows():
        i_6m = row["F_x,i for i=2,...,60"]
        f_6m = row["Forward for EURIBOR 6M"]
        i_3m = row["F_x,i for i=2,...,8"]
        f_3m = row["Forward for EURIBOR 3M"]
        if not pd.isna(i_6m) and not pd.isna(f_6m):
            fwd_6m_dict[int(i_6m)] = f_6m
        if not pd.isna(i_3m) and not pd.isna(f_3m):
            fwd_3m_dict[int(i_3m)] = f_3m

    def forward_curve(i, is_3m):
        return fwd_3m_dict.get(i, np.nan) if is_3m else fwd_6m_dict.get(i, np.nan)

    # Collect implied vol results using Newton–Raphson inversion.
    results = []
    for idx, row in inputs_df.iterrows():
        maturity = row["maturity"]  # T_n
        w = row["omega"]            # +1 (cap) or -1 (floor)
        strike = row["strike"]
        spot_prem = row["spot premia in dec."]

        if spot_prem == 0.0:
            continue

        vol = newton_implied_vol_cap_floor(spot_prem, discount_curve, forward_curve, maturity, strike, w, shift=0.06)
        results.append({
            "maturity": maturity,
            "omega": w,
            "strike": strike,
            "market_premium": spot_prem,
            "implied_ln_vol": vol
        })

    results_df = pd.DataFrame(results)
    print("\n--- Full Results (Shifted Black via Newton–Raphson) ---")
    print(results_df)

    matrix_df = results_df.pivot_table(index="maturity", columns="strike", values="implied_ln_vol")
    print("\n--- Shifted Black Implied Vol Surface (matrix) ---")
    print(matrix_df)

    output_file = "./advanced interest rate project 2/shifted_black_implied_vols_output_newton.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name="LongFormatResults", index=False)
        matrix_df.to_excel(writer, sheet_name="MatrixResults")

    print(f"\nSaved results to '{output_file}'.")

if __name__ == "__main__":
    main()
