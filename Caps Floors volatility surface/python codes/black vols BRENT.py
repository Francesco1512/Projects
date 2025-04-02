import numpy as np
import pandas as pd
from math import sqrt, log
from scipy.stats import norm
from scipy.optimize import brentq

######################################################################
# 1) Define the shifted Black formula for a single caplet/floorlet
######################################################################
def shifted_black_option_price(forward, strike, vol, time_to_expiry, w, shift=0.0075):
    """
    Shifted Black formula for a single caplet/floorlet (European call/put on the forward).

    We assume a shift of 0.03 is added to both forward and strike, i.e. (F + 0.03) and (K + 0.03).
    Then we apply the usual Black lognormal formula:

      d1 = [ln((F+S)/(K+S)) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
      d2 = d1 - sigma * sqrt(T)

      call_price = (F+S)*Phi(d1) - (K+S)*Phi(d2)
      put_price  = (K+S)*Phi(-d2) - (F+S)*Phi(-d1)

    Where w=+1 => caplet (call), w=-1 => floorlet (put).

    If vol is extremely small, we approximate payoff by intrinsic (shift cancels out).

    Parameters
    ----------
    forward : float
        The forward rate F.
    strike : float
        The strike K.
    vol : float
        Black (lognormal) volatility (sigma).
    time_to_expiry : float
        Time to expiry in years for this caplet.
    w : int
        +1 if it is a caplet (call), -1 if it is a floorlet (put).
    shift : float
        The shift to add to forward and strike (default = 0.03).

    Returns
    -------
    float
        Price of the shifted-Black caplet/floorlet (not discounted).
    """
    F_shifted = forward + shift
    K_shifted = strike + shift

    # If shifted forward or strike is negative, the formula wouldn't make sense,
    # but in your case shift=0.03 is large enough to keep them positive.
    # If vol is extremely small, approximate by intrinsic
    if vol < 1e-14:
        intrinsic = max(w * (F_shifted - K_shifted), 0)
        return intrinsic

    # Lognormal Black formula
    try:
        d1 = (np.log(F_shifted / K_shifted) + 0.5 * vol**2 * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
    except ValueError:
        # If something is invalid (e.g. negative inside log), return 0 or handle gracefully
        return 0.0
    d2 = d1 - vol * np.sqrt(time_to_expiry)

    if w == +1:
        # call
        price = F_shifted * norm.cdf(d1) - K_shifted * norm.cdf(d2)
    else:
        # put
        price = K_shifted * norm.cdf(-d2) - F_shifted * norm.cdf(-d1)

    return price

######################################################################
# 2) Summation for the cap/floor: discounted sum of caplets/floorlets
######################################################################
def price_cap_floor(discount_curve, forward_curve, maturity_Tn, strike, w, vol):
    """
    Computes the SHIFTED BLACK theoretical price of a cap/floor with final maturity Tn,
    skipping the first caplet/floorlet (i.e. summation from i=2..n),
    under the rule:
      - If maturity_Tn <= 2 years => use 3M forward data (i=2..8), tau=0.25
      - Otherwise => use 6M forward data (i=2..60), tau=0.5.

    Then for each caplet, we use the function 'shifted_black_option_price'
    to compute the (undiscounted) option price, multiply by discount factor * tau, and sum.
    """
    is_3m = (maturity_Tn <= 2.0)
    i_max = 8 if is_3m else 60

    total_price = 0.0
    for i in range(2, i_max + 1):
        F_i = forward_curve(i, is_3m)
        if np.isnan(F_i):
            break  # no more forward data

        if is_3m:
            T_i = i * 0.25
            tau_i = 0.25
        else:
            T_i = i * 0.5
            tau_i = 0.5

        if T_i > maturity_Tn:
            break

        # Discount factor
        Pd_i = discount_curve(T_i)

        # SHIFTED BLACK formula for caplet
        caplet_price = shifted_black_option_price(F_i, strike, vol, T_i, w, shift=0.0075)

        # Add discounted payoff
        total_price += Pd_i * tau_i * caplet_price

    return total_price

######################################################################
# 3) Invert the SHIFTED BLACK formula to get implied lognormal vol
######################################################################

def implied_vol_cap_floor(mkt_price, discount_curve, forward_curve, maturity_Tn, strike, w):
    """
    Numerically invert the SHIFTED BLACK price to find the implied lognormal volatility.
    (Vol bracket is [1e-9, 5.0]. If no sign change, returns np.nan.)
    """
    if mkt_price < 1e-14:
        return 0.0

    def objective(vol):
        model_price = price_cap_floor(discount_curve, forward_curve, maturity_Tn, strike, w, vol)
        return model_price - mkt_price

    try:
        return brentq(objective, 1e-9, 5.0, maxiter=500)
    except ValueError:
        return np.nan

######################################################################
# 4) Main script: read data, compute SHIFTED BLACK implied vols, pivot
######################################################################
def main():
    file_name = "./advanced interest rate project 2/inputs 2.xlsx"
    discount_df = pd.read_excel(file_name, sheet_name="discount curve ois")
    fwd_df = pd.read_excel(file_name, sheet_name="fwd rates euribor")
    inputs_df = pd.read_excel(file_name, sheet_name="inputs")

    # Build discount curve interpolation
    discount_maturities = discount_df["Maturity"].values
    discount_factors = discount_df["Discount factor ois"].values

    def discount_curve(T):
        if T <= discount_maturities[0]:
            return discount_factors[0]
        if T >= discount_maturities[-1]:
            return discount_factors[-1]
        return np.interp(T, discount_maturities, discount_factors)

    # Build forward dictionaries
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

    # Collect implied vol results
    results = []
    for idx, row in inputs_df.iterrows():
        maturity = row["maturity"]  # T_n
        w = row["omega"]            # +1 (cap) or -1 (floor)
        strike = row["strike"]
        spot_prem = row["spot premia in dec."]

        # If skipping zero premium, keep this check:
        if spot_prem == 0.0:
            continue

        # Solve for SHIFTED BLACK implied vol
        vol = implied_vol_cap_floor(spot_prem, discount_curve, forward_curve, maturity, strike, w)

        results.append({
            "maturity": maturity,
            "omega": w,
            "strike": strike,
            "market_premium": spot_prem,
            "implied_ln_vol": vol  # now it's a lognormal vol under the shifted Black model
        })

    results_df = pd.DataFrame(results)
    print("\n--- Full Results (Shifted Black, one row per maturity-strike-omega) ---")
    print(results_df)

    # Create a pivot matrix => rows: maturity, columns: strike, values: implied lognormal vol
    matrix_df = results_df.pivot_table(
        index="maturity",
        columns="strike",
        values="implied_ln_vol"
    )

    print("\n--- Shifted Black Implied Vol Surface (matrix) ---")
    print(matrix_df)

    # Save to Excel
    output_file = "./advanced interest rate project 2/shifted_black_implied_vols_output.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name="LongFormatResults", index=False)
        matrix_df.to_excel(writer, sheet_name="MatrixResults")

    print(f"\nSaved results to '{output_file}'.")

if __name__ == "__main__":
    main()
