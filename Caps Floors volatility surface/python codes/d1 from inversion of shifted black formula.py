import numpy as np
import pandas as pd
from math import sqrt, exp, log
from scipy.stats import norm
from scipy.optimize import brentq

def shifted_black_option_price(forward, strike, shift, vol, time_to_expiry, w):
    """
    Shifted Black formula for a single caplet (w=+1) or floorlet (w=-1).

    Parameters:
    -----------
    forward : float
        Forward rate F.
    strike : float
        Strike rate K.
    shift : float
        Shift S to avoid negative rates (F+S, K+S must be > 0).
    vol : float
        Implied volatility (lognormal).
    time_to_expiry : float
        Time to expiry in years.
    w : int
        +1 => caplet (call), -1 => floorlet (put).

    Returns:
    --------
    float
        The *undiscounted* caplet/floorlet price under shifted Black.
    """
    if vol < 1e-14 or time_to_expiry < 1e-14:
        # Degenerate case => approximate by intrinsic
        intrinsic = max(w * ((forward + shift) - (strike + shift)), 0.0)
        return intrinsic

    # d1 and d2
    numerator = log((forward + shift) / (strike + shift)) + 0.5 * vol**2 * time_to_expiry
    denom = vol * sqrt(time_to_expiry)
    d1 = numerator / denom
    d2 = d1 - vol * sqrt(time_to_expiry)

    # For a caplet (call), price = (F+S)*Phi(d1) - (K+S)*Phi(d2).
    # For a floorlet (put), price = (K+S)*Phi(-d2) - (F+S)*Phi(-d1).
    if w > 0:  # caplet
        price = (forward + shift) * norm.cdf(d1) - (strike + shift) * norm.cdf(d2)
    else:      # floorlet
        price = (strike + shift) * norm.cdf(-d2) - (forward + shift) * norm.cdf(-d1)

    return price
def price_cap_floor_shifted_black(discount_curve, forward_curve,
                                  shift, maturity_Tn, strike, w, vol):
    """
    Computes the SHIFTED BLACK theoretical price of a cap/floor with final maturity Tn,
    skipping the first caplet/floorlet (i.e. summation from i=2..n),
    under the rule:
      - If maturity_Tn <= 2 years => use 3M data => i=2..8, tau=0.25
      - Otherwise => use 6M data => i=2..60, tau=0.5

    For each caplet/floorlet:
      * We compute the forward F_i from forward_curve(i, is_3m).
      * We discount by discount_curve(T_i).
      * Multiply the SHIFTED BLACK (undiscounted) option price by discount * tau.

    Returns:
    --------
    float
        The full cap/floor price.
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

        # Stop summation if T_i goes beyond the product maturity
        if T_i > maturity_Tn:
            break

        # Discount factor
        Pd_i = discount_curve(T_i)

        # SHIFTED BLACK (undiscounted) payoff:
        caplet_price = shifted_black_option_price(F_i, strike, shift, vol, T_i, w)

        # Add discounted payoff
        total_price += Pd_i * tau_i * caplet_price

    return total_price
def implied_vol_cap_floor_shifted_black(mkt_price,
                                        discount_curve, forward_curve,
                                        maturity_Tn, strike, w,
                                        shift=0.005,
                                        vol_lower=1e-9, vol_upper=5.0, tol=1e-8):
    """
    Numerically invert the SHIFTED BLACK formula for a cap/floor with final maturity Tn,
    given a market price. Uses Brent's method to find vol in [vol_lower, vol_upper].

    Prints the final or last tried d1 if no solution is found.

    Returns:
    --------
    float
        The implied volatility under shifted Black, or np.nan if no solution was found.
    """
    # If the market price is extremely small, you can treat vol = 0 as the implied vol
    if mkt_price < 1e-14:
        return 0.0

    # We'll keep track of the last sigma tested by Brent
    last_sigma = [np.nan]

    def objective(vol):
        last_sigma[0] = vol  # store the current guess
        model_price = price_cap_floor_shifted_black(
            discount_curve, forward_curve, shift, maturity_Tn, strike, w, vol
        )
        return model_price - mkt_price

    try:
        vol_solution = brentq(objective, vol_lower, vol_upper, xtol=tol, rtol=tol, maxiter=500)
        # => We found a root => vol_solution is the implied vol

        # Once we have vol_solution, compute d1 with SHIFTED BLACK for the *final* caplet
        # or at least for an example maturity. It's ambiguous which T_i we might show d1 for
        # (since the "cap/floor" is a sum of many caplets), but let's show it for Tn.
        # We'll just treat forward = forward_curve(...) for Tn, ignoring some details.
        F_approx = forward_curve(int(2 if maturity_Tn<=2.0 else 2), maturity_Tn<=2.0)
        # If that doesn't exist, you might do something else or skip.
        if not np.isnan(F_approx) and F_approx is not None:
            # Use Tn for time_to_expiry in d1 formula:
            numerator = np.log((F_approx + shift) / (strike + shift)) + 0.5 * vol_solution**2 * maturity_Tn
            denom = vol_solution * np.sqrt(maturity_Tn)
            d1_final = numerator / denom if denom != 0 else np.nan
            print(f"[OK] SHIFTED BLACK vol = {vol_solution:.6f}, final d1 ~ {d1_final:.6f}")
        else:
            print(f"[OK] SHIFTED BLACK vol = {vol_solution:.6f}, (cannot compute final d1 for Tn={maturity_Tn})")

        return vol_solution

    except ValueError:
        # The solver never found a solution in [vol_lower, vol_upper].
        print(f"[FAIL] No implied volatility found for Tn={maturity_Tn}, strike={strike}, w={w} in bracket!")
        last_guess = last_sigma[0]
        if not np.isnan(last_guess):
            # Show a "pseudo" d1 from that last guess so we can see how extreme it was
            # We'll do the same approach as above for an approximate forward/time.
            F_approx = forward_curve(int(2 if maturity_Tn<=2.0 else 2), maturity_Tn<=2.0)
            if not np.isnan(F_approx) and F_approx is not None:
                numerator = np.log((F_approx + shift) / (strike + shift)) + 0.5 * last_guess**2 * maturity_Tn
                denom = last_guess * np.sqrt(maturity_Tn)
                d1_pseudo = numerator / denom if denom != 0 else np.nan
                print(f"    Last sigma tried = {last_guess:.6f}, d1 ~ {d1_pseudo:.6f}")
        return np.nan
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

    # Build forward dictionaries (same as your original code)
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

    SHIFT_VALUE = 0.005  # <-- Hardcoded here; you can make it dynamic from 'inputs_df' if desired.

    results = []
    for idx, row in inputs_df.iterrows():
        maturity = row["maturity"]  # T_n
        w = row["omega"]            # +1 => cap, -1 => floor
        strike = row["strike"]
        spot_prem = row["spot premia in dec."]

        if spot_prem == 0.0:
            continue

        # SHIFTED BLACK implied vol
        vol_solution = implied_vol_cap_floor_shifted_black(
            mkt_price=spot_prem,
            discount_curve=discount_curve,
            forward_curve=forward_curve,
            maturity_Tn=maturity,
            strike=strike,
            w=w,
            shift=0.005,        # <--- pass shift here
            vol_lower=1e-9,
            vol_upper=5.0,
            tol=1e-8
        )

        results.append({
            "maturity": maturity,
            "omega": w,
            "strike": strike,
            "market_premium": spot_prem,
            "shift": 0.005,
            "implied_lognormal_vol": vol_solution
        })

    results_df = pd.DataFrame(results)
    print("\n--- Full Results (Shifted Black, one row per maturity-strike-omega) ---")
    print(results_df)

    # Pivot: rows: maturity, columns: strike, values: implied vol
    matrix_df = results_df.pivot_table(
        index="maturity",
        columns="strike",
        values="implied_lognormal_vol"
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
