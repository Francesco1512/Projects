import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Example: Compare "Brent" and "Approx" values for varying alpha, lambda
#          over 3 panels: ITM, ATM, OTM.
#          This skeleton shows how you might produce a figure similar
#          to your reference image.
# ---------------------------------------------------------------------

##############################
# 1) Setup / Parameter Grids #
##############################

# For simplicity, assume we have:
#   forward F
#   range of "relative moneyness": (K - F)
#   arrays of alpha and lambda to loop over
F = 100.0
rel_money = np.linspace(-0.05, 0.05, 50)  # e.g. from -5% to +5% around F
alphas    = [0.01, 0.02, 0.03]
lambdas   = [0.0, 0.02]   # or more values if you like
expiry    = 5            # just an example (tenor=5)
beta      = 0.5          # example
rho       = 0.0          # example
vol_of_vol= 0.4          # example

# Suppose "Brent_sabr_price(K, alpha, lambda, ...)" and
#         "Approx_sabr_price(K, alpha, lambda, ...)"
# are functions you’ve written that compute the two results
# (they could be *volatilities* or *option prices*—your choice).
# Here we’ll just mock them:
def Brent_sabr_value(K, alpha, lam):
    # e.g. complicated PDE or Monte Carlo
    return alpha - 0.2*lam + 0.03*(K-F)**2  # TOTALLY FAKE for illustration

def Approx_sabr_value(K, alpha, lam):
    # e.g. Hagan’s asymptotic formula
    return alpha - 0.1*lam + 0.025*(K-F)**2 # TOTALLY FAKE for illustration

# For convenience, we’ll define a helper that returns
# the difference (Brent - Approx) for a given K, alpha, lam:
def difference_sabr(K, alpha, lam):
    return Brent_sabr_value(K, alpha, lam) - Approx_sabr_value(K, alpha, lam)


########################################
# 2) Generate Data and Create Subplots #
########################################

# We want 3 subplots horizontally:
#   left  : In-the-money (K < F => negative rel. money)
#   middle: At-the-money region (near K=F)
#   right : Out-of-the-money (K > F => positive rel. money)
#
# We'll slice rel_money into 3 regions.  In practice you can
# choose how "narrow" the ATM region is, etc.
n = len(rel_money)
idx_left  = np.where(rel_money < -0.005)  # e.g. below -0.5% = ITM
idx_mid   = np.where((rel_money >= -0.005) & (rel_money <= 0.005))
idx_right = np.where(rel_money > 0.005)

fig, axes = plt.subplots(1, 3, figsize=(15,6), sharey=True)
titles = ["In-the-money", "ATM", "Out-of-the-money"]

# Each curve in the plot will correspond to a particular
# combination of alpha and lambda.  We'll color by alpha, line style by lambda.
# (Feel free to adapt to your aesthetic preference.)

# We’ll pre‐define some color/linestyle sequences:
color_cycle = ["purple", "teal", "gold"]  # for alpha
style_cycle = ["-", "--", "-."]           # for lambda (just an example)
# If you have more than 3 alphas or lambdas, expand these lists.

# Make sure we handle case: len(alphas) * len(lambdas) might exceed our style combos.
# We'll just do an index:
def color_for(i): return color_cycle[i % len(color_cycle)]
def style_for(j): return style_cycle[j % len(style_cycle)]


for ax_idx, ax in enumerate(axes):
    # Choose which subset of rel_money we use:
    if ax_idx == 0:
        sub_idx = idx_left
    elif ax_idx == 1:
        sub_idx = idx_mid
    else:
        sub_idx = idx_right

    # Get the actual array of relative moneyness for this subplot
    rm_sub = rel_money[sub_idx]

    # For each alpha-lambda combination, compute the difference array
    for i, alpha_ in enumerate(alphas):
        for j, lam_ in enumerate(lambdas):
            diff_vals = []
            for rm in rm_sub:
                K = F + rm
                diff_vals.append( difference_sabr(K, alpha_, lam_) )

            diff_vals = np.array(diff_vals)
            # Build a label that indicates (alpha, lambda)
            label_str = f"alpha={alpha_}, lam={lam_}"
            ax.plot(rm_sub,
                    diff_vals,
                    linestyle=style_for(j),
                    color=color_for(i),
                    label=label_str)

    ax.set_title(titles[ax_idx])
    ax.set_xlabel("Relative Moneyness (K - F)")
    ax.grid(True)

# The leftmost axis can have the shared y‐axis label
axes[0].set_ylabel("Difference (Brent - Approx)")

# Show a combined legend in the middle or wherever you like:
# One way is to gather handles/labels from the last axis or so:
handles, labels = axes[0].get_legend_handles_labels()
# But we have repeats. So let's do a dictionary trick to remove duplicates:
unique = dict(zip(labels, handles))
fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3)

fig.suptitle(f"Difference (Brent - Approx) vs. Relative Moneyness\nVarying alpha, lambda (expiry={expiry})")
plt.tight_layout()
plt.show()

