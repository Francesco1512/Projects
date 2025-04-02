import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Replace these with your actual computed d1 values ===
d1_cap_shift_03 = 0.2447
d1_floor_shift_03 = 0.612
d1_cap_shift_005 = 1.3406
d1_floor_shift_005 = 13.638
# =========================================================

# Create a range of x-values for plotting the standard normal PDF
x = np.linspace(-5, 5, 400)
y = norm.pdf(x)

# Plot the standard normal PDF
plt.plot(x, y, label="Standard Normal PDF")

# Add vertical lines for shift=0.03 (use one color)
plt.axvline(d1_cap_shift_03, label="Cap d1 (shift=0.03)", color="blue")
plt.axvline(d1_floor_shift_03, label="Floor d1 (shift=0.03)", color="violet")

# Add vertical lines for shift=0.005 (use another color)
plt.axvline(d1_cap_shift_005, label="Cap d1 (shift=0.005)", color="red")
plt.axvline(d1_floor_shift_005, label="Floor d1 (shift=0.005)", color="orange")

# Add legend and labels
plt.legend()
plt.xlabel("d1")
plt.ylabel("Density")
plt.title("Standard Normal PDF for Caps and Floors")

plt.show()
