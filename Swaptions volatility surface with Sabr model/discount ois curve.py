from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

#-----------------INTERPOLATION OF THE ANNUALIZED DISCOUNT RATES CURVE TO MATHC THE TENOR OF THE IRS--------

# Load the Excel file and the specific sheet
file_path = "./advanced interest rate project 2/inputs 2.xlsx"  # Replace with your file's path
output_file = "./advanced interest rate project 2/interpolated_discount_rates.xlsx"  # Output file name

# Load the worksheet and clean the data
data = pd.read_excel(file_path, sheet_name="Sheet1")

# Extract relevant columns
data_cleaned = data.iloc[2:, [1, 0]]  # Select 'maturity in years' and 'discount rate annualized'
data_cleaned.columns = ["maturity_in_years", "discount_rate_annualized"]  # Rename columns
data_cleaned = data_cleaned.dropna()  # Remove rows with missing values

# Convert to numeric values
data_cleaned["maturity_in_years"] = pd.to_numeric(data_cleaned["maturity_in_years"], errors='coerce')
data_cleaned["discount_rate_annualized"] = pd.to_numeric(data_cleaned["discount_rate_annualized"], errors='coerce')
data_cleaned = data_cleaned.dropna()  # Drop invalid rows

# Define target maturities: 0 to 60 (with quarters)
target_maturities = np.concatenate([np.arange(0, 61, 1) + offset for offset in [0, 0.25, 0.5, 0.75]])

# Perform monotonic cubic spline interpolation (PCHIP)
interpolator = PchipInterpolator(data_cleaned["maturity_in_years"], data_cleaned["discount_rate_annualized"])
interpolated_values = interpolator(target_maturities)

# Combine original and interpolated data into a single DataFrame
original_data = data_cleaned.copy()
original_data["type"] = "original"

interpolated_data = pd.DataFrame({
    "maturity_in_years": target_maturities,
    "discount_rate_annualized": interpolated_values,
    "type": "interpolated"
})

# Combine both datasets
final_data = pd.concat([original_data, interpolated_data]).sort_values("maturity_in_years").reset_index(drop=True)

# Save the final data to a new Excel file
final_data.to_excel(output_file, index=False)

#--------------------------------COMPUTING THE DISCOUNT FACTOR CURVE FROM THE ANN. DISCOUNT RATES----------------
# Load the new Excel file with interpolated data
file_path = "./advanced interest rate project 2/interpolated_discount_rates.xlsx"  # Replace with the correct file path
output_file = "./advanced interest rate project 2/updated_discount_factors.xlsx"  # Output file name

# Load the data
data = pd.read_excel(file_path)

# Calculate discount factors: P(0;T) = exp(-r(T) * T)
data["discount_factors"] = np.exp(-data["discount_rate_annualized"] * data["maturity_in_years"])

# Save the updated data to a new Excel file
data.to_excel(output_file, index=False)
print(f"Updated data with discount factors has been saved to {output_file}")

# Plot 1: Comparison of interpolated vs. original discount annualized rates
#original_data = data[data["type"] == "original"]
#interpolated_data = data[data["type"] == "interpolated"]

#plt.figure(figsize=(10, 6))
#plt.plot(original_data["maturity_in_years"], original_data["discount_rate_annualized"], label="Original Curve", marker='o')
#plt.plot(interpolated_data["maturity_in_years"], interpolated_data["discount_rate_annualized"], label="Interpolated Curve", linestyle='--')
#plt.title("Comparison of Original and Interpolated Discount Annualized Rates")
#plt.xlabel("Maturity (Years)")
#plt.ylabel("Annualized Discount Rates")
#plt.legend()
#plt.grid()
#plt.savefig("comparison_discount_rates.png")
#print("Plot 1: Comparison of original and interpolated discount annualized rates saved as './advanced interest rate project/comparison_discount_rates.png'.")

# Plot 2: Discount factors curve
plt.figure(figsize=(10, 6))
plt.plot(data["maturity_in_years"], data["discount_factors"], label="Discount Factors Curve", color='g')
plt.title(" Euribor 3M Discount Factors Curve")
plt.xlabel("Maturity (Years)")
plt.ylabel("Discount Factors")
plt.legend()
plt.grid()
plt.savefig("discount_factors_curve.png")
print("Plot 2: Discount factors curve saved as './advanced interest rate project 2/discount_factors_curve.png'.")

plt.show()

#-------

