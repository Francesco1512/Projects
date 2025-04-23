import pandas as pd
from scipy.interpolate import PchipInterpolator

# Load the Excel files
discount_factors_path = "./advanced interest rate project/task 1/discount factors.xlsx"  # Update with your actual file path
corrected_inputs_path = "./advanced interest rate project/corrected inputs.xlsx"  # Update with your actual file path

xls_discount = pd.ExcelFile(discount_factors_path)
xls_corrected = pd.ExcelFile(corrected_inputs_path)

# Load the relevant sheets
df_discount = pd.read_excel(xls_discount, sheet_name="euribor 6m discount factor curv")
df_fwd_ibor = pd.read_excel(xls_corrected, sheet_name="Fwd ibor rate")

# Extract relevant columns from the discount factors data
time_to_maturity = df_discount["time to maturity in year"].dropna().astype(float).values
discount_factors = df_discount["discount factors euribor6M"].dropna().astype(float).values

# Create interpolation function for discount factors
interp_function = PchipInterpolator(time_to_maturity, discount_factors)

# Extract expiry (T_i-1) and tenor (T_i) from the Fwd ibor rate sheet
expiry = df_fwd_ibor["expiry"].astype(float).values  # T_i-1
tenor = df_fwd_ibor["tenor"].astype(float).values    # T_i

# Compute interpolated discount factors for expiry and tenor
P_Ti_minus_1 = interp_function(expiry)
P_Ti = interp_function(tenor)

# Compute new forward IBOR rate using the formula: F = (1/0.5) * ((P(0;T_i-1)/P(0;T_i)) - 1)
new_fwd_ibor_rate = (1 / 0.5) * ((P_Ti_minus_1 / P_Ti) - 1)

# Add the new column to the dataframe
df_fwd_ibor["new fwd ibor rate from euribor6m"] = new_fwd_ibor_rate

# Save the updated dataframe to a new Excel file
updated_file_path = "./advanced interest rate project/updated_corrected_inputs.xlsx"
df_fwd_ibor.to_excel(updated_file_path, sheet_name="Fwd ibor rate", index=False)

print(f"Updated file saved as {updated_file_path}")
