import pandas as pd
from scipy.interpolate import PchipInterpolator

# Load the Excel files
discount_factors_path = "./advanced interest rate project/task 1/discount factors.xlsx"
corrected_inputs_path = "./advanced interest rate project/updated inputs.xlsx"

xls_discount = pd.ExcelFile(discount_factors_path)
xls_corrected = pd.ExcelFile(corrected_inputs_path)

# Load relevant sheets
df_discount = pd.read_excel(xls_discount, sheet_name="eur ois discount factor curve")
df_fwd_ibor = pd.read_excel(xls_corrected, sheet_name="Fwd ibor rate")
df_inputs = pd.read_excel(xls_corrected, sheet_name="Sheet1")

# Extract relevant columns
time_to_maturity = df_discount["maturity in years"].dropna().astype(float).values
discount_factors = df_discount["discount factors OIS"].dropna().astype(float).values

# Create interpolation function for discount factors
interp_function = PchipInterpolator(time_to_maturity, discount_factors)

# Extract swap start and end times
swap_start = df_inputs["expiry"].min()
swap_end = df_inputs["tenor"].max()

# Generate all semi-annual payment dates starting at t=1.5 if swap starts at t=1.0
payment_dates = [swap_start + 0.5 * i for i in range(1, int((swap_end - swap_start) / 0.5) + 1)]

# Extract forward IBOR rate dictionary based on expiry and tenor
fwd_ibor_dict = {(row["expiry"], row["tenor"]): row["forward IBOR rate from EURIBOR 6M"] for _, row in df_fwd_ibor.iterrows()}

# Compute the sum of (forward rate * discount factor * 0.5) for each expiry and tenor pair
irs_values = []
for index, row in df_inputs.iterrows():
    swap_maturity = row["expiry"] + row["tenor"]  # Compute actual swap end time
    exp, ten = row["expiry"], swap_maturity
    applicable_rates = [fwd_ibor_dict[(e, t)] * interp_function(t) * 0.5 
                        for e, t in fwd_ibor_dict.keys() if e == exp and t == swap_maturity]
    irs_values.append(sum(applicable_rates))

# Store results in the "Inputs" sheet
df_inputs["numerator IRS rate"] = irs_values

# Save the updated DataFrame back to the same Excel file
updated_corrected_inputs_path = "./advanced interest rate project/new_corrected_inputs.xlsx"
df_inputs.to_excel(updated_corrected_inputs_path, sheet_name="Inputs", index=False)

print(f"file saved. Results saved to '{updated_corrected_inputs_path}'")



