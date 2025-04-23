import pandas as pd
import numpy as np

# Load the necessary files
discount_factors_file = "./advanced interest rate project/task 1/discount factors.xlsx"  # Change to your file path
corrected_inputs_file = "./advanced interest rate project/corrected inputs.xlsx"  # Change to your file path

# Load discount factors data
df_discount_factors = pd.read_excel(discount_factors_file, sheet_name="Sheet1")

# Load corrected inputs data
df_corrected_inputs_sheet1 = pd.read_excel(corrected_inputs_file, sheet_name="Sheet1")

# Extract unique expiry values
unique_expiries = np.sort(df_corrected_inputs_sheet1["expiry"].unique())

# Function to find the closest discount factor based on maturity
def get_closest_discount_factor(maturity, df_discount_factors):
    closest_index = (df_discount_factors["maturity in years"] - maturity).abs().idxmin()
    return df_discount_factors.loc[closest_index, "discount factors"]

# Compute forward IBOR rates for different tenor values starting from each expiry
new_data = []

for expiry in unique_expiries:
    tenor = expiry + 0.5  # Start from expiry + 0.5
    while tenor <= 30.75:  # Up to 30.75 years
        T_i_minus_1 = tenor - 0.5  # Previous maturity (semiannual step back)
        T_i = tenor  # Current maturity

        # Get the closest discount factors
        P_x_T_i_minus_1 = get_closest_discount_factor(T_i_minus_1, df_discount_factors)
        P_x_T_i = get_closest_discount_factor(T_i, df_discount_factors)

        # Compute the forward rate using the given formula
        if (T_i - T_i_minus_1) > 0 and P_x_T_i > 0:
            F_x_i = (1 / (T_i - T_i_minus_1)) * ((P_x_T_i_minus_1 / P_x_T_i) - 1)
        else:
            F_x_i = np.nan  # Avoid division by zero

        new_data.append({"expiry": expiry, "tenor": tenor, "forward_IBOR_rate": F_x_i})

        # Increment tenor by 0.5 for the next step
        tenor += 0.5

# Convert results into a DataFrame and sort by expiry and tenor
df_new_data = pd.DataFrame(new_data).sort_values(by=["expiry", "tenor"])

# Save to Excel file
output_file = "bootstrapped forward Ibor rates curve.xlsx"
df_new_data.to_excel(output_file, index=False)

print(f"Computation completed. The results are saved in {output_file}")
