import pandas as pd

# Load the input files
final_inputs_file = "./advanced interest rate project/final inputs for formula.xlsx"
discount_factors_file = "./advanced interest rate project/discount factors.xlsx"

# Read the relevant worksheets and columns
sheet3_data = pd.read_excel(final_inputs_file, sheet_name="Sheet3")
discount_factors_data = pd.read_excel(discount_factors_file)

# Ensure column names are stripped of extra spaces
sheet3_data.columns = sheet3_data.columns.str.strip()
discount_factors_data.columns = discount_factors_data.columns.str.strip()

# Extract relevant columns
expiry_column = sheet3_data["expiry"]
maturity_column = discount_factors_data["maturity_in_years"]
discount_factors_column = discount_factors_data["discount_factors"]

# Match expiry with the closest maturity and extract corresponding discount factors
def find_closest(expiry, maturities, discount_factors):
    differences = abs(maturities - expiry)
    closest_index = differences.idxmin()
    return discount_factors[closest_index]

sheet3_data["matched_discount_factors"] = expiry_column.apply(
    lambda expiry: find_closest(expiry, maturity_column, discount_factors_column)
)

# Save the updated data to a new Excel file
output_file = "./advanced interest rate project/updated_final_inputs_with_closest_discount_factors.xlsx"
sheet3_data.to_excel(output_file, index=False)

output_file
