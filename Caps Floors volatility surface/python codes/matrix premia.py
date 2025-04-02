import pandas as pd

# Read the Excel file from Sheet2
df = pd.read_excel("./advanced interest rate project 2/inputs 2.xlsx", sheet_name="Sheet2")

# The assumed structure:
#   - Column 0: maturity
#   - Column 1: "STK" (ATM strike)
#   - Column 2: "ATM" (premia for the ATM strike)
#   - Columns 3 and onward: the column headers are the strike values and the cells contain the relative premia

# Prepare a list to collect the new rows
rows = []

# Process each row in the dataframe
for _, row in df.iterrows():
    # Extract the maturity, ATM strike, and its premia
    maturity = row.iloc[0]
    atm_strike = row["STK"]
    atm_premia = row["ATM"]
    
    # Append the ATM strike information
    rows.append({
        "maturity": maturity,
        "strike": atm_strike,
        "premia": atm_premia
    })
    
    # Now process the additional strikes (from the fourth column onward)
    for col in df.columns[3:]:
        strike = col  # the column header is the strike value
        premia = row[col]
        rows.append({
            "maturity": maturity,
            "strike": strike,
            "premia": premia
        })

# Convert the collected rows into a DataFrame
df_long = pd.DataFrame(rows)

# Save the transformed data to a new Excel file
df_long.to_excel("./advanced interest rate project 2/transformed_data.xlsx", index=False)

print("Transformation complete. The new file 'transformed_data.xlsx' has been created.")
