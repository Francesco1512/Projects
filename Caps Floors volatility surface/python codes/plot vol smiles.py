import pandas as pd
import matplotlib.pyplot as plt

def plot_normal_vol_smiles(file_name, sheet_name, maturities_to_plot):
    """
    Reads a vol matrix (rows = maturity, columns = strike)
    and plots the vol smile (normal vols) for the given list of maturities.
    
    Parameters
    ----------
    file_name : str
        Path to your Excel (or CSV) file.
    sheet_name : str
        Sheet name if Excel; ignore if reading CSV.
    maturities_to_plot : list
        The maturities (e.g. [1, 2, 5, 10]) you want to plot.
    """
    # 1) Read the data
    df = pd.read_excel(file_name, sheet_name=sheet_name, header=0)
    
    # 2) Set "maturity" as the index, so each row is a maturity
    df.set_index("maturity", inplace=True)
    
    # Now df looks like:
    #            -1.5    -1.25    -1.0    -0.5    0.0   0.25  0.5  ...  10.0
    # maturity
    #    1.0      0.21     0.15     ... 
    #    1.5      0.03     0.07     ...
    #    ...
    
    # 3) Convert the remaining columns (strike labels) to float, if needed
    df.columns = df.columns.astype(float)
    
    # 4) Plot the smiles for the selected maturities
    plt.figure(figsize=(8, 5))
    
    for mat in maturities_to_plot:
        if mat not in df.index:
            print(f"Warning: maturity {mat} not found in data.")
            continue
        
        # Extract the row for this maturity
        row = df.loc[mat]  # This is a Series indexed by strike
        strikes = row.index.values
        vols = row.values
        
        # Filter out any NaNs if present
        valid_mask = ~pd.isna(vols)
        strikes = strikes[valid_mask]
        vols = vols[valid_mask]
        
        # Plot
        plt.plot(strikes, vols, marker='', label=f"Maturity={mat}")
    
    plt.title("Implied normal vol")
    plt.xlabel("Strike")
    plt.ylabel("Normal Vol ")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_name = "./advanced interest rate project 2/inputs 2.xlsx"
    sheet_name = "plotting vol smiles"
    maturities_to_plot = [5, 10, 20, 30]  # years
    
    plot_normal_vol_smiles(file_name, sheet_name, maturities_to_plot)
