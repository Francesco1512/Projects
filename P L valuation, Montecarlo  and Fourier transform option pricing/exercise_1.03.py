import sys
from sys import stdout as cout
from math import *
import numpy as np
import matplotlib.pyplot as plt
from CFLib_25.euro_opt import euro_put, impVolFromStdFormPut
from heston_cf import np_pricer #(hestn, strikes, T, Xc=Xc )
from CFLib_25.Heston import Heston
from CFLib_25.config import get_input_parms
from CFLib_25.CIR import CIR,  cir_evol, QT_cir_evol, fast_exact_cir_evol
from CFLib_25.heston_evol import mc_heston
from CFLib_25.timer import Timer
# -----------------------------------------------------

def usage():
    print("Usage: ./main.py -in input_file [ -- help ] [-nv ev] [ -ns es] [-y year] [-nt intrvls] [ -dt dt]")
    print("                                [-r r] [-eta eta] [-k lambda] [-th nubar] [ -ro nu] [ -rho rho]")
    print("        input_file: the input file holding interest rates")
    print("        ev        : log base 2 of the number of vol trajectories")
    print("                    defaults to 10")
    print("        es        : log base 2 of the number of MC trajectories per vol trajectory")
    print("                    If undefined will default to 10")
    print("        year      : the number of years of the simulation")
    print("                    defaults to 1")
    print("        intrvl    : the number of intervals we are going to measure")
    print("                    defaults to 12")
    print("        dt        : the step for the CIR model")
    print("                    defaults to 1day")
    print(" ")
    print("        r         : interest rate")
    print("                    defaults to 0.0")
    print("        eta       : the vol of the vol process")
    print("                    defaults to .01")
    print("        lambda    : the kappa paramter in the cooresponding CIR model")
    print("                    defaults to .1153")
    print("        nubar     : the theta parameter in the corresponding CIR model")
    print("                    defaults to .024")
    print("        nu        : the initial value for vol of vol")
    print("                    defaults to .0635")
    print("        rho       : correlation between vol and underlying wiener processes")
    print("                    defaults to .2125")
#-------------

def run(argv):
    import sys
    import matplotlib.pyplot as plt
    from CFLib_25.euro_opt import euro_put

    dd = 1./365.
    parms = get_input_parms(argv)

    if "help" in parms:
        usage()
        return

    # --- input ---
    Strike = float(parms.get("Strike", "1.03"))
    So     = float(parms.get("So", "1.0"))
    nv     = int(parms.get("nv", "10"))
    ns     = int(parms.get("ns", "10"))
    Yrs    = float(parms.get("y", "1.13"))
    Nt     = int(parms.get("nt", "20"))
    dt     = float(parms.get("dt", f"{1./365.}"))
    tM     = float(parms.get("tm", "0.5"))  # maturity to mark P&L
    delta_str = parms.get("delta", "-0.5,-0.1,-0.05,0,0.05,0.1,0.5")
    deltas = [float(d.strip()) for d in delta_str.split(",")]

    lmbda = float(parms.get("k"  , "7.7648"))
    nubar = float(parms.get("th" , "0.0601"))
    eta   = float(parms.get("eta", "2.0170"))
    nu_o  = float(parms.get("ro" , "0.0475"))
    rho   = float(parms.get("rho", "-0.6952"))

    r     = float(parms.get("r", "0.01"))
    q     = float(parms.get("q", "0.00"))

    NV = (1 << nv)
    NS = (1 << ns)
    Dt = Yrs / Nt
    iM = int(tM / Dt)
    tau = Yrs - iM * Dt
    sigma_I = 0.2196
    Pi_0 = 0.1033

    feller = eta * eta / (2 * lmbda * nubar)
    print("@ %-12s: Feller = %8.4f" % ("Info", feller))

    nCir = 1 + int(Yrs / dt)
    dt   = float(Yrs / nCir)

    Obj = np.random.RandomState()
    Obj.seed(29283)

    # --- CIR volatility simulation ---
    vol, Ivol = QT_cir_evol(Obj, CIR(kappa=lmbda, sigma=eta, theta=nubar, ro=nu_o),
                            nCir, dt, Nt, Dt, NV)
    Ivol = Ivol.transpose()
    vol = vol.transpose()

    All_S_tM = np.zeros(NV * NS)
    counter = 0

    To = Timer()
    T1 = Timer()

    To.start()
    elapsed = 0.0

    for n in range(NV):
        T1.start()
        S = mc_heston(Obj, 1.0, vol[n], Ivol[n], CIR(kappa=lmbda, sigma=eta, theta=nubar, ro=nu_o),
                      rho, Dt, NS)
        S_tM = S[iM]
        All_S_tM[counter:counter+NS] = S_tM
        counter += NS
        elapsed += T1.stop()
        cout.write("%6d  (%10d)   %8.4f sec.\r" % (n, NS * n, elapsed))
        cout.flush()

    tEnd = To.stop()
    cout.write("%6d  (%10d)   elapsed %8.4f sec.\n" % (NV, NS * NV, tEnd))

    # --- Sensitivity to implied vol ---
    colors = ['purple', 'blue', 'cyan', 'green', 'orange', 'red', 'black']

    for i, delta in enumerate(deltas):
        sigma_adj = sigma_I * (1 + delta)
        BS_vals = np.array([
            euro_put(So=s_j, r=r, q=0.0, T=tau, sigma=sigma_adj, k=Strike)
            for s_j in All_S_tM
        ])
        discounted = np.exp(-r * tM) * BS_vals
        V = Pi_0 - discounted

        VaR_10 = np.percentile(V, 10)
        VaR_5  = np.percentile(V, 5)
        VaR_1  = np.percentile(V, 1)

        print(f"\nΔ = {int(delta*100)}%, σ = {sigma_adj:.4f}")
        print(f"Mean P&L: {np.mean(V):.5f}")
        print(f"Std Dev : {np.std(V):.5f}")
        print(f"VaR 10% : {VaR_10:.5f}")
        print(f"VaR 5%  : {VaR_5:.5f}")
        print(f"VaR 1%  : {VaR_1:.5f}")

        # Plot individual histogram
        plt.figure(figsize=(8,5))
        plt.hist(V, bins=50, density=True, edgecolor='black', alpha=0.75, color=colors[i % len(colors)])
        plt.axvline(VaR_10, color='orange', linestyle='--', label='VaR 10%')
        plt.axvline(VaR_5, color='red', linestyle='--', label='VaR 5%')
        plt.axvline(VaR_1, color='purple', linestyle='--', label='VaR 1%')
        plt.title(f"P&L Distribution (Δ = {int(delta*100)}%, σ = {sigma_adj:.4f})")
        plt.xlabel("P&L")
        plt.ylabel("Density")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run(sys.argv[1:])


