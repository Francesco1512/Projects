#!/usr/bin/env python3

import sys
from sys import stdout as cout
from math import *
import numpy as np
import matplotlib.pyplot as plt
from heston_cf import np_pricer #(hestn, strikes, T, Xc=Xc )
from CFLib_25.Heston import Heston
from CFLib_25.config import get_input_parms
from CFLib_25.CIR import CIR,  cir_evol, QT_cir_evol, fast_exact_cir_evol
from CFLib_25.heston_evol import mc_heston
from CFLib_25.timer import Timer
from CFLib_25.euro_opt import impVolFromStdFormPut
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

    dd = 1./365.
    hh = dd/24.
    mm = hh/60.


    parms = get_input_parms(argv)

    if "help" in parms:
        usage()
        return

    Strike = float( parms.get("Strike", "1"))
    So     = float( parms.get("So", "1.0"))
    nv     = int(parms.get("nv", "10"))
    ns     = int(parms.get("ns", "10"))
    Yrs    = float(parms.get("y", "2"))
    Nt     = int(parms.get("nt", "20"))
    dt     = float(parms.get("dt", f"{1./365.}"))
    # -----------------------------------------

    lmbda = float(parms.get("k"  , "7.7648") )
    nubar = float(parms.get("th" , "0.0601") )
    eta   = float(parms.get("rho", "2.0170") )
    nu_o  = float(parms.get("ro" , "0.0475") )
    rho   = float(parms.get("rho", "-0.6952") )

    r     = float(parms.get("r"  , "0.01"))
    q     = float(parms.get("q"  , "0.00"))

    NV = (1 << nv)
    NS = (1 << ns)
    Dt = Yrs/Nt

    feller = eta*eta/(2*lmbda*nubar)
    print("@ %-12s: Feller = %8.4f" %("Info", feller) )

    # geometry
    nCir = 1 + int(Yrs/dt)
    dt   = float(Yrs/nCir)

    Obj = np.random.RandomState()
    Obj.seed(29283)
    # -------------------------------

    # build the discount factor
    # and the dividend yield curve
    nl = np.arange(0, Nt+1, 1, dtype = np.double)

    df = np.exp(r*nl*Dt)
    qy = np.exp(q*nl*Dt)
    kt = (Strike/So)*(qy/df)

    model = Heston( lmbda = lmbda, nubar = nubar , eta   = eta , nu_o  = nu_o, rho   = rho)


    # the CIR model
    cir = CIR(kappa=lmbda, sigma=eta, theta=nubar, ro = nu_o)

    To = Timer()
    T1 = Timer()

    To.start()
    J = NV*NS
    vol, Ivol = QT_cir_evol( Obj, cir, nCir, dt, Nt, Dt, NV)
    Ivol = Ivol.transpose()
    vol = vol.transpose()
    elapsed = 0.0
    for n in range(NV):
        T1.start()
        S = mc_heston( Obj, 1., vol[n], Ivol[n], cir, rho, Dt, NS )
        payoff = (kt - S.transpose()).transpose()
        put = np.maximum( payoff, 0.0)
        put = np.add.reduce(put,1)/NS

        if n == 0.0: 
            PUT  = put/NV
            pErr = put*put/NV
        else:
            PUT = PUT + put/NV
            pErr = pErr + put*put/NV

        elapsed += T1.stop()
        cout.write("%6d  (%10d)   %8.4f sec.\r" %(n, NS*n, elapsed))
        cout.flush()

    tEnd = To.stop()
    cout.write("%6d  (%10d)   elapsed %8.4f sec.\n" %(NV, NS*NV, tEnd))
    CALL = 1 - kt + PUT
    pErr = np.maximum(pErr-PUT*PUT, 0.0)
    pErr = 2*np.sqrt(pErr/NV)
    # End MC

    pErr = So*pErr/qy
    PUT  = So*PUT/qy
    CALL = So*CALL/qy

    
    # builds the array t = [0, 1, 2, ... , Nt]
    t  = np.arange(0, Nt+1, 1, dtype = np.double)

    # the array t is modified as follows: t = [ 0, Dt, 2*Dt, ...,  Nt*Dt]
    t *= Dt
    # ------------------------------

    print("@ %-12s = %6.4f" %("r",   r))
    print("@ %-12s = %6.4f" %("q",   q))
    print("@ %-12s = %6.4f" %("eta",   eta))
    print("@ %-12s = %6.4f" %("nubar", nubar))
    print("@ %-12s = %6.4f" %("lmbda", lmbda))
    print("@ %-12s = %6.4f" %("nu_o",  nu_o))
    print("@ %-12s = %6.4f" %("rho",   rho))
    print("@ %-12s = %6.4f" %("strike",Strike))

    ftPut = np.zeros(t.shape[0])
    ftCall= np.zeros(t.shape[0])
    Out = np_pricer(model, So, r, q, [Strike], t, Xc = 5.)
    for n in range(t.shape[0]):
        x = t[n]
        if x == 0.0: 
            ftPut[n] = max(Strike - So, 0.0)
            ftCall[n] = max(So  - Strike, 0.0)
        else:
            ftPut[n] = Out[x]["put"][0]
            ftCall[n] = Out[x]["call"][0]

    print("\n%6s  %6s  %6s  %6s  %7s %7s  +/- %7s  %7s  %7s  %7s" %("So", "Strike", "kt", "t","Put","Call", "Err","ftPut", "ftCall", "impVol"))
    for x,y,c,z,kT in zip(t,PUT, CALL, pErr,kt):
        if x == 0.0: continue
        put =Out[x]["put"][0]
        call =Out[x]["call"][0]
        impVol = impVolFromStdFormPut(y, x, kT)  # y = MC put price

        print("%6.3f  %6.3f  %6.3f  %6.6f  %7.4f %7.4f  +/- %7.1e  %7.4f  %7.4f %7.4f" %(So, Strike, kT, x,y,c, z, put, call, impVol))
    # ------------------------------

    # plot MC results
    plt.errorbar(t, PUT, yerr=pErr, fmt='x', color='g', label="Put")
    plt.errorbar(t, CALL, yerr=pErr, fmt='x', color='r', label="Call")
    plt.plot(t, ftCall, 'ko', label="ftCall", markersize=4)
    plt.plot(t, ftPut, 'ko', label="ftPut", markersize=4)
    plt.title("European Options - Heston Model\nr=%5.3f  $\\nu_o$=%5.3f  $\lambda$=%5.3f  $\eta$=%5.3f  $\overline{\\nu}$=%5.3f" %(r, nu_o, lmbda, eta, nubar))
    plt.xlabel("Years")
    plt.ylabel("Price")
    plt.legend(loc="best")

    plt.show()
# -------------------------------

if __name__ == "__main__":
    #test()
    run(sys.argv)
