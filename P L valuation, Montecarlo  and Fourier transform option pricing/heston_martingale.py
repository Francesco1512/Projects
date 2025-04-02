#!/usr/bin/env python3

import sys
from sys import stdout as cout
from math import *
import numpy as np
import matplotlib.pyplot as plt
from CFLib_25.config import get_input_parms
from CFLib_25.Heston import Heston
from CFLib_25.CIR import CIR,  cir_evol, QT_cir_evol
from CFLib_25.heston_evol import mc_heston, heston_trj
from CFLib_25.timer import Timer
# -----------------------------------------------------

def usage():
    print("Usage: ./main.py -in input_file [ -- help ] [-out out_file] [-nv ev] [ -ns es] [-y year] [-nt intrvls] [ -dt dt]")
    print("                                [-eta eta] [-k lambda] [-th nubar] [ -ro nu] [ -rho rho]")
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

    So = 1.0
    dd = 1./365.
    hh = dd/24.
    mm = hh/60.
    # --------------------------

    parms = get_input_parms(argv)

    if "help" in parms:
        usage()
        return

    nv = int(parms.get("nv","10") )
    ns = int(parms.get("ns","10") )
    Yrs= float(parms.get("y", "1.13") )
    Nt = int(parms.get("nt","12") )
    if "dt" in parms: dt = float(parms["dt"])
    else:             dt = dd
    # -----------------------------------------


    eta   = float(parms.get("rho", "2.0170") )
    lmbda = float(parms.get("k", "7.7648") )
    nubar = float(parms.get("th", "0.0601") )
    nu_o  = float(parms.get("ro", "0.0475") )
    rho   = float(parms.get("rho", "-0.6952") )

    NV = (1 << nv)
    NS = (1 << ns)
    Dt = Yrs/Nt

    feller = eta**2/(2*lmbda*nubar)
    print("@ %-12s: Feller = %8.4f" %("Info", feller) )

    # geometry
    nCir = int(Yrs/dt)
    dt   = float(Yrs/nCir)

    Obj = np.random.RandomState()
    Obj.seed(29283)
    # -------------------------------

    # S   = np.zeros(shape=(NV, Nt+1, NS), dtype=np.float64) 
    S = heston_trj( Obj
                  , So  
                  , lmbda  # kappa
                  , eta    # sigma
                  , nubar  # theta
                  , nu_o   # ro
                  , rho    # correlation
                  , Yrs    # Length of the trajectory in years
                  , dt     # step per vol inegration
                  , Nt     # number of steps in the S trajectory
                  , NV     # number of vol trajectories
                  , NS     # number of S trajectory per vol trajectory
                  )

    Savg = np.add.reduce(S,2)/NS
    Savg = np.add.reduce(Savg,0)/NV

    Dt   = Yrs/Nt

    # builds the array t = [0, 1, 2, ... , Nt]
    t  = np.arange(0, Nt+1, 1, dtype = np.double)

    # the array t is modified as follows: t = [ 0, Dt, 2*Dt, ...,  Nt*Dt]
    t *= Dt

    Em  = list()
    Err = list()
    for nt in range(Nt+1):
        Em.append( S[:,nt,:].mean() )
        Err.append( S[:,nt,:].std()/sqrt(NV) )

    print("\n%6s  %6s  %6s" %("t","E[S]","Err"))
    for x,y,z in zip(t,Em, Err):
        print("%6.3f  %6.4f  %6.4f" %(x,y,z))
    # ------------------------------

    # plot MC results
    plt.errorbar(t, Em, yerr=Err, fmt='x', color='g', label="MC $2^{%d+%d}$" %(nv, ns))
    plt.title("Martingale Test -- Heston Model\n$\\nu_o$=%4.2f, $\lambda$=%4.2f, $\eta$=%5.2f, $\overline{\\nu}$=%5.2f, $\\rho$=%6.4f" %(nu_o, lmbda, eta, nubar, rho))
    plt.ylim(top=1.1, bottom=.9)
    plt.xlabel("Years")
    plt.ylabel("E[S(t)]")

    # martingale check
    Cte = np.full(shape = Nt+1, fill_value=1., dtype=np.double)
    plt.plot( t, Cte, color='r', label="S_0")
    plt.legend(loc="best")

    plt.savefig('heston_martingale.eps', format='eps', dpi=9600)
    plt.show()

    return
# -------------------------------


if __name__ == "__main__":
    run(sys.argv)
