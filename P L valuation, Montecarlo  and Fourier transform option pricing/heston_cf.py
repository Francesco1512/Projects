#!/usr/bin/env python3

import math
import numpy as np
from math import *
import cmath
from CFLib_25.timer import named_timer
from CFLib_25.Heston import Heston
from CFLib_25.FT_opt import ft_opt

def dump_surface_as_csv(Out, So, Xc, name):
    with open(name,'w') as fp:
        fp.write("So,Strike,t,put,call,pCn,pAn,Xc\n")
        for t in Out.keys():
            xc = Xc*sqrt(t)
            res = Out[t]
            for k, put, call, pCn, pAn in zip( res['Strike'], res['put'], res['call'], res['pCn'], res['pAn']):
                fp.write(" %5.3f, %6.3f, %5.3f, %10.4f, %10.4f, %10.4f, %10.4f, %10.6f\n" %(So, k, t, put, call, pCn, pAn, xc))
# ---------------------------------------------------------------------------------------------

@ named_timer("serial_pricer")
def serial_pricer(model, So, r, qy, strikes, T, Xc = 1.0 ):
    print(f"strikes: {type(strikes)}\n{strikes}") 
    for t in T:
        print(" %5s  %6s  %5s  %10s  %10s  %10s  %10s  %10s" %("So", "Strike", "T", "put", "call", "pCn", "pAn", "Xc"))
        xc = Xc*sqrt(t)
        for Strike in strikes:
            Fw = So*exp((r-qy)*t)
            #
            # The discounted Strike so we can work with the unit martingale
            #
            kT = (Strike/Fw)
            res = ft_opt( model, kT, t, xc)
            res["put"] *= So*exp(-qy*t)
            res["call"] *= So*exp(-qy*t)
            res["pAn"] *= So*exp(-qy*t)
            print(" %5.3f  %6.3f  %5.3f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f" %(So, Strike, t, res["put"], res["call"], res["pCn"], res["pAn"], xc))
    print("@ --")
# ---------------------------------------------------------------------------------------------

@ named_timer("np_pricer")
def np_pricer(model, So, r, qy, strikes, T, Xc = 1.0 ):

    Out = dict()
    if isinstance(strikes, list):
        _vStrikes = np.array(strikes)
    else:
        _vStrikes = strikes

    for t in T:
        Fw = So*exp((r-qy)*t)
        xc = Xc**sqrt(t)

        #
        # The discounted Strike so we can work with the unit martingale
        #
        _vkT = (_vStrikes/Fw)
        Out[t] = ft_opt( model, _vkT, t, xc)
        Out[t]["put"] *= So*exp(-qy*t)
        Out[t]["call"] *= So*exp(-qy*t)
        Out[t]["pAn"] *= So*exp(-qy*t)
        Out[t]["Strike"] = _vStrikes
    return Out
# ---------------------------------------------------------------------------------------------

def run():

    strikes  = 1.03 #[ .9, .95, 1.0, 1.03, 1.05, 1.10, 1.15 ]
    So       = 1
    T        = 1.13 #[ n/12. for n in range(1,13)]
    Xc       = 5.0000
    r        =  0.01
    qy       =  0.0

    lmbda =   7.764800
    nubar =   0.060100
    eta   =   2.017000
    nu_o  =   0.047500
    rho   =  -0.695200


    hestn = Heston( lmbda = lmbda
                  , eta   = eta
                  , nubar = nubar 
                  , nu_o  = nu_o 
                  , rho   = rho
                  )



    print("@ %-12s: %-8s = %8.6f"  %("info", "Felle", eta*eta/(2*lmbda*nubar)) )
    print("@ %-12s: %-8s = %8.6f"  %("info", "lambda", lmbda) )
    print("@ %-12s: %-8s = %8.6f"  %("info", "nubar" , nubar) )
    print("@ %-12s: %-8s = %8.6f"  %("info", "eta"   , eta) )
    print("@ %-12s: %-8s = %8.6f"  %("info", "nu_o"  , nu_o) )
    print("@ %-12s: %-8s = %8.6f"  %("info", "rho"   , rho) )
    print

    
    if not isinstance( strikes, list): strikes = [strikes]
    if not isinstance( T, list):       T = [T]
    print("@ --")

    serial_pricer(hestn, So, r, qy, strikes, T, Xc=Xc)
    print("@ --")
    
    Out = np_pricer(hestn, So, r, qy, strikes, T, Xc=Xc )
    print("@ --")
    dump_surface_as_csv(Out, So, Xc, "heston_np_pricer.csv")

    print("%6s  %6s  %6s" %("t", "k", "put"))
    for t in T:
        for k in strikes:
            put = hestn.HestonPut( So     = So
                                   , Strike = k
                                   , T      = t
                                   , Xc     = Xc
                                   , r      = r
                                   , q      = qy
                                   )
            print("%6.4f  %6.4f  %6.4f" %(t, k, put))

if __name__ == "__main__":
    run()
