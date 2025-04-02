#!/usr/bin/env python3

import sys
from sys import stdout as cout
from scipy.stats import norm
import numpy as np
from math import *
try:
    from .config import get_input_parms
except ImportError:
    from config import get_input_parms

def cn_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d)
# ------------------------------------

def an_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d-s)
# ------------------------------------

'''
    PUT = exp(-rT)Em[ (K - S(T))^+]
        where
    S(T) = So exp( (r-q)*T)*M
        let
    Fw(T) = So exp( (r-q)*T)
    kT    = K/Fw
        then
    PUT = So exp(-qT) Em[ (kT - M)^+]
        = So exp(-qT) StdFormEuroPut( T, sigma, kT)
'''

def StdFormEuroPut(T, vSigma, vKt):
    return ( vKt* cn_put( T, vSigma, vKt) - an_put( T, vSigma, vKt) )

def StdFormEuroCall(T, sigma, vkT):
    return StdFormEuroPut(T, sigma, vkT) + 1. - vkT

def euro_put(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * StdFormEuroPut( T, sigma, kT)
# -----------------------

def euro_call(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * StdFormEuroCall( T, sigma, kT)
# -----------------------

def __scalar_impVolFromStdFormPut(Price, T, Kt):

    #  Zero volatility
    Sl = 0.0

    # the price associated to zero volatility
    # Pl = max(Kt - 1., 0.0)

    Sh = 1.0
    while True:
        Ph = StdFormEuroPut(T, Sh, Kt)
        if Ph > Price: break
        #Sl = Sh
        #Pl = Ph
        Sh = 2*Sh

    # d     := vSh-vSl
    # d/2^N < eps
    # d     < eps* 2^N
    # N     > log(d/eps)/log(2)
    eps = 1.e-08
    d   = Sh-Sl
    N   = 2+int(log(d/eps)/log(2))

    for n in range(N):
        Sm  = .5*(Sh + Sl)
        Pm  = StdFormEuroPut(T, Sm, Kt)
        if Pm > Price: Sh = Sm
        else:          Sl = Sm

    
    return .5*(Sh + Sl)
# -----------------------

def __vec_impVolFromStdFormPut(vPrice, T, vKt):

    #  Zero volatility
    vSl = np.zeros(len(vKt))

    # the price associated to zero volatility
    vPl = np.maximum(vKt - 1., 0.0)

    vSh = np.ones(vKt.shape[0])
    while True:
        vPh = StdFormEuroPut(T, vSh, vKt)
        if ( vPh > vPrice).all(): break
        vSh = 2*vSh

    # d     := vSh-vSl
    # d/2^N < eps
    # d     < eps* 2^N
    # N     > log(d/eps)/log(2)
    eps = 1.e-08
    d   = vSh[0]-vSl[0]
    N   = 2+int(log(d/eps)/log(2))

    for n in range(N):
        vSm  = .5*(vSh + vSl)
        vPm  = StdFormEuroPut(T, vSm, vKt)
        mask = vPm > vPrice
        vSh[mask] = vSm[mask]
        vSl[~mask] = vSm[~mask]

    
    return .5*(vSh + vSl)
# --------------------------------------------

def impVolFromStdFormPut(put, T, k):

    if isinstance(k, float):
        return __scalar_impVolFromStdFormPut(put, T, k)

    return __vec_impVolFromStdFormPut(put, T, k)
# --------------------------------------------

def vanilla_options( **keywrds):

    So     = keywrds["S"]
    k      = keywrds["k"]
    r      = keywrds["r"]
    q      = keywrds.get("q", 0.0)
    T      = keywrds["T"]
    sigma  = keywrds["sigma"]


    kT   = exp((q-r)*T)*k/So
    cnP  = k*exp(-r*T)*cn_put ( T, sigma, kT)
    anP  = So*exp(-q*T)*an_put ( T, sigma, kT)
    put  = euro_put ( So, r, q, T, sigma, k)
    call = euro_call( So, r, q, T, sigma, k)

    return {"put": put, "call": call, "anP": anP, "cnP": cnP}
# --------------------------
def usage():
    print("Computes the value of european Call/Put options")
    print("and put-cash or nothing and put asset or nothing")
    print("Usage: $> ./euro_opt.py [options]")
    print("Options:")
    print("    %-24s: this output" %("--help"))
    print("    %-24s: (dbl) initial value of the underlying, defaults to 1.0" %("-s So"))
    print("    %-24s: (dbl) option strike, defaults to 1.0" %("-k strike"))
    print("    %-24s: (dbl) option strike, defaults to .40" %("-sigma volatility"))
    print("    %-24s: (dbl) option maturity, defaults to 1.0" %("-T maturity"))
    print("    %-24s: (dbl) interest rate, defaults to 0.0" %("-r ir"))
    print("    %-24s: (dbl) dividend yield, defaults to 0.0" %("-q qy"))
# ----------------------------------
    '''
    put  = e^(-rt ) E[ ( k - S(t) ]
           e^(-rt ) E[ ( e^[ (r-q)t] (So+D) M(t) - (k+Delta) ]
           e^(-qt ) (So+Delta) E[ M(t) - K(t) ]
           K(t) = ( k + Delta)/(So+Delta) e^{-(r-q)t}
    '''

def impVolFromPut(**kwrds):
    price = kwrds["price"]
    So = kwrds["So"]
    r  = kwrds["r"]
    q  = kwrds["q"]
    T  = kwrds["T"]
    k  = kwrds["k"]
    Delta = kwrds.get("Delta", 0.0)

    price = price*exp(q*T) / ( So+Delta )
    kT    = exp( -(r-q)*T)*(k+Delta)/(So+Delta)

    return impVolFromStdFormPut(price, T, kT)


def run(args):

    output    = None
    So     = 1.0
    k      = 1.0
    T      = 1.0
    r      = 0.0
    q      = 0.0
    Sigma  = .2347
    inpts  = get_input_parms(args)

    if "help" in inpts:
        usage()
        return

    So     = float( inpts.get("So", 1.0) )
    r      = float( inpts.get("r", 0.0) )
    q      = float( inpts.get("q", 0.0) )
    sigma  = float( inpts.get("sigma", 0.4) )
    k      = float( inpts.get("k", 1.0) )
    T      = float( inpts.get("T", 1.0) )

    print("@ %-24s %8.4f" %("So", So))
    print("@ %-24s %8.4f" %("r", r))
    print("@ %-24s %8.4f" %("q", q))
    print("@ %-24s %8.4f" %("sigma", sigma))
    print("@ %-24s %8.4f" %("k", k))
    print("@ %-24s %8.4f" %("T", T))

    res = vanilla_options( T=T, r =r, q=q, sigma=Sigma, k = k, S = So)
    kT = exp((q-r)*T)*k/So
    fwPut = StdFormEuroPut(T, Sigma, kT)
    impVol = impVolFromStdFormPut(fwPut, T, kT)

    print("@ Put %14.10f,  Call %14.10f,  Pcn %14.10f,  Pan %14.10f  ImpVol: %10.6f" %(res["put"], res["call"], res["cnP"], res["anP"], impVol))

# --------------------------

if __name__ == "__main__":
    run(sys.argv)
