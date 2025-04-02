#!/usr/bin/env python3

import sys
from math import *
import numpy as np
import pandas as pd

try:
    from CFLib_25.finder import find_pos
except ModuleNotFoundError:
    from finder import find_pos
# -----------------------------------------------------

class Zc:

    def __init__( self, **keywrds):
        curve   = keywrds["curve"]
        self.tl = curve[0]
        self.rc = curve[1]
        self.pt = np.exp(-self.tl*self.rc)
    # ------------------------------------------

    @classmethod
    def from_discount_curve(cls, t, P):
        if t[0] == 0:
            t = np.array(t[1:])
            P = P[1:]
        r = -np.log(P)/t
        return cls(curve=(t, r))

    @classmethod
    def from_cc_zero_coupon_rates(cls, t, rc):
        return cls(curve=(t, rc))

    @classmethod
    def from_yc_zero_coupon_rates(cls, t, yc):
        r = np.log( 1 + yc)
        return cls(curve=(t, r))


    def f_0t( self, t ):

        '''
            f_0t := \int_0^t r(s) ds
            
            condition: t >= 0
        '''

        tl  = self.tl
        r  = self.rc
        pt = self.pt

        n = find_pos( t, tl )

        if n < 0            : return t*r[0]
        if n == tl.size - 1 : return t*r[-1]
        
        fs = tl[n]*r[n]
        fe = tl[n+1]*r[n+1]
        return (tl[n+1] - t) * fs/(tl[n+1] - tl[n]) + (t - tl[n])* fe/(tl[n+1]-tl[n])
    #------------------------------------------------

    def rz( self, t)  : return self.f_0t(t)/t
    def ry( self, t)  : return exp(self.f_0t(t)/t) - 1.
    def P_0t( self, t): return exp( -self.f_0t(t) )
    # -----------------------------------------------------------------

    
    def swap_rate( self, tm=0.0, p=10, Dt=1.0):
        '''
        returns swap rate R_p(tm)
        and annuity       A_p(tm)

        where
        R_p(tm) = [ P(0, t_m) - P(0, t_m + p  Dt) ]/A_p(t_m)
        A_p(tm) = Sum_[1 <= j <= p] Dt P(0,t_m+j*Dt )
        '''
        A  = 0.0

        for n in range(1,p+1): A += Dt*self.P_0t( tm + n*Dt)
        Num = self.P_0t(tm) - self.P_0t( tm+p*Dt)
        return Num/A, A
    # ------------------------------------------------

    def get_df(self, tl = None):
        '''
            return a data frame represneting the P_0T object
        '''

        if tl is None: tl = self.tl

        _p  = np.array([ self.P_0t(t) for t in tl ]  )
        _xc = np.array([ self.rz(t)   for t in tl ]  )
        _xy = np.array([ self.ry(t)   for t in tl ]  )

        return pd.DataFrame( data = {"t": tl, "P_0t": _p, "rc": _xc, "ry": _xy})
    # ------------------------------------------------


    def show( self, tl = None ):

        '''
            displays the P_0T and returns the associated df
        '''
        if tl is None: tl = self.tl

        df = self.get_df(tl = tl)
        print("%3s, %10s, %10s, %10s, %10s" %( "pos", "t", "P_0t", "rc", "ry"))
        n = 0
        for t,p,xc,xy in zip(df['t'], df['P_0t'], df['rc'], df['ry']):
            print("%3d, %10.6f, %10.6f, %10.6f, %10.6f" %( n, t, p, xc, xy))
            n += 1
        


# -----------------------------------------------------
