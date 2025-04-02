#!/usr/bin/env python3
import math
import cmath
from math import *
import numpy as np
from CFLib_25.stats import stats

class jumps:

    def __init__(self, **kwargs):
        self._intnsty = kwargs['lmbda']
        self._sgma    = kwargs['sigma']
# -----------------------------------------

    @property
    def intensity(self): return self._intnsty

    @property
    def sigma(self): return self._sgma

    def do_jmp(self, Obj, Dt, J):

        Z   = np.full(shape=J, fill_value=0.0, dtype=np.float64)
        Nj  = Obj.poisson(lam=self.intensity * Dt, size=J)
        sup = Nj.max()
        j   = sup

        while j > 0:
            Z = Z + self.single_jump(Obj, Nj >= j)
            j -= 1

        return Z
    # -------------------------------------------------------

    def jd_evol_step(self, rand, Dt, J):

        '''
        Performs 1 step for J trajectories 
        Black-Scholes diffusion + jumps
        '''

        s = self.sigma * sqrt(Dt)
        X = rand.normal( -.5*s*s, s, J)
        X = X + self.do_jmp(rand, Dt, J) + Dt*self.intensity*self.compensator()
        return np.exp(X)
    # -----------------------------------------------------

    def cf(self,c_k, t):
        s = self.sigma
        # 
        # c_u = i c_k
        #
        c_u = c_k*1j

        c_x  = -.5 *s*s*c_u*c_u
        comp = -.5 *s*s

        #
        # X_cf = dt * ( u*g - f )  
        #
        X_cf =  t*(comp*c_u - c_x)


        c_x  = self.phi_X(c_u)
        comp = self.compensator()
        JMP  =  t*self.intensity*(comp*c_u - c_x)

        return cmath.exp(X_cf + JMP)

    # =================================================================================


class jmp_binary(jumps):

    '''
    Pr( J==u ) = pi
    Pr( J==d ) = 1. - pi
    '''

    def __init__(self, **kwargs):
        self._pi      = kwargs["pi"] 
        self._u       = kwargs["u"] 
        self._d       = kwargs["d"] 
        super().__init__(**kwargs)
    # -------------------------------------------------------

    def single_jump(self, rand, mask):
            J       = len(mask)
            z       = rand.uniform(low=0.0, high=1.0, size=J)
            pi_mask = np.logical_and(mask, z < self._pi)
            up  = np.where(pi_mask, self._u, 0)

            pi_mask = np.logical_and(mask, z > self._pi)
            down  = np.where(pi_mask, self._d, 0)

            return ( up + down )
    # -------------------------------------------------------

    def compensator(self):
        phi_J =  self._pi*exp(self._u) + (1.0 - self._pi)*exp(self._d) 
        return (1.0 - phi_J)

    def phi_X(self, c_u):
        return 1. - self._pi*cmath.exp(self._u*c_u) - (1.-self._pi)*cmath.exp(self._d*c_u)
        
# ================================================================================

class jmp_normal(jumps):

    '''
    P( J < L ) = N_{0,1}( (L - m)/eta )
    '''

    def __init__(self, **kwargs):
        self._m       = kwargs["m"] 
        self._eta     = kwargs["eta"] 
        super().__init__(**kwargs)

    def single_jump(self, rand, mask):
            J       = len(mask)
            X   = rand.normal( self._m, self._eta, J)
            return np.where(mask, X, 0.)
    # -------------------------------------------------------

    def set(self, x):
        self._intnsty = x[0]
        self._m       = x[1]
        self._eta     = x[2]
        self._sgma    = x[3]

    def get(self):
        return [ self._intnsty, self._m, self._eta, self._sgma]

    def compensator(self):
       phi_J = exp( self._m + .5*(self._eta)*(self._eta)); 
       return (1.0 - phi_J)

    def phi_X(self, c_u):
        c_z = self._m*c_u + .5 * self._eta*self._eta*c_u*c_u
        return 1. - cmath.exp(c_z)

#!/usr/bin/env python3
from math import *
import numpy as np
from CFLib_25.stats import stats
from CFLib_25.jumps import jmp_binary, jmp_normal
from CFLib_25.FT_opt import ft_opt
from CFLib_25.euro_opt import impVolFromStdFormPut
# -----------------------------------------

def run_merton( **kwargs):
        Nt     =  kwargs.get("Nt", 12)
        Strike = 1.1
        So     =   1
        r      = 0.00
        q      = 0.00
        sigma  =  .4
        lmbda  = 1.4
        m      = .2
        eta    = 0.4
        Dt     = 1./12.
        J      = ( 1 << 22)
        Xc     = 10.

        Obj = np.random.RandomState(123456)
        jmp = jmp_normal(lmbda=lmbda, m = m, eta = eta, sigma = sigma)

        print("\nNormal Jumps")
        print("%6s %6s %8s  %6s %8s  %6s  %8s" %("t", "Em[S]", "err", "MCput", "Err", "FTput", "impVol"))

        Sn    = np.full(shape=J, fill_value=1.0, dtype=np.float64)
        for n in range(Nt):
            Sn  = Sn*jmp.jd_evol_step(Obj, Dt, J)
            T   = (n+1)*Dt
            Fw  = So * exp((r - q)*T)
            St  = So * exp(- q*T)
            kT  = Strike/Fw
            Es, stDev = stats(Sn)
            put = np.maximum(kT-Sn,0)
            put, err = stats(put)
            res = ft_opt(jmp, kT, T, Xc*sqrt(T))
            impVol = impVolFromStdFormPut(res['put'], T, kT)
            print("%6.4f %6.4f %8.2e  %6.4f %8.2e  %6.4f  %8.4f" %(T, St*Es, St*3.*stDev/sqrt(J), St*put, St*3*err/sqrt(J), St*res['put'], impVol) )
# -----------------------------------------

def run_binary(**kwargs):
    
        Nt     =  kwargs.get("Nt", 12)
        Strike = 1.1
        So     =   1
        r      = 0.0
        q      = 0.0
        sigma  =  .4
        lmbda  = 1.4
        pi     =  .7
        u      =  .2
        d      = -.2
        Dt     = 1./12.
        J      = ( 1 << 22)
        Xc     = 10.

        Obj = np.random.RandomState(123456)
        jmp = jmp_binary(lmbda=lmbda, pi = pi, u = u, d = d, sigma=sigma)

        print("\nBinary Jumps")
        print("%6s %6s %8s  %6s %8s  %6s  %8s" %("t", "Em[S]", "err", "MCput", "Err", "FTput", "impVol"))

        Sn    = np.full(shape=J, fill_value=1.0, dtype=np.float64)
        for n in range(Nt):
            Sn  = Sn*jmp.jd_evol_step(Obj, Dt, J)
            T   = (n+1)*Dt
            Fw  = So * exp((r - q)*T)
            St  = So * exp(- q*T)
            kT  = Strike/Fw
            Es, stDev = stats(Sn)
            put = np.maximum(kT-Sn,0)
            put, err = stats(put)
            res = ft_opt(jmp, kT, T, Xc*sqrt(T))
            impVol = impVolFromStdFormPut(res['put'], T, kT)
            print("%6.4f %6.4f %8.2e  %6.4f %8.2e  %6.4f  %8.4f" %(T, St*Es, St*3.*stDev/sqrt(J), St*put, St*3*err/sqrt(J), St*res['put'], impVol) )
# -----------------------------------------

if __name__ == "__main__":
    run_binary()
    run_merton()
