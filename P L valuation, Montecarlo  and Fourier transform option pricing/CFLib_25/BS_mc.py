from math import *
import numpy as np

def BS_trj(Obj, nt, So, T, J, sigma):

    '''
    rand :      np.Random.RandomState
    So   :      (float) initial value
    sigma:      (float) volatility
    T    :      (float) option maturity
    nt   :      (int)   number of steps in the trajectory
    J    :      (int)   number of trajectories
    
    Generates J trajectories according to the Black+Scholes model
    Each trajectory is made up of nt equally spaced steps.

    The output matrix will have the geometry S[nt+1][J],
    For each trajectory S[0] will hold the initial value.
    '''

    DT = T/nt
    S  = np.ndarray(shape = ( nt+1, J), dtype=np.double)

    X  = Obj.normal( -.5*sigma*sigma*DT, sigma*sqrt(DT), (nt,J))

    S[0] = So

    for n in range(nt):
        # The vector statement is equivalent to the
        # following loop
        #
        # for j in range(0, J): 
        #   S[n+1,j] = S[n,j] * exp( X[n,j] )
        S[n+1] = S[n] * np.exp( X[n] )

    return S
