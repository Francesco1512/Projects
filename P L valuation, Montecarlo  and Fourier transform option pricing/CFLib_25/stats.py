from math import *
import numpy as np 

def stats(x, axis=None):

    '''
    Given the constant array 'x', stats will return the tuple
    ( E[x], StDev(x) := sqrt( E[ (x - E[x])^2 ] ))
    where E, represents the sample average.
    '''

    return x.mean(axis=axis), x.std(axis=axis)
