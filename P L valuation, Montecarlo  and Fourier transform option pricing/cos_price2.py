#%%
"""
Created on Thu Nov 27 2018
Pricing of European Call and Put options with the COS method
@author: Lech A. Grzelak
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from CFLib_25.euro_opt import impVolFromStdFormPut
from CFLib_25.timer import named_timer

# ---------------------- COS Function ----------------------

def CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L):
    K = np.array(K).reshape([len(K),1])
    i = 1j
    x0 = np.log(S0 / K)
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    k = np.linspace(0,N-1,N).reshape([N,1])
    u = k * np.pi / (b - a)
    H_k = CallPutCoefficients(CP,a,b,k)
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))
    return value

def CallPutCoefficients(CP,a,b,k):
    if str(CP).lower()=="c" or str(CP).lower()=="1":
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        H_k = 2.0 / (b - a) * (coef["chi"] - coef["psi"])
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        H_k = 2.0 / (b - a) * (-coef["chi"] + coef["psi"])
    return H_k    

def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    return {"chi":chi,"psi":psi }

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    cp = str(CP).lower()
    K = np.array(K).reshape([len(K),1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if cp == "c" or cp == "1":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif cp == "p" or cp =="-1":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

def HestonCF(u, tau, r, kappa, theta, sigma, rho, v0):
    i = 1j
    d = np.sqrt((rho * sigma * i * u - kappa)**2 + (sigma**2) * (i * u + u**2))
    g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)
    exp_dt = np.exp(-d * tau)
    G = (1 - g * exp_dt) / (1 - g)
    C = (kappa * theta / (sigma**2)) * ((kappa - rho * sigma * i * u - d) * tau - 2 * np.log(G))
    D = ((kappa - rho * sigma * i * u - d) / (sigma**2)) * ((1 - exp_dt) / (1 - g * exp_dt))
    cf = np.exp(C + D * v0 + i * u * r * tau)
    return cf

# ---------------------- TIMED MAIN ----------------------

@named_timer("COS_pricer")
def mainCalculation():
    CP = "p"
    S0 = 1.0
    r = 0.01
    sigma = 2.0170
    K = [1.03]
    N = 512
    L = 10
    tau_list = [1.13]

    # Heston model parameters
    kappa = 7.7648
    theta = 0.0601
    sigma = 2.0170
    v0    = 0.0475
    rho   = -0.6952

    print(f"{'tau':>8}  {'Put_COS':>12}")
    print("-" * 25)

    for tau in tau_list:
        cf = lambda u: HestonCF(u, tau, r, kappa, theta, sigma, rho, v0)
        val_COS = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L)
        print(f"{tau:8.3f}  {val_COS[0,0]:12.8f}")

# ----------------------

if __name__ == "__main__":
    mainCalculation()
