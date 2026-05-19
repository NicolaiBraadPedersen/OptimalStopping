import numpy as np
from scipy.stats import norm

def G(u,t,b_u,b_t,strike,r,sigma):
    if u == t:
        return r*strike/2
    else:
        Phi = norm.cdf(1/(sigma*np.sqrt(u-t))*(np.log(b_u/b_t) - (r-0.5 * sigma**2) * (u-t)))
        return r * strike * np.exp(-r*(u-t))*Phi

def P_e(t,s,T,K,r,sigma):
    d1 = 1 / (sigma * np.sqrt(T - t)) * (np.log(s / K) + (r + 0.5 * sigma ** 2) * (T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    price = np.exp(-r * (T - t)) * K * norm.cdf(-d2) - s * norm.cdf(-d1)
    return price


