import numpy as np
from scipy import stats
from scipy.optimize import newton
from matplotlib import pyplot as plt


s0 = 36
K = 40
r = 0.06
sigma = 0.2
T = 1
n = 500
numiter = 20

trng = np.arange(0,T+T/n,T/n)[::-1]

def bs_put(s0,K,r,t,T,sigma):
    d1 = (np.log(s0/K)+(r+(sigma**2)/2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    price = K*np.exp(-r*(T-t))*stats.norm.cdf(-d2)-s0*stats.norm.cdf(-d1)

    return price


def G(t1, t2,b1,b2,sigma,r,K):

    G = r * K * np.exp(-r*(t2-t1)) * stats.norm.cdf(1/(sigma*np.sqrt(t2-t1))*(np.log(b2/b1)-(r-sigma**2/2)*(t2-t1)))
    return G


def G(t1, t2, b1, b2, sigma, r, K):
    dt = t2 - t1
    mask = dt > 0
    G = np.zeros_like(dt, dtype=float)
    G[mask] = r * K * np.exp(-r * dt[mask]) * stats.norm.cdf(
        (np.log(b2[mask] / b1) - (r - sigma**2 / 2) * dt[mask]) / (sigma * np.sqrt(dt[mask]))
    )
    return G


def step(b_candidate,trng,i,b_guess, sigma,r,K,n,T):
    t = trng[i]
    
    Gs = np.zeros_like(b_guess[:i+1])
    Gs = G(t,trng[:i+1],b_candidate,b_guess[:i+1],sigma,r,K)

    Gs[0]=Gs[0]/2
    Gs[-1]=(r*K/2)/2
    
    int_G = np.sum(Gs)*T/n

    return b_candidate-K+int_G + bs_put(b_candidate,K,r,t,T,sigma)
    


def recursion(trng, sigma,r,K,n,T):

    b_guess = np.ones_like(trng) * K

    for i in range(n):
        b_guess[i+1]=newton(step,b_guess[i],args = (trng,i+1,b_guess,sigma,r,K,n,T),maxiter=100)

    return b_guess



def early_ex(s0, trng, b, sigma, r, K, n, T):
    
    t1 = trng[-1] 
    
    Gs = G(t1, trng, s0, b, sigma, r, K)
    
    Gs[0] = Gs[0] / 2
    Gs[-1] = (r * K / 2) / 2

    eep = np.sum(Gs) * T / n
    return eep





def calc_step(i, trng, b_guess, sigma, r, K, T, n):
    t=trng[i]
    
    Gs = np.zeros_like(b_guess[:i+1])

    Gs = G(t,trng[:i+1],b_guess[i],b_guess[:i+1],sigma,r,K)

    Gs[0]=Gs[0]/2
    Gs[-1]=(r*K/2)/2
    
    int_G = np.sum(Gs)*T/n

    return K-int_G-bs_put(b_guess[i],K,r,t,T,sigma)



def picard_iteration(numiter, trng, sigma, r, K, T, n):
    b_guess = np.ones((n+1,numiter))*K

    for j in range(numiter-1):
        for i in range(n):
            b_guess[i+1,j+1]=calc_step(i+1,trng,b_guess[:,j],sigma,r,K,T,n)
    return b_guess

res1 = recursion(trng=trng,sigma=sigma,r=r,K=K,T=T,n=n)

print('Recursive')
print(round(bs_put(s0, K, r, 0, T, sigma) + early_ex(s0, trng, res1, sigma, r, K, n, T),3))


res2 = picard_iteration(numiter,trng=trng,sigma=sigma,r=r,K=K,T=T,n=n)
print('Picard')
print(round(bs_put(s0, K, r, 0, T, sigma) + early_ex(s0, trng, res2[:,-1], sigma, r, K, n, T),3))
