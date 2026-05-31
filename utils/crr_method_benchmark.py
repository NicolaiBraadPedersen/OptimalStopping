import numpy as np

def crr_put_bound(T, N, r, S, sigma, K):
    dt = 1/N
    u = np.exp( sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    R = np.exp(r*dt)
    p = (R - d) / (u - d)
    #print (u,d, R)

    X = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N+1)])

    XX = X.copy()

    X = np.delete(X, -1)
    XX = np.delete(XX, 0)

    cont_val = []

    Y = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N)])

    for i in range(T * N, 0, -1):

        ev = Y
        bdv = R ** -1 * (p * X + (1-p) * XX)

        if i < T * N and np.sum(ev > bdv)>0:
            cont_val_element = ev[ev < bdv]
            cont_val_element_filt = np.max(cont_val_element) if cont_val_element.size > 0 else K
        elif i == T * N:
            cont_val_element_filt = 0
        else:
            cont_val_element_filt = np.nan

        val = np.maximum(bdv, ev)

        X_temp = X.copy()
        X = np.delete(val.copy(), -1)
        XX = np.delete(val.copy(), 0)
        Y = np.delete(X_temp,0)

        cont_val.append(K - cont_val_element_filt)
    return(cont_val[::-2])
