import numpy as np
from scipy.stats import norm
from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1, f_builder_2, f_builder_3, price_euro_put
from pandas import Timestamp as ts

def price_put_benchmark(T, N, r, S, sigma, K, timed = False):
    t0 = ts.now()
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


    Y = np.array([max(0, K - (S * (d ** k * u ** (T * N-k)))) for k in range(T * N)])

    for i in range(T * N, 0, -1):

        ev = Y
        bdv = R ** -1 * (p * X + (1-p) * XX)
        val = np.maximum(bdv, ev)

        X_temp = X.copy()
        X = np.delete(val.copy(), -1)
        XX = np.delete(val.copy(), 0)
        Y = np.delete(X_temp,0)

    time_to_conv = (ts.now() - t0).total_seconds()

    if not timed:
        return val[0]
    else:
        return val[0], time_to_conv

def price_amr_put(t, mat, s, strike, r, sigma, b, timed = False):
    t0 = ts.now()
    boundary = b
    trap_sum = 0
    n = len(boundary)
    t_delta = mat / (n - 1)

    # Create time points matching how b() creates them
    time_points = np.linspace(t, mat, n)

    for i in range(n-1):
        t_left = time_points[i]
        t_right = time_points[i + 1]

        b_at_left = boundary[i]
        b_at_right = boundary[i+1]

        g_left = g_func(t_left, t, b_at_left, s, strike, r, sigma)
        g_right = g_func(t_right, t, b_at_right, s, strike, r, sigma)

        trap_sum += (g_left + g_right) * t_delta / 2

    price = price_euro_put(t, s, mat, strike, r, sigma) + trap_sum  # Fixed argument order

    if not timed:
        return price
    else:
        return price, (ts.now() - t0).total_seconds()

def g_func(u, t, b_u, s, strike, r, sigma):
    if u <= t:                       # limit as u -> t+ with b_t = s (off-boundary)
        if s > b_u:
            return 0.0
        elif s < b_u:
            return r * strike
        else:
            return r * strike / 2
    phi = norm.cdf((np.log(b_u / s) - (r - 0.5*sigma**2)*(u - t)) / (sigma*np.sqrt(u - t)))
    return r * strike * np.exp(-r*(u - t)) * phi

if __name__ == "__main__":
    s0, t0, mat0, strike0, r0, sigma0, n0 = 36, 0, 1, 40, 0.06, 0.2, 200
    boundary0, time = b_fixed_point(t0, mat0, strike0, r0, sigma0, n0, f_builder_1, tol=0.0001)
    # print(time)
    # boundary1, time = b_num_solv(t0, mat0, strike0, r0, sigma0, n0, f_builder_1)
    # print(time)
    p0, time0 = price_amr_put(t0, mat0, s0, strike0, r0, sigma0, boundary0, timed=True)
    print(time0)
    # p1 = price_amr_put(t0, mat0, s0, strike0, r0, sigma0, boundary1)
    # p_bm = price_put_benchmark(mat0, 5*10**4, r0, s0, sigma0, strike0)
    # print(p0, p1, p_bm)