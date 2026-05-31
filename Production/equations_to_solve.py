import numpy as np
from scipy.stats import norm

def f_builder_1(ctx):
    """Default residual: (K - x) - ∫G - euro_put = 0, matching the original."""
    strike, i, mat, r, sigma = ctx.strike, ctx.i, ctx.mat, ctx.r, ctx.sigma
    t_delta, future_times, b_s_nodes = ctx.t_delta, ctx.future_times, ctx.b_s_nodes

    mtd = getattr(ctx, 'method')

    def f(x):
        g_vals = np.array([
            g_func(s, i, b_s, x, strike, r, sigma)
            for s, b_s in zip(future_times, b_s_nodes)
        ])
        integral = t_delta * (g_vals[0] / 2 + g_vals[1:-1].sum() + g_vals[-1] / 2)
        if mtd == 'num_solver':
            out = (strike - x) - integral - price_euro_put(i, x, mat, strike, r, sigma)
        elif mtd == 'fix_point':
            out = strike - integral - price_euro_put(i, x, mat, strike, r, sigma)
        else:
            out = np.nan
        return out

    return f

def f_builder_2(ctx):
    """Utilize put-call parity..."""
    strike, i, mat, r, sigma = ctx.strike, ctx.i, ctx.mat, ctx.r, ctx.sigma
    t_delta, future_times, b_s_nodes = ctx.t_delta, ctx.future_times, ctx.b_s_nodes

    def f(x):
        g_vals = np.array([
            g_func(s, i, b_s, x, strike, r, sigma)
            for s, b_s in zip(future_times, b_s_nodes)
        ])
        integral = t_delta * (g_vals[0] / 2 + g_vals[1:-1].sum() + g_vals[-1] / 2)
        if method == 'num_solver':
            out = (strike - x) - integral - price_euro_put(i, x, mat, strike, r, sigma)
        elif method == 'fix_point':
            out = strike - integral - price_euro_put(i, x, mat, strike, r, sigma)
        else:
            out = np.nan
        return out

    return f

def f_builder_3(ctx):
    """Use Smooth fit!!!"""
    strike, i, mat, r, sigma = ctx.strike, ctx.i, ctx.mat, ctx.r, ctx.sigma
    t_delta, future_times, b_s_nodes = ctx.t_delta, ctx.future_times, ctx.b_s_nodes

    def f(x, method=method):
        g_vals = np.array([
            g_func(s, i, b_s, x, strike, r, sigma)
            for s, b_s in zip(future_times, b_s_nodes)
        ])
        integral = t_delta * (g_vals[0] / 2 + g_vals[1:-1].sum() + g_vals[-1] / 2)
        if method == 'num_solver':
            out = (strike - x) - integral - price_euro_put(i, x, mat, strike, r, sigma)
        elif method == 'fix_point':
            out = strike - integral - price_euro_put(i, x, mat, strike, r, sigma)
        else:
            out = np.nan
        return out

    return f

def g_func(u,t,b_u,b_t,strike,r,sigma):
    if u == t:
        return r*strike/2
    else:
        phi = norm.cdf(1/(sigma*np.sqrt(u-t))*(np.log(b_u/b_t) - (r-0.5 * sigma**2) * (u-t)))
        return r * strike * np.exp(-r*(u-t))*phi

def price_euro_put(t,s,mat,strike,r,sigma):
    d1 = 1 / (sigma * np.sqrt(mat - t)) * (np.log(s / strike) + (r + 0.5 * sigma ** 2) * (mat - t))
    d2 = d1 - sigma * np.sqrt(mat - t)
    price = np.exp(-r * (mat - t)) * strike * norm.cdf(-d2) - s * norm.cdf(-d1)
    return price
