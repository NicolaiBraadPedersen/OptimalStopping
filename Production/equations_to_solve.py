import numpy as np
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator

def f_builder_1(ctx):
    strike, i, mat, r, sigma = ctx.strike, ctx.i, ctx.mat, ctx.r, ctx.sigma
    t_delta = ctx.t_delta
    ft  = ctx.future_times      # n_steps+1, includes lower limit u = i
    bsn = ctx.b_s_nodes         # n_steps+1, = [rK/2, *b_final]   <-- not ctx.b_final
    mtd = ctx.method
    def f(x):
        g = g_vec_1(ft, i, bsn, x, strike, r, sigma)
        integral = t_delta * (g[0]/2 + g[1:-1].sum() + g[-1]/2)
        out = strike - integral - price_euro_put(i, x, mat, strike, r, sigma)
        if mtd == 'num_solver':
            out = out - x
        return out
    return f

def g_vec_1(u, t, b_u, b_t, strike, r, sigma):
    u   = np.asarray(u, float)
    b_u = np.asarray(b_u, float)        # <-- add this
    out = np.empty_like(u)
    dt  = u - t
    diag = dt <= 0
    out[diag] = r * strike / 2.0
    m = ~diag
    sq = sigma * np.sqrt(dt[m])
    phi = norm.cdf((np.log(b_u[m] / b_t) - (r - 0.5*sigma**2) * dt[m]) / sq)
    out[m] = r * strike * np.exp(-r * dt[m]) * phi
    return out

def f_builder_2(ctx):
    """Utilize put-call parity..."""
    strike, i, mat, r, sigma = ctx.strike, ctx.i, ctx.mat, ctx.r, ctx.sigma
    t_delta = ctx.t_delta
    ft = ctx.future_times  # n_steps+1, includes lower limit u = i
    bsn = ctx.b_s_nodes  # n_steps+1, = [rK/2, *b_final]   <-- not ctx.b_final
    mtd = ctx.method

    def f(x):
        g = g_vec_2(ft, i, bsn, x, strike, r, sigma)
        integral = t_delta * (g[0]/2 + g[1:-1].sum() + g[-1]/2)
        d_m = (np.log(x/strike) + (r-sigma**2/2)*(mat - i)) / (sigma*np.sqrt(mat-i))
        d_p = (np.log(x/strike) + (r+sigma**2/2)*(mat - i)) / (sigma*np.sqrt(mat-i))
        out = (strike*(1-np.exp(-r*(mat-i))) + strike*np.exp(-r*(mat-i)) * norm.cdf(d_m) - integral) / norm.cdf(d_p)
        if mtd == 'num_solver':
            out = out - x
        return out
    return f


def g_vec_2(u, t, b_u, b_t, strike, r, sigma):
    u   = np.asarray(u, float)
    b_u = np.asarray(b_u, float)        # <-- add this
    out = np.empty_like(u)
    dt  = u - t
    diag = dt <= 0
    out[diag] = r * strike / 2.0
    m = ~diag
    sq = sigma * np.sqrt(dt[m])
    phi = norm.cdf((np.log(b_u[m] / b_t) - (r - 0.5*sigma**2) * dt[m]) / sq)
    out[m] = r * strike * np.exp(-r * dt[m]) * phi
    return out

def f_builder_3(ctx):
    """Smooth-pasting (density) form, w = sqrt(v) substitution."""
    strike, i, mat, r, sigma = ctx.strike, ctx.i, ctx.mat, ctx.r, ctx.sigma
    ft  = ctx.future_times      # n_steps+1, includes lower limit u = i
    bsn = ctx.b_s_nodes         # n_steps+1, = [rK/2, *b_final]
    mtd = ctx.method

    w = np.sqrt(ft - i)         # same time nodes, transformed to w = sqrt(v)

    def f(x):
        g = g_vec_3(ft, i, bsn, x, strike, r, sigma)
        integral = np.sum((g[:-1] + g[1:]) / 2 * np.diff(w))   # non-uniform trapezoid
        d_p = (np.log(x/strike) + (r + sigma**2/2)*(mat - i)) / (sigma*np.sqrt(mat - i))
        out = 2*r*strike*integral / norm.cdf(d_p)
        if mtd == 'num_solver':
            out = out - x
        return out
    return f


def g_vec_3(u, t, b_u, b_t, strike, r, sigma):
    u   = np.asarray(u, float)
    b_u = np.asarray(b_u, float)
    out = np.empty_like(u)
    dt  = u - t                          # = v = w^2
    diag = dt <= 0
    out[diag] = norm.pdf(0.0) / sigma    # w=0 limit: e^0 * phi(0) / sigma
    m = ~diag
    sq = sigma * np.sqrt(dt[m])          # = sigma * w
    phi = norm.pdf((np.log(b_u[m] / b_t) - (r - 0.5*sigma**2) * dt[m]) / sq)
    out[m] = np.exp(-r * dt[m]) / sigma * phi
    return out

def price_euro_put(t,s,mat,strike,r,sigma):
    d1 = 1 / (sigma * np.sqrt(mat - t)) * (np.log(s / strike) + (r + 0.5 * sigma ** 2) * (mat - t))
    d2 = d1 - sigma * np.sqrt(mat - t)
    price = np.exp(-r * (mat - t)) * strike * norm.cdf(-d2) - s * norm.cdf(-d1)
    return price