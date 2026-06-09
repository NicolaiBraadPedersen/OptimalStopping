import numpy as np
from types import SimpleNamespace
from scipy.optimize import newton
from pandas import Timestamp as ts


def b_num_solv(t, mat, strike, r, sigma, n, f_builder, verbose=False):
    """
    Solve for the boundary backward from T to t.

    f_builder(ctx) -> f(x):  given the per-step context, returns the residual
                             to root-solve. ctx exposes everything the loop knows.
    """
    t0 = ts.now()
    t_delta = (mat-t) / n
    t_set = np.linspace(t, mat, n + 1)[:-1][::-1]

    b_final = [strike]
    b_guess = strike

    for idx, i in enumerate(t_set):
        n_steps = int(round((mat - i) / t_delta))
        future_times = np.linspace(i, mat, n_steps + 1)

        b_s_nodes = np.empty(n_steps + 1)
        b_s_nodes[0] = strike*r/2
        b_s_nodes[1:] = strike

        ctx = SimpleNamespace(
            t=t, i=i, mat=mat, strike=strike, r=r, sigma=sigma, n=n,
            idx=idx, t_delta=t_delta,
            future_times=future_times, b_s_nodes=b_s_nodes,
            b_final=b_final,
            method = 'num_solver'
        )

        f = f_builder(ctx)
        b_i = newton(f, b_guess, maxiter=100)

        b_final.insert(0, b_i)
        b_guess = b_i

        if verbose:
            print(idx)

    time_to_conv = (ts.now() - t0).total_seconds()
    return b_final, time_to_conv


def b_fixed_point(t, mat, strike, r, sigma, n, f_builder,
                      tol=0.01, max_iter=200, return_history=False, verbose=False):
    """
    Fixed-point (Picard) solve for the boundary over the full time grid.
    Reuses the same f_builder(ctx) -> f(x) contract as b_num_solv, but ctx's
    future boundary comes from the PREVIOUS full iterate, and f is evaluated
    (not root-solved): B[j] = f(B_prev[j]).
    """
    t0 = ts.now()
    t_delta = (mat - t) / n
    times = np.linspace(t, mat, n + 1)        # forward: times[0]=t ... times[n]=mat

    B = np.full(n + 1, float(strike))         # initial guess: strike everywhere
    B[n] = strike                             # terminal condition (at mat), fixed

    history = [B.copy()] if return_history else None
    n_iter, max_rel = 0, np.inf

    for it in range(max_iter):
        B_prev = B
        B = B_prev.copy()

        for j in range(n):                    # interior nodes; j=n (mat) stays fixed
            ti = times[j]
            future_times = times[j:]          # forward, ti ... mat  -> matches num_solv

            b_s_nodes = np.empty(n - j + 1)
            b_s_nodes[0] = strike * r / 2     # lower-limit node, same as num_solv
            b_s_nodes[1:] = B_prev[j + 1:]    # previous iterate at the future nodes

            ctx = SimpleNamespace(
                t=t, i=ti, mat=mat, strike=strike, r=r, sigma=sigma, n=n,
                idx=j, t_delta=t_delta,
                future_times=future_times, b_s_nodes=b_s_nodes,
                b_final=B_prev,
                method='fix_point',
            )

            f = f_builder(ctx)
            B[j] = f(B_prev[j])

        max_rel = np.max(np.abs(B[:n] - B_prev[:n]) / np.abs(B_prev[:n]))
        n_iter = it + 1
        if return_history:
            history.append(B.copy())
        if verbose:
            print(f"iter {n_iter:3d}:  max rel change = {max_rel:.4%}")
        if max_rel < tol:
            break

    time_to_conv = (ts.now() - t0).total_seconds()
    if return_history:
        return B, time_to_conv, history       # <-- was returning history only
    return B, time_to_conv