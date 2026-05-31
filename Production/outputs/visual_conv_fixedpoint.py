import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1, f_builder_2, f_builder_3
from utils.crr_method_benchmark import crr_put_bound

t, mat, strike, r, sigma, n = 0, 1, 40, 0.06, 0.2, 200

n_bm = 5*10**4
b_benchmark = crr_put_bound(mat-t, n_bm, r, 40, sigma, strike)[::-1]
b_benchmark_smooth = pd.Series(b_benchmark).rolling(window=100, center=True).mean()
times_benchmark = np.linspace(t, mat, len(b_benchmark))

for f in [f_builder_1]:#, f_builder_2, f_builder_3]:
    boundary, _, b_history  = b_fixed_point(t,mat, strike,r,sigma,200, f, return_history=True, tol=0.0025)
    boundary_num, _ = b_num_solv(t,mat,strike,r,sigma,n,f)
    times = np.linspace(t, mat, len(boundary))  # same grid the solver used

    # "same color as the first plot" = first color in the cycle, regardless of cycle state
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    color1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    sweeps = b_history[1:]
    alphas = np.linspace(0.2, 1, len(sweeps))  # earliest iter most opaque, fading to converged

    fig, ax = plt.subplots()
    for k, (b, a) in enumerate(zip(sweeps, alphas), start=1):
        if k % 3 == 0:
            ax.plot(times, b[::-1], color=color, alpha=a, label=f"Picard: iter {k}")

    ax.plot(times, boundary_num[::-1], color=color1, label='Numerical Solve')
    ax.plot(times_benchmark, b_benchmark_smooth, color='black', linestyle='dashed', label="benchmark")

    ax.set_xlabel("T-t")
    ax.set_ylabel(r"boundary $b(t)$")
    ax.set_title("Picard iteration history")
    ax.legend()
    plt.show()


