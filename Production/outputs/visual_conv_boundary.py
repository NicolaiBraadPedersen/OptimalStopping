import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1, f_builder_2, f_builder_3
from utils.crr_method_benchmark import crr_put_bound

t, mat, strike, r, sigma, n = 0, 1, 40, 0.06, 0.2, 1000

n_bm = 5*10**4
b_benchmark = crr_put_bound(mat-t, n_bm, r, 40, sigma, strike)[::-1]
times_benchmark = np.linspace(t, mat, len(b_benchmark))

for f in [f_builder_1]:#, f_builder_2, f_builder_3]:
    boundary, _, b_history  = b_fixed_point(t,mat, strike,r,sigma,n, f, return_history=True, tol=0.0001)
    boundary_num, _ = b_num_solv(t,mat,strike,r,sigma,n,f)
    times = np.linspace(t, mat, len(boundary))  # same grid the solver used

    # "same color as the first plot" = first color in the cycle, regardless of cycle state
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    color1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    sweeps = b_history[1:]
    alphas = np.linspace(0.2, 1, 4)  # earliest iter most opaque, fading to converged

    fig, ax = plt.subplots()
    i = 0
    for k, b in enumerate(sweeps):
        if k in [1,2,10,len(sweeps)-1]:
            ax.plot(times, b[::-1], color=color, alpha=alphas[i], label=f"Picard: iter {k}")
            i += 1

    ax.plot(times, boundary_num[::-1], color=color1, label='Numerical Solve', linestyle='dotted', linewidth=5)
    ax.plot(times_benchmark, b_benchmark, color='black', label="benchmark", alpha = 0.6, linewidth=2)

    ax.set_xlabel("T-t")
    ax.set_ylabel(r"boundary $b(t)$")
    ax.set_title("Picard iteration history")
    ax.legend()
    plt.show()


