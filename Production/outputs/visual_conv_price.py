import numpy as np
import matplotlib.pyplot as plt

from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1, f_builder_2, f_builder_3
from price import price_amr_put, price_put_benchmark

s, t0, mat, strike0, r, sigma, n0 = 36, 0, 1, 40, 0.06, 0.2, 500
i = 1
for f in [f_builder_1, f_builder_2, f_builder_3]:
    fig, ax = plt.subplots(2, 1, figsize=(8, 9))
    p_bm = price_put_benchmark(mat, 5 * 10 ** 4, r, s, sigma, strike0)

    for n in [10, 50, 400]:
        _, _, b_history  = b_fixed_point(t0,mat, strike0,r,sigma,n, f, return_history=True, tol=0.0001)

        # "same color as the first plot" = first color in the cycle, regardless of cycle state
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        color1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

        sweeps = b_history[1:]
        sweeps_num = np.arange(1,len(sweeps)+1)
        alphas = np.linspace(0.2, 1, 4)  # earliest iter most opaque, fading to converged

        sweeps_res = []
        for k, b in enumerate(sweeps):
            p0 = price_amr_put(t0, mat, s, strike0, r, sigma, b)
            sweeps_res.append(p0/p_bm)

        ax[0].plot(sweeps_num, sweeps_res, label=fr'$steps = {n}$')

    ax[0].hlines(1, color='black', xmin = 1, xmax=50, label = "benchmark", linestyle='dashed')
    ax[0].set_xlabel("# Iterations")
    ax[0].set_ylabel(r"Relative Price ")
    ax[0].set_ylim(bottom=0.9775,top=1.1625)
    ax[0].set_xlim(left=-1, right=51)
    ax[0].set_title("Visual Convergence of Price | Picard Iteration | # of Iterations")
    ax[0].legend()

    n_grid = np.concatenate([np.arange(10, 200, 4), np.arange(300,1100,100)])
    price_grid = []
    price_grid_num = []
    for n in n_grid:
        boundary, _ = b_fixed_point(t0, mat, strike0, r, sigma, n, f, return_history=False, tol=0.0001)
        boundary_num, _ = b_num_solv(t0, mat, strike0, r, sigma, n, f)
        p0 = price_amr_put(t0, mat, s, strike0, r, sigma, boundary)
        p1 = price_amr_put(t0, mat, s, strike0, r, sigma, boundary_num)

        price_grid.append(p0/p_bm)
        price_grid_num.append(p1/p_bm)

    ax[1].plot(n_grid, price_grid, label = 'Picard Iteration')
    ax[1].plot(n_grid, price_grid_num, label = 'Numerical Solver', linestyle='dotted', linewidth=5)

    ax[1].hlines(1, color='black', xmin = 10, xmax=1000, label = "benchmark", linestyle='dashed')
    ax[1].set_xlabel("#steps")
    ax[1].set_ylabel(r"Relative Price")
    #ax[1].set_ylim(bottom=0.9775,top=1.1625)
    ax[1].set_xlim(left=-5, right=105)
    ax[1].set_title("Visual Convergence of Price | Both Methods | # of Steps")
    ax[1].legend()
    fig.suptitle(f'Method {i}')
    plt.tight_layout()
    plt.savefig(f'conv_price_method_{i}.png')
    plt.close()

    i += 1


