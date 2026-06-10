import numpy as np
import matplotlib.pyplot as plt

from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1 #, f_builder_2, f_builder_3
from price import price_amr_put, price_put_benchmark

s, t0, mat, strike0, r0, sigma0, n0 = 36, 0, 1, 40, 0.06, 0.2, 500

for f in [f_builder_1]:  # , f_builder_2, f_builder_3]:
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 9))

    for sigma, r in zip([0.2, 0.2, 0.4, 0.4], [0.06, 0.03, 0.06, 0.03]):
        p_bm = price_put_benchmark(mat, 5*10**4, r, s, sigma, strike0)
        boundary, _, b_history = b_fixed_point(
            t0, mat, strike0, r, sigma, n0, f, return_history=True, tol=0.0001
        )

        # --- top panel: relative-price convergence (your existing plot) ---
        sweeps = b_history[1:]
        sweeps_num = np.arange(1, len(sweeps) + 1)

        sweeps_res = []
        for b in sweeps:
            p0 = price_amr_put(t0, mat, s, strike0, r, sigma, b)
            sweeps_res.append(p0 / p_bm)

        label = fr'$(\sigma, r) = {sigma}, {r}$'
        line, = ax_top.plot(sweeps_num, sweeps_res, label=label)
        color = line.get_color()  # reuse this exact color in the bottom panel
        ax_top.plot(sweeps_num[-1], sweeps_res[-1],
                   marker='o', color=color, markersize=12,
                   markeredgecolor='black', markeredgewidth=0.8, zorder=5)

        # --- bottom panel: converged boundary for this loop ---
        tau = np.linspace(t0, mat, len(boundary))
        ax_bot.plot(tau, boundary[::-1], color=color, label=label)

    # top panel cosmetics
    ax_top.hlines(1, xmin=1, xmax=50, color='black', linestyle='dotted', label="benchmark")
    ax_top.set_xlabel("# Iterations")
    ax_top.set_ylabel(r"Relative Price")
    ax_top.set_ylim(bottom=0.9775, top=1.1625)
    # ax_top.set_xlim(left=-1, right=51)
    ax_top.set_title("Visual Convergence of Price | Picard Iteration")
    ax_top.legend()

    # bottom panel cosmetics
    ax_bot.set_xlabel("$T-t$")
    ax_bot.set_ylabel("Boundary $b(t)$")
    ax_bot.set_title("Converged Early-Exercise Boundary | Picard Iteration")
    ax_bot.legend()
    fig.suptitle(f'Method 1')

    plt.tight_layout()
    plt.savefig(fr'.png\conv_params_method_1.png')
    plt.close()


