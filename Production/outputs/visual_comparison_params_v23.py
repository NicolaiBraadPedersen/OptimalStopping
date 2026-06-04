import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1, f_builder_2, f_builder_3
from price import price_amr_put, price_put_benchmark

s, t0, mat, strike0, r0, sigma0, n0 = 36, 0, 1, 40, 0.06, 0.2, 500

fig, ax = plt.subplots(2, 1, figsize=(8, 9))
for i, f in enumerate([f_builder_2, f_builder_3]):

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
        line, = ax[i].plot(sweeps_num, sweeps_res, label=label)
        color = line.get_color()  # reuse this exact color in the bottom panel
        ax[i].plot(sweeps_num[-1], sweeps_res[-1],
                   marker='o', color=color, markersize=12,
                   markeredgecolor='black', markeredgewidth=0.8, zorder=5)

    # top panel cosmetics
    ax[i].hlines(1, xmin=1, xmax=50, color='black', linestyle='dotted', label="benchmark")
    ax[i].set_xlabel("# Iterations")
    ax[i].set_ylabel(r"Relative Price")
    ax[i].set_ylim(bottom=0.9775, top=1.1625)
    # ax[i].set_xlim(left=-1, right=51)
    ax[i].set_title(f"Visual Convergence of Price | Picard Iteration | Method {i+2}")
    ax[i].legend()

plt.tight_layout()
plt.show()


