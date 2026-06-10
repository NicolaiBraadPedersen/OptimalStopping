import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1, f_builder_2, f_builder_3
from price import price_amr_put, price_put_benchmark

s, t0, mat, strike0, r0, sigma0, n0 = 36, 0, 1, 40, 0.06, 0.2, 500

fig, ax = plt.subplots(2, 1, figsize=(8, 9))
i = 0
p_bm = price_put_benchmark(mat, 5 * 10 ** 4, r0, s, sigma0, strike0)

for n in [10, 50, 400]:
    _, _, b_history  = b_fixed_point(t0,mat, strike0,r0,sigma0,n, f_builder_2, return_history=True, tol=0.0001)

    # "same color as the first plot" = first color in the cycle, regardless of cycle state
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    color1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    sweeps = b_history[1:]
    sweeps_num = np.arange(1,len(sweeps)+1)
    alphas = np.linspace(0.2, 1, 4)  # earliest iter most opaque, fading to converged

    sweeps_res = []
    for k, b in enumerate(sweeps):
        p0 = price_amr_put(t0, mat, s, strike0, r0, sigma0, b)
        sweeps_res.append(p0/p_bm)

    ax[0].plot(sweeps_num, sweeps_res, label=fr'$steps = {n}$')

ax[0].hlines(1, color='black', xmin = 1, xmax=10, label = "benchmark", linestyle='dashed')
ax[0].set_xlabel("# Iterations")
ax[0].set_ylabel(r"Relative Price ")
ax[0].set_ylim(bottom=0.9775,top=1.1625)
#ax[0].set_xlim(left=-1, right=10)
ax[0].set_title("Visual Convergence of Price | Picard Iteration | # of Iterations")
ax[0].legend()

i = 1
f = f_builder_2

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
    line, = ax[1].plot(sweeps_num, sweeps_res, label=label)
    color = line.get_color()  # reuse this exact color in the bottom panel
    ax[1].plot(sweeps_num[-1], sweeps_res[-1],
               marker='o', color=color, markersize=12,
               markeredgecolor='black', markeredgewidth=0.8, zorder=5)

# top panel cosmetics
ax[1].hlines(1, xmin=1, xmax=10, color='black', linestyle='dotted', label="benchmark")
ax[1].set_xlabel("# Iterations")
ax[1].set_ylabel(r"Relative Price")
ax[1].set_ylim(bottom=0.9775, top=1.1625)
# ax[i].set_xlim(left=-1, right=51)
ax[1].set_title(f"Visual Convergence of Price | Picard Iteration")
fig.suptitle("Method 2")
ax[1].legend()

plt.tight_layout()
plt.savefig(fr'.png\conv_method_2.png')


