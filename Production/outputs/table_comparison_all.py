import numpy as np
import pandas as pd
from itertools import product

from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1, f_builder_2
from Production.outputs.price import price_amr_put, price_put_benchmark

####################
### Initial test ###
####################

sigmas = [0.2,0.3,0.4]
rates = [0.02,0.04,0.06]
maturities = [1,2]
stock0 = [36,40,44]
n = 200
t = 0
K = 40

price_dict = {}
i = 0
for sig, r, T, s0 in product(sigmas, rates, maturities, stock0):
    steps_total = n * T

    # method 2 (num_solv) — no max_iter
    boundary1, time1 = b_num_solv(t, T, K, r, sig, steps_total, f_builder_1)
    boundary3, time3 = b_num_solv(t, T, K, r, sig, steps_total, f_builder_2)

    # method 1 (fixed_point) — one entry per max_iter
    m1_f1 = {}
    for it in [20, 40]:
        b, tm = b_fixed_point(t, T, K, r, sig, steps_total, f_builder_1,
                              max_iter=it, tol=0.0001)
        m1_f1[it] = {"price": price_amr_put(t, T, s0, K, r, sig, b), "time": tm}

    m1_f2 = {}
    for it in [3, 5]:
        b, tm = b_fixed_point(t, T, K, r, sig, steps_total, f_builder_2,
                              max_iter=it, tol=0.0001)
        m1_f2[it] = {"price": price_amr_put(t, T, s0, K, r, sig, b), "time": tm}

    p_bm, time_bm = price_put_benchmark(T, 5*10**4, r, s0, sig, K, timed = True)

    key = (sig, r, T, s0)          # n included so nothing gets overwritten
    price_dict[key] = {
        "m1_f1": m1_f1,               # {3: {price, time}, 5: {price, time}}
        "m2_f1": {"price": price_amr_put(t, T, s0, K, r, sig, boundary1), "time": time1},
        "m1_f2": m1_f2,               # {20: {price, time}, 40: {price, time}}
        "m2_f2": {"price": price_amr_put(t, T, s0, K, r, sig, boundary3), "time": time3},
        "bm":    {"price": p_bm, "time": time_bm},
    }
    print(i, i/81)
    i += 1

###################
### Stress Test ###
###################


sigmas = [0.2,0.4,0.6]
rates = [0.01,0.04,0.1]
maturities = [1,10]
stock0 = [30,40,50]
n = 200
t = 0
K = 40

price_dict_stress = {}
i = 0
for sig, r, T, s0 in product(sigmas, rates, maturities, stock0):
    steps_total = n * T

    # method 2 (num_solv) — no max_iter
    boundary1, time1 = b_num_solv(t, T, K, r, sig, steps_total, f_builder_1)
    boundary3, time3 = b_num_solv(t, T, K, r, sig, steps_total, f_builder_2)

    # method 1 (fixed_point) — one entry per max_iter
    m1_f1 = {}
    for it in [20, 70]:
        b, tm = b_fixed_point(t, T, K, r, sig, steps_total, f_builder_1,
                              max_iter=it, tol=0.0001)
        m1_f1[it] = {"price": price_amr_put(t, T, s0, K, r, sig, b), "time": tm}

    m1_f2 = {}
    for it in [3, 20]:
        b, tm = b_fixed_point(t, T, K, r, sig, steps_total, f_builder_2,
                              max_iter=it, tol=0.0001)
        m1_f2[it] = {"price": price_amr_put(t, T, s0, K, r, sig, b), "time": tm}

    key = (sig, r, T, s0)          # n included so nothing gets overwritten
    price_dict_stress[key] = {
        "m1_f1": m1_f1,               # {3: {price, time}, 5: {price, time}}
        "m2_f1": {"price": price_amr_put(t, T, s0, K, r, sig, boundary1), "time": time1},
        "m1_f2": m1_f2,               # {20: {price, time}, 40: {price, time}}
        "m2_f2": {"price": price_amr_put(t, T, s0, K, r, sig, boundary3), "time": time3},
        #"bm":    {"price": price_put_benchmark(T, 5*10**4, r, s0, sig, K), "time": np.nan},
    }
    print(i, i/54)
    i += 1



########################
### For latex layout ###
########################

def fmt_price(x):
    return f"{x:.2f}"


def fmt_stat(x):
    """4 decimals, capped at 5 significant digits."""
    if x == 0:
        return "0.0000"
    int_digits = max(1, int(np.floor(np.log10(abs(x)))) + 1)
    decimals = min(4, max(0, 5 - int_digits))
    return f"{x:.{decimals}f}"


def make_latex_table(price_dict):
    entries = list(price_dict.values())
    first = entries[0]

    has_bm = "bm" in first                      # auto-detect benchmark presence
    its_f1 = sorted(first["m1_f1"].keys())      # auto-detect Picard iteration counts
    its_f2 = sorted(first["m1_f2"].keys())

    columns = [
        ("m2_f1", lambda d: d["m2_f1"]),                               # M1, Numerical Solve
        (f"m1_f1_{its_f1[0]}", lambda d, k=its_f1[0]: d["m1_f1"][k]),  # M1, Picard lo
        (f"m1_f1_{its_f1[1]}", lambda d, k=its_f1[1]: d["m1_f1"][k]),  # M1, Picard hi
        ("m2_f2", lambda d: d["m2_f2"]),                               # M2, Numerical Solve
        (f"m1_f2_{its_f2[0]}", lambda d, k=its_f2[0]: d["m1_f2"][k]),  # M2, Picard lo
        (f"m1_f2_{its_f2[1]}", lambda d, k=its_f2[1]: d["m1_f2"][k]),  # M2, Picard hi
    ]

    if has_bm:
        bm_prices = np.array([d["bm"]["price"] for d in entries])
        bm_times  = np.array([d["bm"]["time"]  for d in entries])

    stats = {}
    for name, get in columns:
        prices = np.array([get(d)["price"] for d in entries])
        times  = np.array([get(d)["time"]  for d in entries])
        stats[name] = {"sum_price": prices.sum(), "sum_time": times.sum()}
        if has_bm:
            err = prices - bm_prices
            stats[name]["mean_err"] = err.mean() * 100
            stats[name]["max_abs_err"] = np.abs(err).max() * 100

    col_order = [c for c, _ in columns]

    def row(label, key, fmt, bm_cell=""):
        cells = " & ".join(fmt(stats[c][key]) for c in col_order)
        return f"{label} & {cells} & {bm_cell} \\\\" if has_bm else f"{label} & {cells} \\\\"

    ncols = 8 if has_bm else 7
    colspec = "l ccc ccc c" if has_bm else "l ccc ccc"
    bm_header = r" & \multirow{3}{*}{BM}" if has_bm else ""
    tail = " & " if has_bm else " "

    lines = [
        rf"\begin{{tabular}}{{{colspec}}}",
        rf"\multicolumn{{{ncols}}}{{c}}{{\textbf{{Portfolio of American Put Options}}}} \\",
        r"\midrule",
        rf" & \multicolumn{{3}}{{c}}{{Method 1}} & \multicolumn{{3}}{{c}}{{Method 2}}{bm_header} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r" & \multirow{2}{*}{\shortstack{Numerical\\Solve}} & \multicolumn{2}{c}{Picard Iteration}"
        rf" & \multirow{{2}}{{*}}{{\shortstack{{Numerical\\Solve}}}} & \multicolumn{{2}}{{c}}{{Picard Iteration}}{tail}\\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){6-7}",
        rf" &  & {its_f1[0]} & {its_f1[1]} &  & {its_f2[0]} & {its_f2[1]}{tail}\\",
        r"\midrule",
        row(r"$\Sigma$ price", "sum_price", fmt_price,
            bm_cell=fmt_price(bm_prices.sum()) if has_bm else ""),
        row(r"$\Sigma$ time (seconds)", "sum_time", fmt_stat,
            bm_cell=fmt_stat(bm_times.sum()) if has_bm else ""),
    ]
    if has_bm:
        lines += [
            row(r"$\mathbb{E}[p - p_{\mathrm{BM}}]$", "mean_err", fmt_stat),
            row(r"$\max\left|p - p_{\mathrm{BM}}\right|$", "max_abs_err", fmt_stat),
        ]
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


print(make_latex_table(price_dict))
print(make_latex_table(price_dict_stress))   # no flag needed anymore