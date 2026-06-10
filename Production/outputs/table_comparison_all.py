import numpy as np
import pandas as pd
from itertools import product

from Production.boundary_methods import b_fixed_point, b_num_solv
from Production.equations_to_solve import f_builder_1, f_builder_2
from Production.outputs.price import price_amr_put, price_put_benchmark


sigmas = [0.2,0.3,0.4]
rates = [0.2,0.4,0.6]
maturities = [1,2,3]
stock0 = [36,40,44]
n = 100
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

    key = (sig, r, T, s0)          # n included so nothing gets overwritten
    price_dict[key] = {
        "m1_f1": m1_f1,               # {3: {price, time}, 5: {price, time}}
        "m2_f1": {"price": price_amr_put(t, T, s0, K, r, sig, boundary1), "time": time1},
        "m1_f2": m1_f2,               # {20: {price, time}, 40: {price, time}}
        "m2_f2": {"price": price_amr_put(t, T, s0, K, r, sig, boundary3), "time": time3},
        "bm":    {"price": price_put_benchmark(T, 5*10**4, r, s0, sig, K), "time": np.nan},
    }
    print(i, i/81)
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
    int_digits = max(1, int(np.floor(np.log10(abs(x)))) + 1)  # digits before decimal point
    decimals = min(4, max(0, 5 - int_digits))
    return f"{x:.{decimals}f}"


def make_latex_table(price_dict):
    # column -> extractor: given one entry of price_dict, return {"price": ..., "time": ...}
    columns = [
        ("m2_f1", lambda d: d["m2_f1"]),        # Method 1, Numerical Solve
        ("m1_f1_20", lambda d: d["m1_f1"][20]), # Method 1, Picard 20
        ("m1_f1_40", lambda d: d["m1_f1"][40]), # Method 1, Picard 40
        ("m2_f2", lambda d: d["m2_f2"]),        # Method 2, Numerical Solve
        ("m1_f2_3", lambda d: d["m1_f2"][3]),   # Method 2, Picard 3
        ("m1_f2_5", lambda d: d["m1_f2"][5]),   # Method 2, Picard 5
    ]

    bm_prices = np.array([d["bm"]["price"] for d in price_dict.values()])

    stats = {}  # col_name -> dict of the four row values
    for name, get in columns:
        prices = np.array([get(d)["price"] for d in price_dict.values()])
        times  = np.array([get(d)["time"]  for d in price_dict.values()])
        err = prices - bm_prices
        stats[name] = {
            "sum_price": prices.sum(),
            "sum_time":  times.sum(),
            "mean_err":  err.mean(),
            "var_err":   err.var(),
        }

    col_order = [c for c, _ in columns]

    def row(label, key, fmt, bm_cell=""):
        cells = " & ".join(fmt(stats[c][key]) for c in col_order)
        return f"{label} & {cells} & {bm_cell} \\\\"

    lines = [
        r"\begin{tabular}{l ccc ccc c}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Method 1} & \multicolumn{3}{c}{Method 2} & \multirow{3}{*}{BM} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r" & \multirow{2}{*}{\shortstack{Numerical\\Solve}} & \multicolumn{2}{c}{Picard Iteration}"
        r" & \multirow{2}{*}{\shortstack{Numerical\\Solve}} & \multicolumn{2}{c}{Picard Iteration} & \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){6-7}",
        r" &  & 20 & 40 &  & 3 & 5 & \\",
        r"\midrule",
        row(r"$\Sigma$ price", "sum_price", fmt_price, bm_cell=fmt_price(bm_prices.sum())),
        row(r"$\Sigma$ time (seconds)", "sum_time", fmt_stat),
        row(r"$\mathbb{E}[p - p_{\mathrm{BM}}]$", "mean_err", fmt_stat),
        row(r"$\mathbb{V}[p - p_{\mathrm{BM}}]$", "var_err", fmt_stat),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


print(make_latex_table(price_dict))