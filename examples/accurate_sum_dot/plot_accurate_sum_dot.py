#!/usr/bin/env python3
"""Plot the CSV output of example_accurate_sum_dot (#611).

The example reproduces Section 6 of Ogita, Rump and Oishi, "Accurate sum and
dot product", SIAM J. Sci. Comput. 26(6), 2005. This script draws its figures:

    accurate-sum-dot-dot2.png      Figure 6.1: Dot2, n = 100, with (5.8)
    accurate-sum-dot-dotk.png      Figures 6.2, 6.3: DotK, K = 3, 4, with (5.11)
    accurate-sum-dot-n2000.png     Figure 6.4: Dot2 and DotK, K = 3..7, n = 2000
    accurate-sum-dot-working.png   the working-precision loops, plain and FMA,
                                   against Dot2 and DotXBLAS, n = 100
    accurate-sum-dot-float.png     the same algorithms in float, n = 100
    accurate-sum-dot-timing.png    Tables 6.4, 6.5: time relative to the plain loop

Examples:
    ./build/examples/example_accurate_sum_dot --out results
    examples/accurate_sum_dot/plot_accurate_sum_dot.py results --out docs/img

Only the Python standard library plus matplotlib are required (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

# Relative errors of exactly 0 cannot go on a log axis; they are drawn here.
FLOOR = 1e-19
YLIM = (5e-20, 1e1)

COLORS = {
    "dot": "#7f7f7f",
    "dot_fma": "#bcbd22",
    "dot2": "#1f77b4",
    "dot_xblas": "#9467bd",
    "dot3": "#ff7f0e",
    "dot4": "#2ca02c",
    "dot5": "#d62728",
    "dot6": "#8c564b",
    "dot7": "#e377c2",
    "exact": "#17becf",
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def column(rows, name):
    return [float(r[name]) for r in rows]


def scatter(ax, rows, alg, label=None, bound=True, size=6):
    cond = column(rows, "cond")
    err = [max(e, FLOOR) for e in column(rows, "err_" + alg)]
    color = COLORS[alg]
    ax.scatter(cond, err, s=size, color=color, alpha=0.6, linewidths=0,
               label=label or alg)
    if bound:
        pairs = sorted(zip(cond, column(rows, "bound_" + alg)))
        ax.plot([c for c, _ in pairs], [min(b, 1e3) for _, b in pairs],
                color=color, linestyle="--", linewidth=1.2)


def setup(ax, title, eps, step=20, top=120):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xticks([10.0 ** e for e in range(0, top + 1, step)])
    ax.set_xlim(0.5, 10.0 ** (top + step / 2))
    ax.set_xlabel("condition number")
    ax.set_ylabel("relative error")
    ax.set_title(title, fontsize=10)
    ax.axhline(eps, color="black", linewidth=0.6, linestyle=":")
    ax.grid(True, which="major", linewidth=0.3, alpha=0.5)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="the directory example_accurate_sum_dot wrote (--out)")
    ap.add_argument("--out", default=".", help="where to write the PNGs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d100 = read_csv(os.path.join(args.data, "dot_double_n100.csv"))
    d2000 = read_csv(os.path.join(args.data, "dot_double_n2000.csv"))
    f100 = read_csv(os.path.join(args.data, "dot_float_n100.csv"))
    timing = read_csv(os.path.join(args.data, "timing.csv"))
    eps_d, eps_f = 2.0 ** -53, 2.0 ** -24

    # Figure 6.1
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    scatter(ax, d100, "dot2", "Dot2", size=8)
    setup(ax, f"Dot2, double, n = 100, {len(d100)} samples; dashed: bound (5.8)", eps_d)
    save(fig, args.out, "accurate-sum-dot-dot2.png")

    # Figures 6.2 and 6.3
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax, k in zip(axes, (3, 4)):
        scatter(ax, d100, f"dot{k}", f"DotK, K = {k}", size=8)
        setup(ax, f"DotK, K = {k}, double, n = 100; dashed: bound (5.11)", eps_d)
    save(fig, args.out, "accurate-sum-dot-dotk.png")

    # Figure 6.4
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for alg, label in [("dot2", "Dot2")] + [(f"dot{k}", f"K = {k}") for k in range(3, 8)]:
        scatter(ax, d2000, alg, label, bound=False, size=5)
    setup(ax, f"Dot2 and DotK, K = 3..7, double, n = 2000, {len(d2000)} samples", eps_d)
    ax.legend(fontsize=8, markerscale=2.5, loc="lower right")
    save(fig, args.out, "accurate-sum-dot-n2000.png")

    # Working precision against twice the working precision
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for alg, label in [("dot", "plain loop"), ("dot_fma", "FMA loop"),
                       ("dot_xblas", "DotXBLAS"), ("dot2", "Dot2")]:
        scatter(ax, d100, alg, label, bound=alg in ("dot", "dot2"), size=5)
    setup(ax, "double, n = 100; dashed: the recursive bound and (5.8)", eps_d, 10, 40)
    ax.legend(fontsize=8, markerscale=2.5, loc="lower right")
    save(fig, args.out, "accurate-sum-dot-working.png")

    # float
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for alg, label in [("dot", "plain loop"), ("dot2", "Dot2"), ("dot3", "K = 3"),
                       ("dot4", "K = 4"), ("exact", "exact")]:
        scatter(ax, f100, alg, label, bound=alg not in ("dot", "exact"), size=5)
    setup(ax, f"float, n = 100, {len(f100)} samples; dashed: (5.8), (5.11)", eps_f, 5, 35)
    ax.set_ylim(1e-10, 1e1)
    ax.legend(fontsize=8, markerscale=2.5, loc="lower right")
    save(fig, args.out, "accurate-sum-dot-float.png")

    # Tables 6.4 and 6.5, at the largest length
    n_max = max(int(r["n"]) for r in timing)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, table, title in [(axes[0], "6.4", "summation (Table 6.4)"),
                             (axes[1], "6.5", "dot product (Table 6.5)")]:
        rows = [r for r in timing if r["table"] == table and int(r["n"]) == n_max]
        names = [r["algorithm"] for r in rows]
        ratios = [float(r["ratio"]) for r in rows]
        ax.barh(names, ratios, color="#1f77b4")
        for i, v in enumerate(ratios):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(ratios) * 1.15)
        ax.set_xlabel("time relative to the plain loop")
        ax.set_title(f"{title}, double, n = {n_max}", fontsize=10)
    save(fig, args.out, "accurate-sum-dot-timing.png")


if __name__ == "__main__":
    main()
