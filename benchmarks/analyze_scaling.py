#!/usr/bin/env python3
"""Analyze multi-core GEMM scaling (#108) from run_scaling.sh CSVs.

The CSVs label their `backend` column "<name>-t<T>" (e.g. native-fast-t8), so a
single file carries every thread count for one backend. For each backend and
size this reports GFLOP/s, speedup vs T=1, and parallel efficiency
(speedup / T), and -- with --plot -- draws speedup-vs-threads curves (with an
ideal-linear reference) at the largest size.

Examples:
    ./analyze_scaling.py data/gemm_scaling_*.csv
    ./analyze_scaling.py data/gemm_scaling_*.csv --plot data/gemm_scaling.png

Standard library only (matplotlib only for --plot). Benchmark tooling.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict

LABEL_RE = re.compile(r"^(?P<name>.+)-t(?P<t>\d+)$")


def load(paths):
    """Return {backend: {size: {threads: gflops}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    m = LABEL_RE.match(row["backend"])
                    if not m or row.get("gflops", "") == "":
                        continue
                    data[m["name"]][int(row["size"])][int(m["t"])] = float(row["gflops"])
        except OSError as exc:
            sys.exit(f"error: cannot read {path}: {exc}")
        except (KeyError, ValueError) as exc:
            sys.exit(f"error: {path}: malformed CSV ({exc})")
    return data


def print_tables(data):
    for name in sorted(data):
        print(f"\n== {name} ==")
        for size in sorted(data[name]):
            by_t = data[name][size]
            base_t = min(by_t)                   # baseline = smallest thread count present
            base = by_t.get(base_t)
            print(f"  N={size}")
            print(f"    {'threads':>7} {'GFLOP/s':>10} {'speedup':>8} {'efficiency':>11}")
            for t in sorted(by_t):
                g = by_t[t]
                sp = g / base if base else float("nan")
                eff = sp / (t / base_t) if base else float("nan")
                print(f"    {t:>7} {g:>10.2f} {sp:>7.2f}x {100.0 * eff:>10.1f}%")


def make_plot(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Largest common size for the headline curve.
    sizes = sorted({s for name in data for s in data[name]})
    if not sizes:
        sys.exit("no data to plot")
    size = sizes[-1]

    fig, (ax_sp, ax_g) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in sorted(data):
        by_t = data[name].get(size)
        if not by_t:
            continue
        ts = sorted(by_t)
        base_g = by_t[ts[0]]                 # speedup relative to this backend's smallest T
        ax_sp.plot(ts, [by_t[t] / base_g for t in ts], marker="o", label=name)
        ax_g.plot(ts, [by_t[t] for t in ts], marker="o", label=name)
    # Ideal line normalized to the same baseline as the speedup curves (the
    # smallest thread count present), so it is correct even if THREADS omits 1.
    all_ts = sorted({t for name in data for t in data[name].get(size, {})})
    if all_ts:
        base_t = all_ts[0]
        ax_sp.plot(all_ts, [t / base_t for t in all_ts], "k--", alpha=0.5, label="ideal (linear)")

    ax_sp.set(title=f"GEMM speedup vs threads (N={size})", xlabel="threads",
              ylabel="speedup vs 1 thread")
    ax_g.set(title=f"GEMM GFLOP/s vs threads (N={size})", xlabel="threads", ylabel="GFLOP/s")
    for ax in (ax_sp, ax_g):
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("MTL5 multi-core GEMM scaling")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Multi-core GEMM scaling analysis.")
    ap.add_argument("csv", nargs="+", help="gemm_scaling_*.csv file(s)")
    ap.add_argument("--plot", metavar="PNG", help="write a speedup/GFLOP-s scaling plot")
    args = ap.parse_args()
    data = load(args.csv)
    if not data:
        sys.exit("no scaling data found (expected backend labels like 'native-fast-t4')")
    print_tables(data)
    if args.plot:
        make_plot(data, args.plot)


if __name__ == "__main__":
    main()
