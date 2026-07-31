#!/usr/bin/env python3
"""Analyze on-node scaling from run_scaling*.sh CSVs.

The CSVs label their `backend` column "<name>-t<T>" (e.g. native-t8), so a single
file carries every thread count for one backend. Each row is one case identified
by (operation, size); this reports the case's throughput (the CSV `gflops`
column -- GFLOP/s for GEMM/factor, elements/ns or nnz/ns for sweeps/solves),
speedup vs the smallest thread count, and parallel efficiency, and -- with
--plot -- draws speedup-vs-threads curves (with an ideal-linear reference).

Series are keyed by (operation, size) so multiple operations, matrix shapes, or
sparse solvers that share a `size` no longer collide (#297 phase 4). A generic
`operation == "gemm"` with several sizes still splits into one series per size,
so the legacy GEMM CSVs read unchanged.

Examples:
    ./analyze_scaling.py data/scaling_*.csv
    ./analyze_scaling.py data/scaling_sparse.csv --plot data/sparse.png
    ./analyze_scaling.py data/scaling_ewise.csv --op ewise-vec   # filter series

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
    """Return {backend: {(operation, size): {threads: gflops}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    m = LABEL_RE.match(row["backend"])
                    if not m or row.get("gflops", "") == "":
                        continue
                    key = (row.get("operation", "?"), int(row["size"]))
                    data[m["name"]][key][int(m["t"])] = float(row["gflops"])
        except OSError as exc:
            sys.exit(f"error: cannot read {path}: {exc}")
        except (KeyError, ValueError) as exc:
            sys.exit(f"error: {path}: malformed CSV ({exc})")
    return data


def series_label(op, size):
    # If the operation already encodes its shape (contains a digit), the size is
    # redundant; otherwise append N=<size> to disambiguate (e.g. plain "gemm").
    return op if any(ch.isdigit() for ch in op) else f"{op} N={size}"


def bitexact_warn(paths):
    """Warn (do not fail) if any row carries an explicit bitexact=0 column."""
    bad = 0
    for path in paths:
        try:
            with open(path, newline="") as fh:
                r = csv.DictReader(fh)
                if r.fieldnames and "bitexact" in r.fieldnames:
                    for row in r:
                        if row.get("bitexact", "") == "0":
                            bad += 1
        except OSError:
            pass
    if bad:
        print(f"WARNING: {bad} row(s) reported bitexact=0 -- those speedups are invalid.\n")


def print_tables(data):
    for name in sorted(data):
        print(f"\n== {name} ==")
        for (op, size) in sorted(data[name]):
            by_t = data[name][(op, size)]
            base_t = min(by_t)                   # baseline = smallest thread count present
            base = by_t.get(base_t)
            print(f"  {series_label(op, size)}")
            print(f"    {'threads':>7} {'throughput':>11} {'speedup':>8} {'efficiency':>11}")
            for t in sorted(by_t):
                g = by_t[t]
                sp = g / base if base else float("nan")
                eff = sp / (t / base_t) if base else float("nan")
                print(f"    {t:>7} {g:>11.3f} {sp:>7.2f}x {100.0 * eff:>10.1f}%")


def make_plot(data, out, op_filter=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_sp, ax_g) = plt.subplots(1, 2, figsize=(11, 4.5))
    all_ts = set()
    plotted = 0
    for name in sorted(data):
        for (op, size) in sorted(data[name]):
            if op_filter and op_filter not in op:
                continue
            by_t = data[name][(op, size)]
            ts = sorted(by_t)
            if len(ts) < 2:
                continue
            all_ts.update(ts)
            base_g = by_t[ts[0]]
            lbl = f"{name}: {series_label(op, size)}"
            ax_sp.plot(ts, [by_t[t] / base_g for t in ts], marker="o", label=lbl)
            ax_g.plot(ts, [by_t[t] for t in ts], marker="o", label=lbl)
            plotted += 1
    if not plotted:
        sys.exit("no series to plot (check --op filter / input)")
    ts = sorted(all_ts)
    base_t = ts[0]
    ax_sp.plot(ts, [t / base_t for t in ts], "k--", alpha=0.5, label="ideal (linear)")

    ax_sp.set(title="Speedup vs threads", xlabel="threads", ylabel="speedup vs smallest T")
    ax_g.set(title="Throughput vs threads", xlabel="threads", ylabel="throughput (CSV gflops col)")
    for ax in (ax_sp, ax_g):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("MTL5 on-node scaling (#297)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="On-node scaling analysis (#297).")
    ap.add_argument("csv", nargs="+", help="scaling_*.csv file(s)")
    ap.add_argument("--plot", metavar="PNG", help="write a speedup/throughput scaling plot")
    ap.add_argument("--op", metavar="SUBSTR", help="only plot series whose operation contains SUBSTR")
    args = ap.parse_args()
    bitexact_warn(args.csv)
    data = load(args.csv)
    if not data:
        sys.exit("no scaling data found (expected backend labels like 'native-t4')")
    print_tables(data)
    if args.plot:
        make_plot(data, args.plot, args.op)


if __name__ == "__main__":
    main()
