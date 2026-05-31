#!/usr/bin/env python3
"""Plot MTL5 benchmark results (GFLOP/s vs N) from bench_all CSV output.

bench_all writes a CSV with columns:

    operation, backend, size, median_ns, min_ns, max_ns, mean_ns,
    stddev_ns, gflops, iterations

This script plots one curve per (backend) for each operation. Pass several
CSVs (e.g. an OpenBLAS run and an MKL run) to overlay them -- each file's
series are labelled with the file's stem (or --labels) so you get a
native / OpenBLAS / MKL comparison in one figure.

Examples:
    # One figure per operation in a CSV, written next to it
    ./plot_results.py results.csv

    # Overlay OpenBLAS vs MKL for gemm, save a single PNG
    ./plot_results.py blas_openblas.csv blas_mkl.csv \\
        --labels openblas,mkl --op gemm --out gemm_gflops.png

    # Plot wall-clock time instead of GFLOP/s, log-log
    ./plot_results.py results.csv --metric median_ns --logx --logy

Only the Python standard library plus matplotlib are required (no pandas).
This is benchmark *tooling*; it is not part of the MTL5 C++ library or its
(separate) Python bindings.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; override-able via MPLBACKEND
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("error: matplotlib is required (pip install matplotlib)")

# CSV column -> human-readable y-axis label
METRICS = {
    "gflops": "GFLOP/s",
    "median_ns": "median time (ns)",
    "mean_ns": "mean time (ns)",
    "min_ns": "min time (ns)",
}


def load(path):
    """Return {(operation, backend): [(size, {col: float})...]} sorted by size."""
    series = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                size = int(row["size"])
                vals = {k: float(row[k]) for k in METRICS if k in row and row[k] != ""}
            except (KeyError, ValueError) as exc:
                sys.exit(f"error: {path}: malformed row {row!r} ({exc})")
            series[(row["operation"], row["backend"])].append((size, vals))
    for key in series:
        series[key].sort(key=lambda t: t[0])
    return series


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot MTL5 bench_all CSV results.")
    ap.add_argument("csv", nargs="+", help="one or more bench_all CSV files")
    ap.add_argument("--op", help="only plot this operation (default: all, one subplot each)")
    ap.add_argument("--metric", default="gflops", choices=sorted(METRICS),
                    help="y-axis metric (default: gflops)")
    ap.add_argument("--labels", help="comma-separated label per CSV (default: file stem)")
    ap.add_argument("--out", help="output image path (default: <first-csv>_<metric>.png)")
    ap.add_argument("--logx", action="store_true", help="log-scale the size axis")
    ap.add_argument("--logy", action="store_true", help="log-scale the metric axis")
    ap.add_argument("--show", action="store_true", help="display interactively instead of saving")
    args = ap.parse_args(argv)

    if args.labels:
        labels = args.labels.split(",")
        if len(labels) != len(args.csv):
            ap.error(f"--labels has {len(labels)} entries but {len(args.csv)} CSVs given")
    else:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.csv]
    multi = len(args.csv) > 1

    loaded = [(lbl, load(p)) for lbl, p in zip(labels, args.csv)]

    # Determine the set of operations to plot.
    ops = []
    for _, series in loaded:
        for op, _backend in series:
            if op not in ops:
                ops.append(op)
    if args.op:
        if args.op not in ops:
            sys.exit(f"error: operation '{args.op}' not found in input (have: {', '.join(ops)})")
        ops = [args.op]

    ncols = 1 if len(ops) == 1 else 2
    nrows = (len(ops) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    ylabel = METRICS[args.metric]

    for idx, op in enumerate(ops):
        ax = axes[idx // ncols][idx % ncols]
        for lbl, series in loaded:
            for (sop, backend), points in sorted(series.items()):
                if sop != op:
                    continue
                pts = [(s, v[args.metric]) for s, v in points if args.metric in v]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                name = f"{backend} ({lbl})" if multi else backend
                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, label=name)
        ax.set_title(op)
        ax.set_xlabel("N (matrix / vector dimension)")
        ax.set_ylabel(ylabel)
        if args.logx:
            ax.set_xscale("log", base=2)
        if args.logy:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    # Blank any unused subplots.
    for j in range(len(ops), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"MTL5 benchmark: {ylabel} vs N", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.show:
        matplotlib.use("TkAgg", force=True)  # best-effort interactive
        plt.show()
    else:
        out = args.out or f"{os.path.splitext(args.csv[0])[0]}_{args.metric}.png"
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
