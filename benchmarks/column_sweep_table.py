#!/usr/bin/env python3
"""Tables for docs/benchmarks/column-sweep.md from bench_column_sweep runs (#884).

Reads benchmarks/data/<machine>/column_sweep/<session>/<build>.csv, as
run_column_sweep.sh / .ps1 write them, and prints Markdown:

  * per machine and build, for gemv, trmv and the forward substitution: the
    geometric mean over the shapes of (old sweep time / public time), its
    minimum, and the geometric mean of (row kernel time / public time).
    "Old sweep" is #777's column sweep (sweep1_ns), "public" the public call
    with #884's dispatch (public_ns), "row kernel" the interleaved row kernel
    at the build's default width (rows_ns). Above 1 means the public call is
    faster.
  * every shape where the public call is more than 15% slower than the old
    sweep, with the time it loses.

Usage: benchmarks/column_sweep_table.py [benchmarks/data]
"""
import csv
import glob
import math
import os
import sys

KERNELS = (("gemv", "gemv"), ("trmv_lower", "trmv (lower)"), ("lower_trisolve", "forward substitution"))


def load(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    rows = list(csv.reader(lines))
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:] if len(r) == len(head)]


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
    files = sorted(glob.glob(os.path.join(root, "*", "column_sweep", "*", "*.csv")))
    if not files:
        sys.exit(f"column_sweep_table: no runs under {root}/*/column_sweep/")
    header = "| machine | build | " + " | ".join(
        f"{name}: vs old (min) / vs rows" for _, name in KERNELS) + " |"
    print(header)
    print("|" + "---|" * (2 + len(KERNELS)))
    losses = []
    for path in files:
        machine = path.split(os.sep)[-4]
        build = os.path.basename(path)[:-4]
        data = load(path)
        cells = []
        for key, _ in KERNELS:
            rs = [r for r in data if r["kernel"] == key]
            old = [float(r["sweep1_ns"]) / float(r["public_ns"]) for r in rs]
            rows = [float(r["rows_ns"]) / float(r["public_ns"]) for r in rs]
            cells.append(f"{geo(old):.2f} ({min(old):.2f}) / {geo(rows):.2f}")
            for r, q in zip(rs, old):
                if q < 0.85:
                    losses.append((machine, build, key, r["type"], f"{r['m']}x{r['n']}", q,
                                   float(r["public_ns"]) - float(r["sweep1_ns"])))
        print(f"| {machine} | {build} | " + " | ".join(cells) + " |")
    print()
    print("| machine | build | kernel | type | shape | old / public | lost |")
    print("|---|---|---|---|---|---|---|")
    for m, b, k, t, s, q, d in sorted(losses, key=lambda x: x[5]):
        lost = f"{d / 1000:.1f} us" if d >= 1000 else f"{d:.0f} ns"
        print(f"| {m} | {b} | {k} | {t} | {s} | {q:.2f} | {lost} |")


if __name__ == "__main__":
    main()
