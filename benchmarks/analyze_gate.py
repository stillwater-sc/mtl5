#!/usr/bin/env python3
"""Analyze the native-fast benchmark against OpenBLAS -- the epic #82 gate.

Reads bench_all CSVs (columns: operation,backend,size,...,gflops,...) and, for
each operation and size, reports the native-fast GFLOP/s as a **percentage of
OpenBLAS** and, optionally, as a **percentage of FMA peak** (pass --peak-gflops
for your machine: cores_used * freq_GHz * fma_units * lanes * 2).

It also doubles as the optional performance guard (`--gate`): exit non-zero if
native-fast GEMM falls below a threshold fraction of OpenBLAS for sizes at or
above a floor. This is deliberately NOT wired into the per-push CI -- shared CI
runners make absolute perf gates flaky -- but it is handy locally and in an
opt-in workflow.

Examples:
    # Comparison table (% of OpenBLAS), all ops
    ./analyze_gate.py data/blas_sweep_native-fast.csv data/blas_sweep_openblas.csv

    # Add % of FMA peak (e.g. 1 P-core, 4.0 GHz, 2 FMA units, 4 fp64 lanes)
    ./analyze_gate.py data/blas_sweep_*.csv --peak-gflops 64

    # Gate: fail if GEMM native-fast < 80% of OpenBLAS for N >= 256
    ./analyze_gate.py data/blas_sweep_native-fast.csv data/blas_sweep_openblas.csv \\
        --gate --op gemm --threshold 0.80 --min-size 256

Standard library only (no pandas). Benchmark tooling, not part of the library.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict


def load(paths):
    """Return {operation: {backend: {size: gflops}}} merged across CSV files."""
    data: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    op, backend = row["operation"], row["backend"]
                    size = int(row["size"])
                    g = row.get("gflops", "")
                    if g == "":
                        continue
                    data[op][backend][size] = float(g)
        except OSError as exc:
            sys.exit(f"error: cannot read {path}: {exc}")
        except (KeyError, ValueError) as exc:
            sys.exit(f"error: {path}: malformed CSV ({exc})")
    return data


def pick(backends, *candidates):
    """First backend label present, matched case-insensitively."""
    low = {b.lower(): b for b in backends}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def print_table(data, peak):
    for op in sorted(data):
        backends = data[op]
        nf = pick(backends, "native-fast", "native_fast")
        ob = pick(backends, "openblas", "blas")
        if nf is None:
            continue
        print(f"\n== {op} ==")
        hdr = f"{'N':>6} {'native-fast':>12}"
        if ob:
            hdr += f" {'openblas':>12} {'% OpenBLAS':>11}"
        if peak:
            hdr += f" {'% FMA peak':>11}"
        print(hdr)
        for n in sorted(backends[nf]):
            g = backends[nf][n]
            line = f"{n:>6} {g:>12.2f}"
            if ob and n in backends[ob] and backends[ob][n] > 0:
                line += f" {backends[ob][n]:>12.2f} {100.0 * g / backends[ob][n]:>10.1f}%"
            elif ob:
                line += f" {'--':>12} {'--':>11}"
            if peak:
                line += f" {100.0 * g / peak:>10.1f}%"
            print(line)


def run_gate(data, op, threshold, min_size):
    backends = data.get(op, {})
    nf = pick(backends, "native-fast", "native_fast")
    ob = pick(backends, "openblas", "blas")
    if nf is None or ob is None:
        sys.exit(f"gate: need both native-fast and openblas data for '{op}' "
                 f"(have: {sorted(backends)})")
    failures = []
    checked = 0
    for n in sorted(backends[nf]):
        if n < min_size or n not in backends[ob] or backends[ob][n] <= 0:
            continue
        checked += 1
        frac = backends[nf][n] / backends[ob][n]
        status = "ok" if frac >= threshold else "LOW"
        print(f"  N={n:>5}  native-fast={backends[nf][n]:8.2f}  "
              f"openblas={backends[ob][n]:8.2f}  {100*frac:5.1f}%  [{status}]")
        if frac < threshold:
            failures.append((n, frac))
    if checked == 0:
        sys.exit(f"gate: no comparable sizes >= {min_size} for '{op}'")
    if failures:
        worst = min(f for _, f in failures)
        print(f"\nGATE FAIL: {op} below {threshold*100:.0f}% of OpenBLAS at "
              f"{len(failures)}/{checked} sizes (worst {worst*100:.1f}%).")
        return 1
    print(f"\nGATE PASS: {op} >= {threshold*100:.0f}% of OpenBLAS at all "
          f"{checked} sizes (N >= {min_size}).")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Native-fast vs OpenBLAS gate analyzer.")
    ap.add_argument("csv", nargs="+", help="bench_all CSV file(s) to merge")
    ap.add_argument("--peak-gflops", type=float, default=None,
                    help="machine FMA peak (GFLOP/s) for the %% FMA peak column")
    ap.add_argument("--gate", action="store_true", help="run the pass/fail perf gate")
    ap.add_argument("--op", default="gemm", help="operation to gate (default: gemm)")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="min native-fast/openblas fraction (default: 0.80)")
    ap.add_argument("--min-size", type=int, default=256,
                    help="only gate sizes >= this (default: 256)")
    args = ap.parse_args()
    if args.peak_gflops is not None and args.peak_gflops <= 0:
        ap.error("--peak-gflops must be > 0")

    data = load(args.csv)
    if args.gate:
        print(f"Gate: {args.op} native-fast >= {args.threshold*100:.0f}% of "
              f"OpenBLAS for N >= {args.min_size}")
        sys.exit(run_gate(data, args.op, args.threshold, args.min_size))
    print_table(data, args.peak_gflops)


if __name__ == "__main__":
    main()
