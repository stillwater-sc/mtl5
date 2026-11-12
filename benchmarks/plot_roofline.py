#!/usr/bin/env python3
"""Roofline plot of one platform-benchmark session (#737).

Reads the mtl5_bench JSON of one published session and places every entry
the roofline model covers at its arithmetic intensity (flops / bytes) and its
measured GFLOP/s. The roofs are the run's own probes, the same ones
mtl5_bench's %roof is computed against (benchmarks/platform/run.hpp):

    attainable = min(FMA peak of the dtype, bandwidth x intensity)

so a memory-bound entry at 100% of its roof sits on the slope, and a
compute-bound one at 100% sits on its dtype's ceiling.

One panel per thread count of the variant (1 thread and all threads). Left
out, and counted in the caption:

  * integer entries: the model has no integer ceiling (the card shows n/a);
  * entries that record no flops or bytes (Krylov solvers, preconditioners,
    orderings, sparse eigensolvers): they have no intensity to place.

Example:
    benchmarks/plot_roofline.py benchmarks/data/jetson-orin/mtl5_bench/20261111T193624Z \\
        --out docs/img/benchmarks/jetson-orin/roofline.svg

Only the Python standard library plus matplotlib are required. This is
benchmark tooling, not part of the MTL5 library.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# A fixed order and colour per area, so every machine's plot reads the same.
AREAS = ["blas1", "blas2", "blas3", "factor", "eigen", "elementwise",
         "sparse", "direct", "smoother", "mg", "mixed"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000"]
MARKERS = {"f64": "o", "f32": "^"}  # anything else (mixed precision): "s"


def placeable(e: dict) -> bool:
    return (e.get("flops", 0) > 0 and e.get("bytes", 0) > 0 and e.get("gflops", 0) > 0
            and e.get("area") != "integer")


def load_runs(session: Path, variant: str) -> list[dict]:
    runs = []
    for f in session.glob(f"{variant}-t*.json"):
        m = re.fullmatch(rf"{re.escape(variant)}-t(\d+)\.json", f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        runs.append({"threads": int(m.group(1)), "data": d})
    if not runs:
        sys.exit(f"no {variant}-t*.json in {session}")
    return sorted(runs, key=lambda r: r["threads"])


def summarize(run: dict) -> str:
    """One line per run: how far below its roof each side of the ridge sits."""
    p = run["data"]["probes"]
    bw = p["bandwidth_gbytes_per_s"]
    mem, comp = [], []
    for e in run["data"]["results"]:
        if not placeable(e):
            continue
        peak = p["fma_peak_f32_gflops"] if e["dtype"] == "f32" else p["fma_peak_f64_gflops"]
        (mem if bw * e["flops"] / e["bytes"] < peak else comp).append(e["roofline_pct"])

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float("nan")

    def share(xs, lo):
        return sum(x >= lo for x in xs)

    return (f"{run['threads']}T: memory-bound {len(mem)} (median {med(mem):.0f}% of roof, "
            f"{share(mem, 50)} at >=50%), compute-bound {len(comp)} (median {med(comp):.0f}%, "
            f"{share(comp, 50)} at >=50%)")


def panel(ax, run: dict, xlim, ylim) -> None:
    p = run["data"]["probes"]
    bw = p["bandwidth_gbytes_per_s"]
    peaks = {"f64": p["fma_peak_f64_gflops"], "f32": p["fma_peak_f32_gflops"]}

    xs = [xlim[0] * (xlim[1] / xlim[0]) ** (i / 200) for i in range(201)]
    for dtype, style in (("f32", "--"), ("f64", "-")):
        ax.plot(xs, [min(peaks[dtype], bw * x) for x in xs], style, color="#444444", lw=1.4,
                label=f"{dtype} roof: {peaks[dtype]:.1f} GFLOP/s, ridge {peaks[dtype] / bw:.2f} flop/B")
    ax.plot([], [], " ", label=f"bandwidth {bw:.1f} GB/s")

    seen = set()
    for e in run["data"]["results"]:
        if not placeable(e):
            continue
        area = e["area"]
        color = COLORS[AREAS.index(area)] if area in AREAS else "#999999"
        marker = MARKERS.get(e["dtype"], "s")
        hollow = e.get("cache_resident") or e.get("suspect_model")
        ax.scatter(e["flops"] / e["bytes"], e["gflops"], s=30, marker=marker,
                   facecolors="none" if hollow else color, edgecolors=color, linewidths=1.0,
                   label=area if area not in seen else None, zorder=3)
        seen.add(area)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("arithmetic intensity (flop/byte)")
    ax.set_ylabel("GFLOP/s")
    t = run["threads"]
    ax.set_title(f"{t} thread{'s' if t > 1 else ''}")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session", type=Path, help="a published session directory")
    ap.add_argument("--variant", default="native-fast", help="build variant (default native-fast)")
    ap.add_argument("--out", type=Path, required=True, help="output image (.svg or .png)")
    args = ap.parse_args()

    runs = load_runs(args.session, args.variant)
    import matplotlib
    matplotlib.use("Agg")
    # Text stays text in the SVG: smaller files, and the same bytes on every run.
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "mtl5-roofline"
    import matplotlib.pyplot as plt

    # Shared axes across the panels, so a point's move between them is visible.
    ai, gf, roofs = [], [], []
    for r in runs:
        p = r["data"]["probes"]
        roofs += [p["fma_peak_f64_gflops"], p["fma_peak_f32_gflops"]]
        for e in r["data"]["results"]:
            if placeable(e):
                ai.append(e["flops"] / e["bytes"])
                gf.append(e["gflops"])
    xlim = (10 ** math.floor(math.log10(min(ai)) - 0.3), 10 ** math.ceil(math.log10(max(ai)) + 0.3))
    ylim = (10 ** math.floor(math.log10(min(gf))), 10 ** math.ceil(math.log10(max(roofs + gf)) + 0.1))

    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 5.4), sharey=True)
    axes = axes if len(runs) > 1 else [axes]
    for ax, r in zip(axes, runs):
        panel(ax, r, xlim, ylim)
        print(summarize(r))

    d0 = runs[0]["data"]
    total = len(d0["results"])
    shown = sum(placeable(e) for e in d0["results"])
    fig.suptitle(f"{d0['machine']}, {args.variant} ({d0.get('simd_backend', '?')}), "
                 f"session {args.session.name}: {shown} of {total} entries", fontsize=11)
    # The roofs differ per panel, so each panel keeps its own roof legend; the
    # area legend is shared and sits under the panels, clear of the points.
    seen = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[:3], l[:3], loc="upper left", fontsize=7.5)
        for hh, ll in zip(h[3:], l[3:]):
            seen.setdefault(ll, hh)
    areas = sorted(seen, key=lambda a: AREAS.index(a) if a in AREAS else 99)
    fig.legend([seen[a] for a in areas], areas, loc="lower center", ncol=len(areas),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.008, "circle f64, triangle f32, square mixed precision; hollow: cache-resident or "
             "suspect model. Not shown: integer (no integer roof) and entries without flops/bytes "
             "(Krylov, preconditioners, orderings, sparse eigensolvers).", fontsize=7, color="#555555",
             ha="center")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white", metadata={"Date": None} if args.out.suffix == ".svg" else None)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
