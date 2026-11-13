#!/usr/bin/env python3
"""Roofline plots of one platform-benchmark session (#737).

Reads the mtl5_bench JSON of one published session and places every entry
the roofline model covers at its arithmetic intensity (flops / bytes) and its
measured GFLOP/s. The roofs are each run's own probes, the same ones
mtl5_bench's %roof is computed against (benchmarks/platform/run.hpp):

    attainable = min(FMA peak of the dtype, bandwidth x intensity)

so a memory-bound entry at 100% of its roof sits on the slope, and a
compute-bound one at 100% sits on its dtype's ceiling.

The figure is a grid: one row per thread count (1 thread, then all threads)
and one column per build variant, by default

  * native       the generic build: the baseline a naive implementation gives;
  * native-fast  the best MTL5 does;
  * blas         OpenBLAS: what tuned libraries reach, MTL5's target.

A variant the session didn't run (no blas where CMake found no BLAS) is left
out, and the title says so. Every panel shares the same axes, so an entry's
move from one column to the next is a direct comparison.

Left out of every panel, and stated in the caption:

  * integer entries: the model has no integer ceiling (the card shows n/a);
  * entries that record no flops or bytes (Krylov solvers, preconditioners,
    orderings, sparse eigensolvers, the direct solvers' .analyze phases): they
    have no intensity to place.

Reference-library entries (ref_*: the blas variant's CHOLMOD and PARDISO,
#810) are drawn as diamonds. --areas keeps only the named areas, e.g.
--areas direct,direct3d for the direct solvers beside their references.

Example:
    benchmarks/plot_roofline.py benchmarks/data/jetson-orin/mtl5_bench/20261111T193624Z \\
        --out docs/img/benchmarks/jetson-orin/roofline-platform.svg

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
# direct3d (#810) is last, so adding it left every earlier area's colour as it was.
AREAS = ["blas1", "blas2", "blas3", "factor", "eigen", "elementwise",
         "sparse", "direct", "smoother", "mg", "mixed", "direct3d"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000", "#a6761d"]
MARKERS = {"f64": "o", "f32": "^"}  # anything else (mixed precision): "s"
# A reference library's entry (ref_cholmod, ref_pardiso_*, #810) is a diamond,
# so it reads as the target beside the MTL5 entries of its area.
REF_MARKER = "D"
TITLES = {"native": "native (generic baseline)", "native-fast": "native-fast (MTL5's best)",
          "blas": "OpenBLAS (target)"}


def placeable(e: dict) -> bool:
    return (e.get("flops", 0) > 0 and e.get("bytes", 0) > 0 and e.get("gflops", 0) > 0
            and e.get("area") != "integer")


def load_runs(session: Path, variant: str, keep: set[str] | None = None) -> dict[int, dict]:
    """{threads: run JSON} for one variant; empty if the session didn't run it.
    With `keep`, only the results of those areas remain."""
    runs = {}
    for f in session.glob(f"{variant}-t*.json"):
        m = re.fullmatch(rf"{re.escape(variant)}-t(\d+)\.json", f.name)
        if m:
            d = json.loads(f.read_text())
            if keep:
                d["results"] = [e for e in d["results"] if e["area"] in keep]
            runs[int(m.group(1))] = d
    return runs


def summarize(variant: str, threads: int, d: dict) -> str:
    """One line per run: how far below its roof each side of the ridge sits."""
    p = d["probes"]
    bw = p["bandwidth_gbytes_per_s"]
    mem, comp = [], []
    for e in d["results"]:
        if not placeable(e):
            continue
        peak = p["fma_peak_f32_gflops"] if e["dtype"] == "f32" else p["fma_peak_f64_gflops"]
        (mem if bw * e["flops"] / e["bytes"] < peak else comp).append(e["roofline_pct"])

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float("nan")

    return (f"{variant} {threads}T: memory-bound {len(mem)} (median {med(mem):.0f}% of roof, "
            f"{sum(x >= 50 for x in mem)} at >=50%), compute-bound {len(comp)} "
            f"(median {med(comp):.0f}%, {sum(x >= 50 for x in comp)} at >=50%)")


def panel(ax, d: dict, title: str, xlim, ylim) -> None:
    p = d["probes"]
    bw = p["bandwidth_gbytes_per_s"]
    peaks = {"f64": p["fma_peak_f64_gflops"], "f32": p["fma_peak_f32_gflops"]}

    xs = [xlim[0] * (xlim[1] / xlim[0]) ** (i / 200) for i in range(201)]
    for dtype, style in (("f32", "--"), ("f64", "-")):
        ax.plot(xs, [min(peaks[dtype], bw * x) for x in xs], style, color="#444444", lw=1.4,
                label=f"{dtype} roof {peaks[dtype]:.1f} GFLOP/s")
    ax.plot([], [], " ", label=f"bandwidth {bw:.1f} GB/s")

    seen = set()
    for e in d["results"]:
        if not placeable(e):
            continue
        area = e["area"]
        color = COLORS[AREAS.index(area)] if area in AREAS else "#999999"
        marker = REF_MARKER if e["op"].startswith("ref_") else MARKERS.get(e["dtype"], "s")
        hollow = e.get("cache_resident") or e.get("suspect_model")
        ax.scatter(e["flops"] / e["bytes"], e["gflops"], s=26, marker=marker,
                   facecolors="none" if hollow else color, edgecolors=color, linewidths=1.0,
                   label=area if area not in seen else None, zorder=3)
        seen.add(area)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    # Upper left lies above the slope, where no point can be.
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:3], l[:3], loc="upper left", fontsize=7)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session", type=Path, help="a published session directory")
    ap.add_argument("--variants", default="native,native-fast,blas",
                    help="comma-separated build variants, one column each "
                         "(default native,native-fast,blas)")
    ap.add_argument("--out", type=Path, required=True, help="output image (.svg or .png)")
    ap.add_argument("--areas", default="",
                    help="comma-separated areas to keep (default: every area)")
    args = ap.parse_args()
    keep = {a for a in args.areas.split(",") if a}

    wanted = [v for v in args.variants.split(",") if v]
    runs = {v: load_runs(args.session, v, keep) for v in wanted}
    variants = [v for v in wanted if runs[v]]
    missing = [v for v in wanted if not runs[v]]
    if not variants:
        sys.exit(f"none of {wanted} in {args.session}")
    threads = sorted({t for v in variants for t in runs[v]})

    import matplotlib
    matplotlib.use("Agg")
    # Text stays text in the SVG: smaller files, and the same bytes on every run.
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "mtl5-roofline"
    import matplotlib.pyplot as plt

    # One set of axes for every panel.
    ai, gf, roofs = [], [], []
    for v in variants:
        for d in runs[v].values():
            p = d["probes"]
            roofs += [p["fma_peak_f64_gflops"], p["fma_peak_f32_gflops"]]
            for e in d["results"]:
                if placeable(e):
                    ai.append(e["flops"] / e["bytes"])
                    gf.append(e["gflops"])
    xlim = (10 ** math.floor(math.log10(min(ai)) - 0.3), 10 ** math.ceil(math.log10(max(ai)) + 0.3))
    ylim = (10 ** math.floor(math.log10(min(gf))), 10 ** math.ceil(math.log10(max(roofs + gf)) + 0.1))

    nr, nc = len(threads), len(variants)
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.4 * nr + 0.9), sharex=True, sharey=True,
                             squeeze=False)
    handles = {}
    for i, t in enumerate(threads):
        for j, v in enumerate(variants):
            ax = axes[i][j]
            d = runs[v].get(t)
            if d is None:
                ax.set_axis_off()
                continue
            panel(ax, d, f"{TITLES.get(v, v)}, {t} thread{'s' if t > 1 else ''}", xlim, ylim)
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll in AREAS:
                    handles.setdefault(ll, hh)
            print(summarize(v, t, d))
        axes[i][0].set_ylabel("GFLOP/s")
    for j in range(nc):
        axes[-1][j].set_xlabel("arithmetic intensity (flop/byte)")

    d0 = runs[variants[0]][threads[0]]
    total = len(d0["results"])
    shown = sum(placeable(e) for e in d0["results"])
    title = (f"{d0['machine']}, session {args.session.name}: {shown} of {total} entries "
             f"per panel")
    if missing:
        title += f" (not run in this session: {', '.join(missing)})"
    fig.suptitle(title, fontsize=11)
    areas = sorted(handles, key=AREAS.index)
    fig.legend([handles[a] for a in areas], areas, loc="lower center", ncol=len(areas),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.008, "circle f64, triangle f32, square mixed precision, diamond reference library "
             "(CHOLMOD, PARDISO); hollow: cache-resident or suspect model.\nNot shown: integer (no "
             "integer roof) and entries without flops/bytes (Krylov, preconditioners, orderings, "
             "sparse eigensolvers, analyze phases).", fontsize=7, color="#555555",
             ha="center", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white", metadata={"Date": None} if args.out.suffix == ".svg" else None)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
