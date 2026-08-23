#!/usr/bin/env bash
# nc-model TIMING session (#479) -- the runner.
#
# WHAT THIS IS FOR, because the name does not say it and you will not remember.
#
# #429 wants to balance the jc partition (`balanced_nc`). Six ways of sizing `nc`
# are candidates, and choosing between them needs evidence from at least three
# microarchitectures -- a model picked from one machine is how the hardcoded
# Haswell defaults got there in the first place.
#
# There are TWO scripts and they are not interchangeable:
#
#   run_nc_sweep.sh   plans.     Pure enumeration, measures nothing, runs in
#                               seconds, machine may be busy. Says WHICH shapes
#                               can tell the models apart on this box.
#   run_nc_bench.sh   measures.  This one. Needs a quiet machine, takes tens of
#                               minutes, and is subject to the full measurement
#                               contract (#442): preflight, provenance, clean
#                               tree.
#
# RUN THE SWEEP FIRST. If it reports `m1_balanced  0 of 20`, this machine cannot
# separate M0 from M1 and this script would spend an hour timing byte-identical
# code. That is a result, and it is much cheaper to learn from the sweep.
#
# WHAT TO READ IN THE OUTPUT. Three lines at the end, in this order:
#
#   Noise floor: N%  (worst deviation among K arm(s) whose nc equalled M0's)
#   Best M1-over-M0 gain where M1's nc actually differed: G% (over J arm(s))
#   Worst first-round excess over the minimum: X%
#
# The noise floor is measured, not assumed: those K arms chose the same `nc` as
# the baseline, so they ran BYTE-IDENTICAL code and any deviation from 1.0 is the
# session's own noise. If G is not comfortably above N, this session did not
# measure the effect -- it measured the box. If X is large the minimum has not
# converged and --rounds is too low. K = 0 means the floor is UNMEASURED, which
# is not the same as zero, and the binary says so rather than printing a clean
# 0.00%.
#
# WHY THE ORDERING BELOW IS FUSSY. Preflight runs BEFORE the build, so build load
# cannot dirty the machine-state record. The sidecar is written by the BINARY and
# reports what build_info.hpp said when it was COMPILED, so the tree must be clean
# and the build must be FULL before the run -- otherwise the CSV lands with
# provenance naming a different commit, or claiming a clean tree that was not.
# That failure is silent: the sidecar comes out well-formed, internally
# consistent, and wrong.
#
# Usage:  benchmarks/run_nc_bench.sh --outdir benchmarks/data/<machine> \
#                                    --threads N [--arch <flag>]
#                                    [--reps R] [--rounds K] [--dtypes "double float"]
#   Prefer the per-machine profiles in benchmarks/machines/*-nc-bench.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARCH="-march=native"
OUTDIR=""
THREADS=""
REPS=3
ROUNDS=12
DTYPES="double"
ALLOW_DIRTY=0
BUILD_DIR="build-nc-bench"

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)    ARCH="$2"; shift 2 ;;
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --reps)    REPS="$2"; shift 2 ;;
        --rounds)  ROUNDS="$2"; shift 2 ;;
        --dtypes)  DTYPES="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        -h|--help) sed -n '1,50p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -n "$OUTDIR" ]  || { echo "error: --outdir is required (one directory per machine)." >&2; exit 2; }
[ -n "$THREADS" ] || { echo "error: --threads is required." >&2
                       echo "  Not optional: jc_nt depends on it, and at 1 every balancing" >&2
                       echo "  model is a no-op BY CONSTRUCTION, so the session cannot say" >&2
                       echo "  anything about M1 -- the pair #429 needs." >&2; exit 2; }

if [ "$THREADS" = "1" ]; then
    echo "error: --threads 1 cannot address #429." >&2
    echo "  At jc_nt == 1 every balancing model is a no-op, so M1 == M0 by" >&2
    echo "  construction and the session would time identical code for an hour." >&2
    echo "  Use the count you intend to SHIP at. Set FORCE_T1=1 to override." >&2
    [ "${FORCE_T1:-0}" = "1" ] || exit 2
fi

# ---- the sweep is the prerequisite, and it is cheap ------------------------
# A timing session on a machine whose models never disagree measures nothing.
# The sweep answers that in seconds; refusing to look is how an hour gets spent
# comparing a binary against itself (#470 did exactly that with the quad kernel
# and lost a benchmark round to it).
#
# CHECKED PER DTYPE, not once against `double`. float and double have different
# blocking parameters -- different nr, different kc, therefore a different nc and
# a different set of shapes where the models disagree. Gating a `--dtypes float`
# session on the double sweep asks about the wrong binary. On all four machines
# swept so far the two dtypes happen to give the same count, which is exactly the
# kind of coincidence that makes a wrong check look right until the machine that
# breaks it arrives.
sweep_fail=0
for dt in $DTYPES; do
    si="$OUTDIR/nc_model_sweep_$dt.csv.sysinfo"
    if [ ! -f "$si" ]; then
        echo "== no $dt sweep at $si" >&2
        echo "   Run benchmarks/machines/<machine>-nc-sweep.sh first. It takes seconds" >&2
        echo "   and says whether this session can measure anything at all; this one" >&2
        echo "   takes tens of minutes and needs the box to itself." >&2
        sweep_fail=1
        continue
    fi
    N="$(sed -n 's/^shapes_differing_vs_m0_m1_balanced=//p' "$si" | head -1)"
    if [ -z "$N" ]; then
        echo "== $dt sweep has no M1-vs-M0 count; re-run the sweep." >&2
        sweep_fail=1
        continue
    fi
    echo "== $dt sweep says M1 differs from M0 on $N shape(s)"
    if [ "$N" = "0" ]; then
        echo "   Nothing to measure for $dt: every arm would be byte-identical to" >&2
        echo "   the baseline. This is a RESULT -- record it and skip the session." >&2
        sweep_fail=1
    fi
done
if [ "$sweep_fail" != "0" ]; then
    echo "" >&2
    echo "Refusing to measure. Set FORCE=1 to override (and say so when reporting)." >&2
    [ "${FORCE:-0}" = "1" ] || exit 2
fi

# ---- run contract: preflight BEFORE building, so build load cannot dirty it --
PREFLIGHT_KV=""
PREFLIGHT_ARGS=(--ignore-path "$OUTDIR")
[ "$ALLOW_DIRTY" = "1" ] && PREFLIGHT_ARGS+=(--allow-dirty)
if [ -x benchmarks/preflight.sh ]; then
    if ! PREFLIGHT_KV=$(benchmarks/preflight.sh "${PREFLIGHT_ARGS[@]}"); then
        printf '%s\n' "$PREFLIGHT_KV" >&2
        echo "preflight failed -- not measuring." >&2
        exit 1
    fi
    printf '%s\n' "$PREFLIGHT_KV"
fi

echo "== configuring ($ARCH)"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DMTL5_BUILD_BENCHMARKS=ON -DMTL5_BUILD_TESTS=OFF -DMTL5_BUILD_EXAMPLES=OFF \
      -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_FAST_GEMM=ON \
      -DCMAKE_CXX_FLAGS="$ARCH" > /dev/null

# FULL build, deliberately not --target: the custom target that regenerates
# build_info.hpp is part of ALL, and a targeted build silently leaves the
# previous commit's stamp in place.
echo "== building (full, so the provenance stamp is regenerated)"
cmake --build "$BUILD_DIR" --parallel 4 > /dev/null

mkdir -p "$OUTDIR"
for dt in $DTYPES; do
    echo "== timing dtype=$dt (threads=$THREADS reps=$REPS rounds=$ROUNDS)"
    MTL5_NUM_THREADS="$THREADS" \
    "$BUILD_DIR/benchmarks/bench_nc_models" \
        --threads "$THREADS" --dtype "$dt" --reps "$REPS" --rounds "$ROUNDS" \
        --csv "$OUTDIR/nc_model_timing_$dt.csv"

    # Append the machine-state half of the contract. The binary writes the build
    # half -- it is the only thing that can see build_info.hpp -- and preflight
    # writes what the machine was doing, which only this script can see.
    si="$OUTDIR/nc_model_timing_$dt.csv.sysinfo"
    if [ -n "$PREFLIGHT_KV" ] && [ -f "$si" ]; then
        printf '%s\n' "$PREFLIGHT_KV" >> "$si"
    fi
done

# Verify rather than trust. Every way of getting the ordering wrong produces a
# well-formed sidecar that names the wrong commit.
echo "== provenance check"
HEAD_SHA="$(git rev-parse --short=12 HEAD)"
fail=0
for dt in $DTYPES; do
    si="$OUTDIR/nc_model_timing_$dt.csv.sysinfo"
    [ -f "$si" ] || { echo "  MISSING $si" >&2; fail=1; continue; }
    got="$(sed -n 's/^git_commit=//p' "$si" | head -1)"
    dirty="$(sed -n 's/^git_dirty=//p' "$si" | head -1)"
    if [ "$got" != "$HEAD_SHA" ]; then
        echo "  MISMATCH $dt: sidecar says $got, HEAD is $HEAD_SHA" >&2; fail=1
    elif [ "$dirty" != "0" ] && [ "$ALLOW_DIRTY" != "1" ]; then
        echo "  MISMATCH $dt: sidecar says git_dirty=$dirty" >&2; fail=1
    else
        echo "  $dt: $got (clean) -- matches HEAD"
    fi
    # Surface the verdict the analysis depends on, so it is seen now rather than
    # discovered when the numbers are already in a document.
    nf="$(sed -n 's/^noise_floor=//p' "$si" | head -1)"
    na="$(sed -n 's/^noise_floor_arms=//p' "$si" | head -1)"
    ab="$(sed -n 's/^effect_above_noise=//p' "$si" | head -1)"
    echo "     noise floor $nf over $na control arm(s); effect_above_noise=$ab"
done
if [ "$fail" != "0" ]; then
    echo "" >&2
    echo "The data does not describe the tree it was produced from. Do not commit it." >&2
    exit 1
fi

echo ""
echo "Commit both files per dtype -- the CSV and its .sysinfo. CI rejects a CSV"
echo "without one (benchmarks/check_sidecars.sh)."
