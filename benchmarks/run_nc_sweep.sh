#!/usr/bin/env bash
# nc-model disagreement sweep (#479) -- the runner.
#
# WHAT THIS IS FOR, because the name does not say it and you will not remember.
#
# #429 wants to balance the jc partition (`balanced_nc`), and #479 has to choose
# between six ways of sizing `nc` before that can ship. Choosing needs evidence
# from at least three microarchitectures -- a model picked from one machine is
# how the hardcoded Haswell defaults got there in the first place.
#
# Timing six models on four machines is a large session. Most of it would measure
# nothing: for most shapes every model computes the SAME `nc`, so timing them
# compares a binary against itself. This sweep finds, per machine, the shapes
# where the models actually differ -- so the expensive session only measures
# those.
#
# IT MEASURES NOTHING ITSELF. Pure enumeration over `nc_for_model` and
# `plan_gemm_grid`; it runs in seconds and the machine can be busy. No preflight,
# no pinning, no quiet box.
#
# WHAT TO READ IN THE OUTPUT. One line:
#
#     m1_balanced     N of 20   <- the pair #429 needs
#
# N is how many shapes are worth timing on this machine. N = 0 is a RESULT, not a
# failure: it says this machine cannot separate M0 from M1 and should not be
# booked for that question. The Xeon gives 8 of 20.
#
# WHY THE ORDERING BELOW IS FUSSY. The sidecar is written by the BINARY and
# reports what `build_info.hpp` said when the binary was COMPILED. So the tree
# must be clean, and the build must be a full one, BEFORE the run -- otherwise
# the CSV lands with provenance naming a different commit, or claiming a clean
# tree that was not. That is silent: the sidecar comes out well-formed,
# internally consistent, and wrong. It took four attempts to get right by hand,
# which is why it is a script now, and why the stamp is verified at the end
# rather than trusted.
#
# Usage:  benchmarks/run_nc_sweep.sh --outdir benchmarks/data/<machine> \
#                                    [--arch <flag>] [--threads N]
#   Prefer the per-machine profiles in benchmarks/machines/*-nc-sweep.sh, which
#   carry the right flag and thread count for each host.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARCH="-march=native"
OUTDIR=""
THREADS=""
ALLOW_DIRTY=0
BUILD_DIR="build-nc-sweep"

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)    ARCH="$2"; shift 2 ;;
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -n "$OUTDIR" ]  || { echo "error: --outdir is required (one directory per machine)." >&2; exit 2; }
[ -n "$THREADS" ] || { echo "error: --threads is required." >&2
                       echo "  It is not optional: jc_nt depends on it, and at 1 every" >&2
                       echo "  balancing model is a no-op, so the run says nothing about" >&2
                       echo "  M1. Use the count you intend to MEASURE at." >&2; exit 2; }

# A dirty tree means the binary cannot be attributed to a commit, and the sidecar
# would say so -- but the data would still be committed. Stop instead.
if [ "$ALLOW_DIRTY" != "1" ] && [ -n "$(git status --porcelain | grep -v "^?? $OUTDIR" || true)" ]; then
    echo "error: working tree is dirty. The sidecar records the commit the BINARY was" >&2
    echo "       built from, so a dirty tree produces data that cannot be attributed." >&2
    echo "       Commit or stash first, or pass --allow-dirty (recorded as dirty)." >&2
    git status --porcelain | head -10 >&2
    exit 1
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
for dt in double float; do
    echo "== sweeping dtype=$dt"
    "$BUILD_DIR/benchmarks/sweep_nc_models" --threads "$THREADS" --dtype "$dt" \
        --csv "$OUTDIR/nc_model_sweep_$dt.csv" | grep -vE '^DISAGREE'
done

# Verify rather than trust. Every way of getting the ordering wrong produces a
# well-formed sidecar that names the wrong commit, so the only check that works
# is comparing it against the tree.
echo "== provenance check"
HEAD_SHA="$(git rev-parse --short=12 HEAD)"
fail=0
for dt in double float; do
    si="$OUTDIR/nc_model_sweep_$dt.csv.sysinfo"
    got="$(sed -n 's/^git_commit=//p' "$si" | head -1)"
    dirty="$(sed -n 's/^git_dirty=//p' "$si" | head -1)"
    if [ "$got" != "$HEAD_SHA" ]; then
        echo "  MISMATCH $dt: sidecar says $got, HEAD is $HEAD_SHA" >&2; fail=1
    elif [ "$dirty" != "0" ] && [ "$ALLOW_DIRTY" != "1" ]; then
        echo "  MISMATCH $dt: sidecar says git_dirty=$dirty" >&2; fail=1
    else
        echo "  $dt: $got (clean) -- matches HEAD"
    fi
done
if [ "$fail" != "0" ]; then
    echo "" >&2
    echo "The data does not describe the tree it was produced from. Do not commit it." >&2
    echo "Re-run from a clean tree; this script does a full build for that reason." >&2
    exit 1
fi

echo ""
echo "Commit both files per dtype -- the CSV and its .sysinfo. CI rejects a CSV"
echo "without one (benchmarks/check_sidecars.sh)."
