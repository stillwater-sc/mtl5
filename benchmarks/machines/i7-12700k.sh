#!/usr/bin/env bash
# Cache-blocking A/B on the i7-12700K (Alder Lake, hybrid).
#
# This file IS the invocation. The pin list below is not a preference, it is the
# machine's topology, and getting it wrong on this part does more than add noise:
# the P-cores and E-cores have DIFFERENT cache hierarchies (48 KB vs 32 KB L1d,
# 1.25 MB private vs 2 MB shared L2), so an unpinned or wrongly pinned run
# detects whichever hierarchy the thread happened to land on and the `detected`
# arm becomes unreproducible (#432). Keeping it in a committed script rather than
# in shell history is the difference between a result and an anecdote.
#
# Topology (lscpu -e=CPU,CORE): 8 P-cores with SMT on CPUs 0..15 (siblings
# adjacent, so one id per core is 0,2,4,...,14) and 4 E-cores on 16..19, which
# are EXCLUDED -- mixing core classes would measure the slower one.
#
# Usage:  benchmarks/machines/i7-12700k.sh            # the four-arm kc/mc split
#         ARMS="detected default" benchmarks/machines/i7-12700k.sh
#         ROUNDS=3 benchmarks/machines/i7-12700k.sh   # anything is overridable
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

# Refuse to write another machine's numbers into this machine's directory. The
# CSVs are named by arm and the runner clears them before writing, so a profile
# executed on the wrong host silently replaces committed evidence with data from
# somewhere else -- the #439 failure with a friendlier face. FORCE=1 overrides,
# for a genuinely equivalent part.
EXPECT="i7-12700K"
ACTUAL=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)
if ! printf '%s' "$ACTUAL" | grep -q "$EXPECT"; then
    echo "This profile is for a $EXPECT; this machine reports:" >&2
    echo "  $ACTUAL" >&2
    echo "Refusing to write $EXPECT data. Set FORCE=1 if this really is one." >&2
    [ "${FORCE:-0}" = 1 ] || exit 2
fi

export BENCH_PROFILE=i7-12700k
export OUTDIR=${OUTDIR:-benchmarks/data/i7-12700k}
export BUILD_DIR=${BUILD_DIR:-build-release}
DEFAULT_PCPUS=0,2,4,6,8,10,12,14                        # P-cores, one id per core
export BENCH_PCPUS=${BENCH_PCPUS:-$DEFAULT_PCPUS}
export THREADS=${THREADS:-"1 8"}                        # 1 and all eight P-cores
export ROUNDS=${ROUNDS:-5}
export REPS=${REPS:-5}
export DTYPE=${DTYPE:-double}
# All four arms in ONE session: this is the machine that decides the kc/mc
# question (#430) -- its L1d is 48 KB, so kc moves 256 -> 384 here, and it owns
# the unexplained 1024^3 T=8 loss.
export ARMS=${ARMS:-"default detected kconly mconly"}

echo "profile: $BENCH_PROFILE"
echo "  build:   cmake --preset release -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON"
if [ "$BENCH_PCPUS" = "$DEFAULT_PCPUS" ]; then
    echo "  pinning: $BENCH_PCPUS (P-cores only; E-cores 16-19 excluded)"
else
    # Do not annotate a list this profile did not choose: claiming "P-cores only"
    # about an overridden list is the kind of confident wrong label the sidecar
    # work exists to remove.
    echo "  pinning: $BENCH_PCPUS (OVERRIDDEN; profile default is $DEFAULT_PCPUS)"
fi
echo "  arms:    $ARMS"
echo "  outdir:  $OUTDIR"
echo

# -march=native matters here: without an ISA flag Highway compiles for the
# 128-bit baseline, and kc/mc/nc all divide by the SIMD width -- the blocking
# under test would be for a vector length the production build never uses.
if [ ! -x "$BUILD_DIR/benchmarks/bench_blocking_ab_default" ]; then
    echo "== configuring and building $BUILD_DIR"
    cmake --preset release -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON >/dev/null
    targets=""
    for a in $ARMS; do targets="$targets bench_blocking_ab_$a"; done
    # shellcheck disable=SC2086
    cmake --build "$BUILD_DIR" --target $targets -j4 >/dev/null
fi

exec "$ROOT/benchmarks/run_blocking_ab.sh"
