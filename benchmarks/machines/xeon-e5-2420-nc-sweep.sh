#!/usr/bin/env bash
# nc-model disagreement sweep on the Xeon E5-2420 v2 (Ivy Bridge-EP, SSE4) (#479).
#
# See benchmarks/run_nc_sweep.sh for what this is FOR -- in one line: it finds,
# without measuring anything, which matrix shapes on this machine can tell the
# six candidate `nc` models apart, so a later timing session only runs those.
#
# Read one line of the output:  m1_balanced  N of 20
# N = 0 means this machine cannot separate M0 from M1 and should not be booked
# for that question. That is a result, not a failure.
#
# Six physical cores, no SMT interleave in the id space (siblings are 0/6..5/11).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXPECT="E5-2420"
ACTUAL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' \
          || tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
if [[ "$ACTUAL" != *"$EXPECT"* ]]; then
    echo "This profile is for a $EXPECT; this machine reports:"
    echo "  $ACTUAL"
    [ "${FORCE:-0}" = "1" ] || { echo "Refusing to write $EXPECT data. Set FORCE=1 if this really is one."; exit 2; }
fi

echo "profile: xeon-e5-2420-nc-sweep"
echo "  cpu:     $ACTUAL"
echo "  arch:    -march=native"
echo "  threads: 6"
echo ""

exec "$REPO_ROOT/benchmarks/run_nc_sweep.sh" \
    --arch "-march=native" --threads 6 --outdir "benchmarks/data/xeon-e5-2420" "$@"
