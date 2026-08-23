#!/usr/bin/env bash
# nc-model TIMING session on the i7-12700K (Alder Lake) (#479).
#
# See benchmarks/run_nc_bench.sh for what this is FOR. In one line: it times the
# six candidate `nc` models against each other on the shapes the SWEEP nominated
# for this machine, so #429 can choose one on evidence rather than on the
# argument that an even partition must be faster.
#
# RUN THE SWEEP FIRST (*-nc-sweep.sh). It takes seconds and says whether this
# machine can separate M0 from M1 at all. If it cannot, this script would spend
# an hour timing byte-identical code.
#
# Read the last three lines of the output: the measured noise floor, the best M1
# gain, and the first-round excess. A gain that is not comfortably above the
# floor is a measurement of the box, not of the model.
#
# Eight P-cores. E-cores are excluded from the thread count deliberately.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXPECT="12700K"
ACTUAL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' \
          || tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
if [[ "$ACTUAL" != *"$EXPECT"* ]]; then
    echo "This profile is for a $EXPECT; this machine reports:"
    echo "  $ACTUAL"
    [ "${FORCE:-0}" = "1" ] || { echo "Refusing to write $EXPECT data. Set FORCE=1 if this really is one."; exit 2; }
fi

echo "profile: i7-12700k-nc-bench"
echo "  cpu:     $ACTUAL"
echo "  reps:    3   rounds: 7"
echo "  arch:    -march=alderlake"
echo "  threads: 8"
echo ""

exec "$REPO_ROOT/benchmarks/run_nc_bench.sh" \
    --arch "-march=alderlake" --threads 8 --outdir "benchmarks/data/i7-12700k" "$@"
