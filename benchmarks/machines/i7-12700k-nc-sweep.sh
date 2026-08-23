#!/usr/bin/env bash
# nc-model disagreement sweep on the i7-12700K (Alder Lake) (#479).
#
# See benchmarks/run_nc_sweep.sh for what this is FOR -- in one line: it finds,
# without measuring anything, which matrix shapes on this machine can tell the
# six candidate `nc` models apart, so a later timing session only runs those.
#
# Read one line of the output:  m1_balanced  N of 20
# N = 0 means this machine cannot separate M0 from M1 and should not be booked
# for that question. That is a result, not a failure.
#
# Eight P-cores. E-cores are excluded from the thread count deliberately.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

. "$(dirname "${BASH_SOURCE[0]}")/_identify.sh"
require_machine "12700K" "i7-12700K"

echo "profile: i7-12700k-nc-sweep"
echo "  cpu:     $MTL5_MACHINE_ID"
echo "  arch:    -march=alderlake"
echo "  threads: 8"
echo ""

# "$@" comes FIRST so the profile's own --arch/--threads/--outdir come last
# and WIN. The runner takes the last occurrence of a repeated option, so with
# "$@" trailing a caller could silently redirect this machine's data into
# another machine's directory -- the cross-machine overwrite #439 already cost
# a set of results. Pass-through still works for everything the profile does
# not pin (--reps, --rounds, --dtypes, --allow-dirty).
exec "$REPO_ROOT/benchmarks/run_nc_sweep.sh" \
    "$@" \
    --arch "-march=alderlake" --threads 8 --outdir "benchmarks/data/i7-12700k"
