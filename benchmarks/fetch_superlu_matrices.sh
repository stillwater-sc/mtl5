#!/usr/bin/env bash
# Fetch SuiteSparse unsymmetric matrices for the SuperLU performance scoreboard
# (bench_superlu, epic #186 / #180). Matrices are downloaded into ./data/superlu
# next to this script and are NOT committed to the repo.
#
# Usage:
#   benchmarks/fetch_superlu_matrices.sh         # the default unsymmetric suite
#   benchmarks/fetch_superlu_matrices.sh big     # also the very large ones (rajat30)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/data/superlu"
mkdir -p "$DATA"
BASE="https://suitesparse-collection-website.herokuapp.com/MM"

fetch() {
    local group="$1" name="$2"
    if [ -f "$DATA/$name/$name.mtx" ]; then echo "have $name"; return; fi
    echo "downloading $group/$name ..."
    # --retry/--connect-timeout harden against transient network failures;
    # -C - resumes a partial download. No --max-time: some matrices are large.
    curl -L --fail --connect-timeout 30 --retry 3 --retry-delay 5 -C - \
        -o "$DATA/$name.tar.gz" "$BASE/$group/$name.tar.gz"
    tar -xzf "$DATA/$name.tar.gz" -C "$DATA"
    rm -f "$DATA/$name.tar.gz"
}

# Default general-unsymmetric suite (small -> large), where SuperLU's supernodal
# BLAS-3 kernels pay off and the scalar native LU should lag.
fetch Hamm  add32      # ~5k, circuit (unsymmetric)
fetch Wang  wang3      # ~26k, semiconductor device simulation
fetch Wang  wang4      # ~26k, semiconductor device simulation
fetch Simon raefsky3   # ~21k, fluid/structural (classic SuperLU matrix)
fetch Simon bbmat      # ~38k, CFD (heavy fill -- supernodal sweet spot)

# Very large stress matrix (opt-in; native LU likely needs the ext: prefix).
if [ "${1:-}" = "big" ]; then
    fetch Rajat rajat30   # ~644k, large unsymmetric
fi

echo
echo "Run, e.g.:"
echo "  ./build/benchmarks/bench_superlu --csv superlu_scoreboard.csv \\"
echo "      $DATA/add32/add32.mtx $DATA/wang3/wang3.mtx $DATA/raefsky3/raefsky3.mtx"
