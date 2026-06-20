#!/usr/bin/env bash
# Fetch SuiteSparse circuit matrices for the KLU performance scoreboard
# (bench_klu, epic #138). Matrices are downloaded into ./data/klu next to this
# script and are NOT committed to the repo.
#
# Usage:
#   benchmarks/fetch_klu_matrices.sh             # the default circuit suite
#   benchmarks/fetch_klu_matrices.sh big         # also the very large ones (circuit5M)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/data/klu"
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

# Default circuit suite (small -> large).
fetch Rajat       rajat14        # ~180 x 180, tiny
fetch Hamm        add32          # ~5k, classic circuit
fetch Rajat       rajat30        # ~644k, large unsymmetric (the parity target)

# Very large stress matrices (opt-in; circuit5M is hundreds of MB).
if [ "${1:-}" = "big" ]; then
    fetch Freescale circuit5M
fi

echo
echo "Run, e.g.:"
echo "  ./build/benchmarks/bench_klu --csv klu_scoreboard.csv \\"
echo "      $DATA/rajat14/rajat14.mtx $DATA/add32/add32.mtx $DATA/rajat30/rajat30.mtx"
