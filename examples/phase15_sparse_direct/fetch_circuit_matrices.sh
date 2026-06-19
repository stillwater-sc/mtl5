#!/usr/bin/env bash
# Fetch circuit-simulation matrices from the SuiteSparse Matrix Collection
# (https://sparse.tamu.edu/) for the circuit_matrix_klu example. Matrices are
# downloaded into ./data next to this script and are NOT committed to the repo.
#
# Usage (run from anywhere):
#   examples/phase15_sparse_direct/fetch_circuit_matrices.sh           # small demo: rajat30
#   examples/phase15_sparse_direct/fetch_circuit_matrices.sh all       # also circuit5M (very large, ~hundreds of MB)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/data"
mkdir -p "$DATA"

BASE="https://suitesparse-collection-website.herokuapp.com/MM"

fetch() {
    local group="$1" name="$2"
    if [ -f "$DATA/$name/$name.mtx" ]; then
        echo "already have $name"
        return
    fi
    echo "downloading $group/$name ..."
    curl -L --fail -o "$DATA/$name.tar.gz" "$BASE/$group/$name.tar.gz"
    tar -xzf "$DATA/$name.tar.gz" -C "$DATA"
    rm -f "$DATA/$name.tar.gz"
    echo "  -> $DATA/$name/$name.mtx"
}

# Small/medium runnable demo.
fetch Rajat rajat30

# Very large stress/scaling target (opt-in; can be hundreds of MB and slow to
# factor with the v1 natural per-block ordering).
if [ "${1:-}" = "all" ]; then
    fetch Freescale circuit5M
fi

echo
echo "Run, e.g.:"
echo "  ./build/examples/example_circuit_matrix_klu $DATA/rajat30/rajat30.mtx"
