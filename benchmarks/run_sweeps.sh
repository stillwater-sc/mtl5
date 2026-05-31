#!/usr/bin/env bash
# Build one bench_all per backend and run the sweeps, writing one CSV per
# backend into benchmarks/data/. This is the "one executable per backend"
# methodology: each variant is compiled with the BLAS/LAPACK flags a dependent
# application would set for the whole program, and the public mtl:: API
# dispatches accordingly. Native is a generic-only build (no BLAS/LAPACK).
#
# Usage:
#   benchmarks/run_sweeps.sh [sweep-spec]
#
# Environment:
#   BENCH_CPU    CPU id to pin to via taskset (recommend a P-core on hybrid
#                CPUs for stable single-thread numbers). Empty = no pinning.
#   BENCH_SWEEP  sweep spec (default 65:1025:80, all-odd / non-power-of-2).
#   MKL_SETVARS  path to oneAPI setvars.sh (default /opt/intel/oneapi/setvars.sh);
#                the MKL variant is skipped if it is not found.
#
# Example (pin to P-core 4):  BENCH_CPU=4 benchmarks/run_sweeps.sh
set -euo pipefail

# Resolve repo root from this script's location (portable).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SWEEP="${1:-${BENCH_SWEEP:-65:1025:80}}"
DATA="benchmarks/data"
MKL_SETVARS="${MKL_SETVARS:-/opt/intel/oneapi/setvars.sh}"
mkdir -p "$DATA"

PIN=()
if [[ -n "${BENCH_CPU:-}" ]]; then
    PIN=(taskset -c "$BENCH_CPU")
    echo "Pinning runs to CPU ${BENCH_CPU}"
else
    echo "WARNING: BENCH_CPU not set -- no CPU pinning. On hybrid (P/E-core)"
    echo "         CPUs the small L1 kernels may land on E-cores and skew results."
fi

# configure_build <build-dir> <extra-cmake-args...>
# Starts from a clean build dir so a variant can never pick up a stale binary
# from a previous run with different flags.
configure_build() {
    local dir="$1"; shift
    rm -rf "$dir"
    cmake -B "$dir" -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null
    cmake --build "$dir" --target bench_all -j"${JOBS:-4}" >/dev/null
}

# run_variant <build-dir> <label>
run_variant() {
    local bin="$1/benchmarks/bench_all"; local label="$2"
    echo ">> $label: BLAS L1/L2/L3 sweep"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "${PIN[@]}" "$bin" --suite blas --sweep "$SWEEP" \
        --label "$label" --csv "$DATA/blas_sweep_${label}.csv"
    echo ">> $label: LAPACK factorization sweep"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "${PIN[@]}" "$bin" --suite lapack --lapack-sweep "$SWEEP" \
        --label "$label" --csv "$DATA/lapack_sweep_${label}.csv"
}

echo "=== native (generic-only) ==="
configure_build build-native
run_variant build-native native

echo "=== openblas ==="
configure_build build-openblas -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON
run_variant build-openblas openblas

if [[ -f "$MKL_SETVARS" ]]; then
    echo "=== mkl ==="
    # shellcheck disable=SC1090
    source "$MKL_SETVARS" >/dev/null 2>&1
    configure_build build-mkl -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON -DBLA_VENDOR=Intel10_64lp
    run_variant build-mkl mkl
else
    echo "=== mkl: SKIPPED (no $MKL_SETVARS) ==="
fi

echo
echo "Done. CSVs in $DATA/. Plot with e.g.:"
echo "  ./benchmarks/plot_results.py $DATA/blas_sweep_*.csv --out $DATA/blas_sweep_gflops.png"
echo "  ./benchmarks/plot_results.py $DATA/lapack_sweep_*.csv --out $DATA/lapack_sweep_gflops.png"
