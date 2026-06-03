#!/usr/bin/env bash
# Multi-core GEMM scaling (#108): measure how the native-fast blocked GEMM scales
# across threads, alongside threaded OpenBLAS / MKL, on the same machine.
#
# Threading is a RUNTIME axis of the same per-backend binary:
#   native-fast : MTL5_NUM_THREADS=T
#   openblas    : OPENBLAS_NUM_THREADS=T
#   mkl         : MKL_NUM_THREADS=T
# For T threads we pin to the first T physical performance cores (one logical id
# per core -- HT siblings excluded) so scaling reflects cores, not SMT.
#
# Writes one CSV per backend (gemm_scaling_<backend>.csv) whose `backend` column
# is labelled "<backend>-t<T>", so analyze_scaling.py can recover (backend, T).
#
# Environment:
#   BENCH_PCPUS  comma list of physical-core logical ids to pin to, longest
#                first-T prefix is used. Default for an i7-12700K: one sibling
#                per P-core (0,2,4,6,8,10,12,14). Set to match YOUR topology
#                (see: lscpu -e=CPU,CORE,MAXMHZ).
#   THREADS      thread counts to sweep (default "1 2 4 8").
#   SCALE_SIZES  GEMM sizes (default "1024,2048").
#   MKL_SETVARS  oneAPI setvars.sh (default /opt/intel/oneapi/setvars.sh); mkl
#                skipped if absent.
#
# Example:  BENCH_PCPUS=0,2,4,6,8,10,12,14 benchmarks/run_scaling.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PCPUS="${BENCH_PCPUS:-0,2,4,6,8,10,12,14}"
THREADS="${THREADS:-1 2 4 8}"
SIZES="${SCALE_SIZES:-1024,2048}"
DATA="benchmarks/data"
MKL_SETVARS="${MKL_SETVARS:-/opt/intel/oneapi/setvars.sh}"
mkdir -p "$DATA"

IFS=',' read -r -a PCPU_ARR <<< "$PCPUS"

# pcpus_for <T>: comma list of the first T physical-core ids.
pcpus_for() {
    local t="$1" out=""
    for ((i = 0; i < t && i < ${#PCPU_ARR[@]}; ++i)); do
        out="${out:+$out,}${PCPU_ARR[$i]}"
    done
    printf '%s' "$out"
}

configure_build() {
    local dir="$1"; shift
    rm -rf "$dir"
    cmake -B "$dir" -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null
    cmake --build "$dir" --target bench_all -j"${JOBS:-4}" >/dev/null
}

# run_scaling_for <build-dir> <backend> <thread-env-var>
run_scaling_for() {
    local bin="$1/benchmarks/bench_all" backend="$2" tvar="$3"
    local out="$DATA/gemm_scaling_${backend}.csv"; rm -f "$out"
    local first=1
    for T in $THREADS; do
        local pin; pin="$(pcpus_for "$T")"
        local tmp; tmp="$(mktemp)"
        echo "  $backend  T=$T  pinned to CPUs $pin"
        env "$tvar=$T" OMP_NUM_THREADS="$T" \
            taskset -c "$pin" "$bin" --suite gemm --sizes "$SIZES" \
            --label "${backend}-t${T}" --csv "$tmp" >/dev/null
        if [[ $first -eq 1 ]]; then cat "$tmp" > "$out"; first=0; else tail -n +2 "$tmp" >> "$out"; fi
        rm -f "$tmp"
    done
    echo "  -> $out"
}

echo "=== native-fast (Highway + -march=native, MTL5_NUM_THREADS) ==="
configure_build build-scaling-native-fast \
    -DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON
run_scaling_for build-scaling-native-fast native-fast MTL5_NUM_THREADS

echo "=== openblas (OPENBLAS_NUM_THREADS) ==="
configure_build build-scaling-openblas -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON
run_scaling_for build-scaling-openblas openblas OPENBLAS_NUM_THREADS

if [[ -f "$MKL_SETVARS" ]]; then
    echo "=== mkl (MKL_NUM_THREADS) ==="
    set +u +e
    # shellcheck disable=SC1090
    source "$MKL_SETVARS" >/dev/null 2>&1 || true
    set -u -e
    configure_build build-scaling-mkl -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON -DBLA_VENDOR=Intel10_64lp
    run_scaling_for build-scaling-mkl mkl MKL_NUM_THREADS
else
    echo "=== mkl: SKIPPED (no $MKL_SETVARS) ==="
fi

echo
echo "Done. Analyze with:"
echo "  ./benchmarks/analyze_scaling.py $DATA/gemm_scaling_*.csv --plot $DATA/gemm_scaling.png"
