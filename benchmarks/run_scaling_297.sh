#!/usr/bin/env bash
# On-node threading scaling for the #297 rollout: native 1->N scaling of every
# threaded kernel family, pinned to distinct PHYSICAL cores so the numbers
# reflect cores, not SMT siblings (the pitfall documented in
# docs/design/multicore-scaling-investigation.md).
#
# Suites (one CSV per family, backend column labelled "native-t<T>"):
#   gemm_rect  -- rectangular GEMM: the BLIS multi-loop (2D) grid            (#311)
#   lu/qr/chol -- dense factorizations                                       (#298,#300)
#   ewise      -- element-wise vector/matrix expression sweeps              (#312)
#   sparse     -- level-scheduled sparse triangular solves                  (#301-#309)
#
# Environment:
#   BENCH_PCPUS  comma list of physical-core logical ids (one sibling per core),
#                longest first-T prefix is pinned. Default for an i7-12700K:
#                0,2,4,6,8,10,12,14. SET THIS TO MATCH YOUR TOPOLOGY
#                (see: lscpu -e=CPU,CORE,MAXMHZ).
#   THREADS      thread counts to sweep (default "1 2 4 8").
#   LAPACK_SIZES dense factor sizes (default 1024,2048,4096).
#   SPARSE_SIZES sparse 2-D grid sides (default 200,320).
#   BUILD        build dir (default build-scaling-297).
#   JOBS         build parallelism (default 4).
#
# Example:  BENCH_PCPUS=0,2,4,6,8,10,12,14 THREADS="1 2 4 8" benchmarks/run_scaling_297.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PCPUS="${BENCH_PCPUS:-0,2,4,6,8,10,12,14}"
THREADS="${THREADS:-1 2 4 8}"
LAPACK_SIZES="${LAPACK_SIZES:-1024,2048,4096}"
SPARSE_SIZES="${SPARSE_SIZES:-200,320}"
BUILD="${BUILD:-build-scaling-297}"
DATA="benchmarks/data"
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

echo "=== configure + build native-fast benchmarks ($BUILD) ==="
cmake -B "$BUILD" -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
    -DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON >/dev/null
cmake --build "$BUILD" --target bench_all bench_sparse -j"${JOBS:-4}" >/dev/null

# run_native <csv-name> <binary> <bench-args...>: sweep THREADS, one process per
# T pinned to the first-T physical cores; concatenate into one per-suite CSV.
run_native() {
    local name="$1"; shift
    local bin="$1"; shift
    local out="$DATA/scaling_${name}.csv"; rm -f "$out"; local first=1
    for T in $THREADS; do
        if (( T > ${#PCPU_ARR[@]} )); then
            echo "error: T=$T exceeds the ${#PCPU_ARR[@]} core(s) in BENCH_PCPUS ($PCPUS)" >&2
            exit 1
        fi
        local pin; pin="$(pcpus_for "$T")"
        local tmp; tmp="$(mktemp)"
        echo "  $name  T=$T  pinned to CPUs $pin"
        env MTL5_NUM_THREADS="$T" taskset -c "$pin" "$bin" "$@" \
            --label "native-t${T}" --csv "$tmp" >/dev/null
        if [[ $first -eq 1 ]]; then cat "$tmp" > "$out"; first=0; else tail -n +2 "$tmp" >> "$out"; fi
        rm -f "$tmp"
    done
    echo "  -> $out"
}

BA="$BUILD/benchmarks/bench_all"
BS="$BUILD/benchmarks/bench_sparse"

echo "=== gemm-rect (multi-loop 2D grid) ==="
run_native gemm_rect "$BA" --suite gemm-rect
echo "=== dense factorizations (lu / qr / cholesky) ==="
run_native lu   "$BA" --suite lu       --lapack-sizes "$LAPACK_SIZES"
run_native qr   "$BA" --suite qr       --lapack-sizes "$LAPACK_SIZES"
run_native chol "$BA" --suite cholesky --lapack-sizes "$LAPACK_SIZES"
echo "=== element-wise sweeps ==="
run_native ewise "$BA" --suite ewise
echo "=== sparse triangular solves (level-scheduled) ==="
run_native sparse "$BS" --sizes "$SPARSE_SIZES"

echo
echo "Done. Analyze with:"
echo "  ./benchmarks/analyze_scaling.py $DATA/scaling_*.csv --plot $DATA/scaling_297.png"
echo "  (add --op <substr> to focus one family, e.g. --op ewise or --op snlu)"
