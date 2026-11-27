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
#   SPARSE_SIZES sparse 2-D grid sides (default 100,160 -- see the cost note
#                below before raising it).
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
# Sparse grid sides. The cost is dominated by the UNTIMED factorization each
# case performs before the (millisecond) solves are measured, and it grows
# steeply. Measured whole-suite wall clock at T=1 on an i7-12700K, with the
# 3-D grid clamped (#322):
#
#     side 100    11 s
#     side 160     3 m 38 s
#     side 200     8 m 46 s
#     side 320    not measured; its 2-D case alone is n=102,400
#
# That is per thread count, and the factorization is serial, so every value in
# THREADS pays it again: this default is ~230 s x 4 = ~15 min, in proportion
# with the rest of this driver.
#
# The previous default of 200,320 did not complete: a T=1 run was killed after
# 3 h 28 min without finishing its first size (#321). #322 clamped the 3-D grid,
# which is what brought side 200 down to the 8 m 46 s above, but 200,320 is
# still far too slow for a four-thread-count sweep. Raise this deliberately.
SPARSE_SIZES="${SPARSE_SIZES:-100,160}"
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
        if [[ $first -eq 1 ]]; then
            cat "$tmp" > "$out"; first=0
            # The binary writes <csv>.sysinfo; the per-T runs merge into one
            # CSV, so the first one's sidecar is kept and completed below.
            [[ -f "$tmp.sysinfo" ]] && mv "$tmp.sysinfo" "$out.sysinfo"
        else
            tail -n +2 "$tmp" >> "$out"
        fi
        rm -f "$tmp" "$tmp.sysinfo"
    done
    # The run's keys (#930): the machine's state, what ran, the thread counts
    # the CSV holds, and the cores they were pinned to.
    if [[ -f "$out.sysinfo" ]]; then
        printf '%s\n' "$PREFLIGHT_KV" >> "$out.sysinfo"
        sidecar_put "$out.sysinfo" "pool_size=$(echo $THREADS | tr ' ' ',')" "label=native"
        sidecar_finish "$out.sysinfo" "benchmark=$(basename "$bin")" "class=${BENCH_CLASS:-scaling $name}" \
                       "bench_cpus=$PCPUS" "harness=run_scaling_297.sh" || SIDECAR_FAIL=1
    else
        echo "  $out.sysinfo was not written" >&2; SIDECAR_FAIL=1
    fi
    echo "  -> $out"
}

BA="$BUILD/benchmarks/bench_all"
BS="$BUILD/benchmarks/bench_sparse"
# The machine's state for the sidecars (#930), recorded, not gated, as before.
PREFLIGHT_KV=$(benchmarks/preflight.sh --repo . --ignore-path "$DATA" --report-only 2>/dev/null || true)
. benchmarks/sidecar.sh
SIDECAR_FAIL=0

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

if [[ "$SIDECAR_FAIL" != 0 ]]; then
    echo "run_scaling_297.sh: a sidecar lacks required keys (above); do not commit it as is" >&2
    exit 1
fi
echo
echo "Done. Analyze with:"
echo "  ./benchmarks/analyze_scaling.py $DATA/scaling_*.csv --plot $DATA/scaling_297.png"
echo "  (add --op <substr> to focus one family, e.g. --op ewise or --op snlu)"
