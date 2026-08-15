#!/usr/bin/env bash
# Build one bench_all per backend and run the sweeps, writing one CSV per
# backend into OUTDIR. This is the "one executable per backend" methodology:
# each variant is compiled with the BLAS/LAPACK flags a dependent application
# would set for the whole program, and the public mtl:: API dispatches
# accordingly. Native is a generic-only build (no BLAS/LAPACK).
#
# This harness follows the run contract in docs/benchmarks/systems.md: preflight
# gates, pinning, a per-machine OUTDIR, and machine state in every sidecar. It
# does NOT interleave -- see the note above the measurement phase.
#
# Usage:
#   OUTDIR=benchmarks/data/<machine> benchmarks/run_sweeps.sh [sweep-spec]
#
# Environment:
#   OUTDIR       REQUIRED. Where the CSVs land, one directory PER MACHINE. There
#                is deliberately no default: the CSVs are named by backend, not
#                by machine, so a shared default silently overwrites another
#                machine's committed results (#439).
#   BENCH_CPU    CPU id to pin to via taskset (recommend a P-core on hybrid
#                CPUs for stable single-thread numbers). Empty = no pinning.
#   BENCH_SWEEP  sweep spec (default 65:1025:80, all-odd / non-power-of-2).
#   BENCH_SUITES suites to run per variant (default "blas lapack"). The native
#                and native-fast builds have no LAPACK, so generic LU/QR/eig at
#                large N is impractical -- set BENCH_SUITES=blas for the GEMM/
#                GEMV/L1 acceptance gate (#93).
#   MKL_SETVARS  path to oneAPI setvars.sh (default /opt/intel/oneapi/setvars.sh);
#                the MKL variant is skipped if it is not found.
#   ALLOW_DIRTY=1  permit a dirty working tree (recorded as tree_git_dirty=1).
#
# Variants: native (generic-only), native-fast (the blocked GEMM / SIMD GEMV
# path: -DMTL5_NATIVE_FAST_GEMM + Highway + -march=native, no external BLAS),
# openblas, blis (BLA_VENDOR=FLAME, if a BLIS BLAS is found), and mkl (if oneAPI
# is present).
#
# Example (P-core 4, GEMM/GEMV/L1 gate only):
#   OUTDIR=benchmarks/data/i7-12700k BENCH_CPU=4 BENCH_SUITES=blas \
#       benchmarks/run_sweeps.sh
set -euo pipefail

# Resolve repo root from this script's location (portable).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SWEEP="${1:-${BENCH_SWEEP:-65:1025:80}}"
SUITES="${BENCH_SUITES:-blas lapack}"
MKL_SETVARS="${MKL_SETVARS:-/opt/intel/oneapi/setvars.sh}"

if [ -z "${OUTDIR:-}" ]; then
    echo "OUTDIR is required: give this machine its own directory, e.g." >&2
    echo "  OUTDIR=benchmarks/data/<machine> $0" >&2
    echo "Existing machine directories:" >&2
    ls -d benchmarks/data/*/ 2>/dev/null | sed 's|^|  |' >&2
    exit 2
fi
DATA="$OUTDIR"
mkdir -p "$DATA"

PIN=()
if [[ -n "${BENCH_CPU:-}" ]]; then
    PIN=(taskset -c "$BENCH_CPU")
    echo "Pinning runs to CPU ${BENCH_CPU}"
else
    echo "WARNING: BENCH_CPU not set -- no CPU pinning. On hybrid (P/E-core)"
    echo "         CPUs the small L1 kernels may land on E-cores and skew results."
fi

# Gate the session, and record the state it ran in (#442). Called twice on
# purpose: once BEFORE the builds, so a dirty tree or a competing compile fails
# in seconds rather than after ten minutes of configuring, and once after them,
# because the builds heat the machine and the temperature that matters is the
# one the MEASUREMENTS start at. The second record goes in the sidecars.
#
# These are single-threaded runs (every vendor's thread count is forced to 1
# below), so the thread budget asked of preflight is 1.
preflight_gate() {
    local kv
    if ! kv=$("$SCRIPT_DIR/preflight.sh" --threads 1 --repo "$ROOT" \
                  --ignore-path "$DATA"); then
        echo "preflight failed -- not measuring. Fix the above, or set ALLOW_DIRTY=1" >&2
        echo "if you accept a dirty tree (it is then recorded as such)." >&2
        exit 1
    fi
    printf '%s\n' "$kv"
}
preflight_gate >/dev/null

# configure_build <build-dir> <extra-cmake-args...>
# Starts from a clean build dir so a variant can never pick up a stale binary
# from a previous run with different flags.
configure_build() {
    local dir="$1"; shift
    rm -rf "$dir"
    cmake -B "$dir" -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null
    cmake --build "$dir" --target bench_all -j"${JOBS:-4}" >/dev/null
}

# Every sidecar this session writes, so machine state can be appended to all of
# them at the end.
SIDECARS=()

# run_variant <build-dir> <label>
# Runs the suites named in $SUITES (blas and/or lapack), one CSV each.
run_variant() {
    local bin="$1/benchmarks/bench_all"; local label="$2"
    local s csv flag
    for s in $SUITES; do
        case "$s" in
            blas)   csv="$DATA/blas_sweep_${label}.csv";   flag=--blas-sweep ;;
            lapack) csv="$DATA/lapack_sweep_${label}.csv"; flag=--lapack-sweep ;;
            *) echo "  (unknown suite '$s' in BENCH_SUITES, skipping)"; continue ;;
        esac
        echo ">> $label: $s sweep"
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
            "${PIN[@]}" "$bin" --suite "$s" "$flag" "$SWEEP" \
            --label "$label" --csv "$csv"
        SIDECARS+=("$csv.sysinfo")
    done
}

# ── build phase ─────────────────────────────────────────────────────────────
# Every variant is built BEFORE any of them is measured. Building and measuring
# in turn meant each backend was timed on a machine still hot from compiling its
# own binary, while the next one compiled during the previous one's cooldown --
# a per-arm bias in a comparison whose whole purpose is comparing arms.
echo "=== building all variants (nothing is measured yet) ==="
LABELS=(); DIRS=(); MKL_ARM=-1

configure_build build-native
LABELS+=(native); DIRS+=(build-native)

configure_build build-native-fast \
    -DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON
LABELS+=(native-fast); DIRS+=(build-native-fast)

configure_build build-openblas -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON
LABELS+=(openblas); DIRS+=(build-openblas)

# BLIS (BLAS-compatible; selected via CMake FindBLAS BLA_VENDOR=FLAME). BLIS is a
# BLAS-only library (LAPACK would come from libflame, not wired here), so it is
# configured with BLAS only -- the BLAS L1/L2/L3 suites are the point of the
# comparison. Skipped if a FLAME/BLIS BLAS cannot be located at configure time.
if cmake -B build-blis-probe -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
        -DMTL5_WITH_BLAS=ON -DBLA_VENDOR=FLAME >/dev/null 2>&1; then
    rm -rf build-blis-probe
    configure_build build-blis -DMTL5_WITH_BLAS=ON -DBLA_VENDOR=FLAME
    LABELS+=(blis); DIRS+=(build-blis)
else
    rm -rf build-blis-probe
    echo "=== blis: SKIPPED (no FLAME/BLIS BLAS found; install libblis-dev or set BLA_VENDOR) ==="
fi

if [[ -f "$MKL_SETVARS" ]]; then
    # oneAPI's setvars.sh references unset variables, which would trip `set -u`
    # and abort mid-source, and it may return non-zero under `set -e`. It is also
    # sourced in a SUBSHELL, here and at run time: sourcing it into this shell
    # would leave oneAPI's LD_LIBRARY_PATH ahead of every later run, so the
    # openblas and blis binaries could resolve MKL's libraries while still being
    # labelled as themselves -- one arm measuring another.
    ( set +u +e
      # shellcheck disable=SC1090
      source "$MKL_SETVARS" >/dev/null 2>&1 || true
      set -u -e
      configure_build build-mkl -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON \
                      -DBLA_VENDOR=Intel10_64lp )
    LABELS+=(mkl); DIRS+=(build-mkl)
    MKL_ARM=$(( ${#LABELS[@]} - 1 ))
else
    echo "=== mkl: SKIPPED (no $MKL_SETVARS) ==="
fi

# ── measurement phase ───────────────────────────────────────────────────────
# Gate again now the builds are done: this is the temperature and the load the
# numbers are actually taken under, and it is what the sidecars record.
#
# NOT interleaved, unlike run_blocking_ab. Interleaving needs several rounds per
# arm, and bench_all truncates its --csv on every invocation, so rounds would
# overwrite rather than accumulate. Doing it properly means append support in
# bench_all and best-of-rounds in the analyzers; until then, building everything
# up front removes the largest order effect. Tracked in #442.
PREFLIGHT_KV=$(preflight_gate)
echo "=== measuring ==="
for i in "${!LABELS[@]}"; do
    echo "=== ${LABELS[$i]} ==="
    if [[ $i -eq $MKL_ARM ]]; then
        ( set +u +e
          # shellcheck disable=SC1090
          source "$MKL_SETVARS" >/dev/null 2>&1 || true
          set -u -e
          run_variant "${DIRS[$i]}" "${LABELS[$i]}" )
        for s in $SUITES; do
            case "$s" in
                blas)   SIDECARS+=("$DATA/blas_sweep_${LABELS[$i]}.csv.sysinfo") ;;
                lapack) SIDECARS+=("$DATA/lapack_sweep_${LABELS[$i]}.csv.sysinfo") ;;
            esac
        done
    else
        run_variant "${DIRS[$i]}" "${LABELS[$i]}"
    fi
done

# ── record machine state in every sidecar ───────────────────────────────────
# A failed postflight read must not silently drop the key: `unavailable` is a
# fact, an absent key is a reader's guess.
if ! AFTER_KV=$("$SCRIPT_DIR/preflight.sh" --phase after); then
    echo "WARNING: postflight thermal read failed; recording thermal_after_c=unavailable." >&2
    AFTER_KV="thermal_after_c=unavailable"
fi
case "$AFTER_KV" in
    *thermal_after_c=*) ;;
    *) AFTER_KV="thermal_after_c=unavailable" ;;
esac

for s in "${SIDECARS[@]}"; do
    [[ -f "$s" ]] || continue
    printf '%s\n%s\n' "$PREFLIGHT_KV" "$AFTER_KV" >> "$s"
    printf 'harness=run_sweeps.sh\nharness_pinned_cpus=%s\nharness_interleaved=0\n' \
           "${BENCH_CPU:-none}" >> "$s"
done

echo
echo "Done. CSVs in $DATA/. Plot with e.g.:"
echo "  ./benchmarks/plot_results.py $DATA/blas_sweep_*.csv --out $DATA/blas_sweep_gflops.png"
echo "  ./benchmarks/plot_results.py $DATA/lapack_sweep_*.csv --out $DATA/lapack_sweep_gflops.png"
