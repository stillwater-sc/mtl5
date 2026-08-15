#!/usr/bin/env bash
# Multi-core GEMM scaling (#108): measure how the native-fast blocked GEMM scales
# across threads, alongside threaded OpenBLAS / MKL, on the same machine.
#
# Threading is a RUNTIME axis of the same per-backend binary:
#   native-fast : MTL5_NUM_THREADS=T
#   openblas    : OPENBLAS_NUM_THREADS=T
#   blis        : BLIS_NUM_THREADS=T   (if a BLIS BLAS is found)
#   mkl         : MKL_NUM_THREADS=T
# For T threads we pin to the first T physical performance cores (one logical id
# per core -- HT siblings excluded) so scaling reflects cores, not SMT.
#
# Writes one CSV per backend (gemm_scaling_<backend>.csv) whose `backend` column
# is labelled "<backend>-t<T>", so analyze_scaling.py can recover (backend, T).
#
# This harness follows the run contract in docs/benchmarks/systems.md: preflight
# gates, pinning, a per-machine OUTDIR, and machine state in every sidecar. It
# does NOT interleave -- see the note above the measurement phase for why that
# needs bench_all to grow append support first.
#
# Environment:
#   OUTDIR       REQUIRED. Where the CSVs land, one directory PER MACHINE, e.g.
#                benchmarks/data/i7-12700k. There is deliberately no default:
#                the CSVs are named by backend, not by machine, and this script
#                deletes them before writing -- a shared default silently
#                destroys another machine's committed results, which is exactly
#                what happened once (#439).
#   BENCH_PCPUS  comma list of physical-core logical ids to pin to, longest
#                first-T prefix is used. Default for an i7-12700K: one sibling
#                per P-core (0,2,4,6,8,10,12,14). Set to match YOUR topology
#                (see: lscpu -e=CPU,CORE,MAXMHZ).
#   THREADS      thread counts to sweep (default "1 2 4 8").
#   SCALE_SIZES  GEMM sizes (default "1024,2048").
#   MKL_SETVARS  oneAPI setvars.sh (default /opt/intel/oneapi/setvars.sh); mkl
#                skipped if absent.
#   ALLOW_DIRTY=1  permit a dirty working tree (recorded as tree_git_dirty=1).
#
# Example:
#   OUTDIR=benchmarks/data/i7-12700k BENCH_PCPUS=0,2,4,6,8,10,12,14 \
#       benchmarks/run_scaling.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PCPUS="${BENCH_PCPUS:-0,2,4,6,8,10,12,14}"
THREADS="${THREADS:-1 2 4 8}"
SIZES="${SCALE_SIZES:-1024,2048}"
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

IFS=',' read -r -a PCPU_ARR <<< "$PCPUS"

TMAX=0
for T in $THREADS; do
    if ! [ "$T" -ge 1 ] 2>/dev/null; then
        echo "THREADS: '$T' is not a positive integer" >&2; exit 2
    fi
    if (( T > ${#PCPU_ARR[@]} )); then
        echo "error: T=$T exceeds the ${#PCPU_ARR[@]} physical core(s) in BENCH_PCPUS ($PCPUS);" >&2
        echo "       add more cores to BENCH_PCPUS or reduce THREADS." >&2
        exit 2
    fi
    (( T > TMAX )) && TMAX=$T
done

# Gate the session, and record the state it ran in (#442). Called twice on
# purpose: once BEFORE the builds, so a dirty tree or a competing compile fails
# in seconds rather than after ten minutes of configuring, and once after them,
# because the builds themselves heat the machine and the temperature that
# matters is the one the MEASUREMENTS start at. The second record is the one
# that goes in the sidecars.
preflight_gate() {
    local kv
    if ! kv=$("$SCRIPT_DIR/preflight.sh" --threads "$TMAX" --repo "$ROOT" \
                  --ignore-path "$DATA"); then
        echo "preflight failed -- not measuring. Fix the above, or set ALLOW_DIRTY=1" >&2
        echo "if you accept a dirty tree (it is then recorded as such)." >&2
        exit 1
    fi
    printf '%s\n' "$kv"
}
preflight_gate >/dev/null

configure_build() {
    local dir="$1"; shift
    rm -rf "$dir"
    cmake -B "$dir" -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null
    cmake --build "$dir" --target bench_all -j"${JOBS:-4}" >/dev/null
}

# Every sidecar this session writes, so machine state can be appended to all of
# them at the end.
SIDECARS=()

# run_scaling_for <build-dir> <backend> <thread-env-var>
run_scaling_for() {
    local bin="$1/benchmarks/bench_all" backend="$2" tvar="$3"
    local out="$DATA/gemm_scaling_${backend}.csv"; rm -f "$out" "$out.sysinfo"
    local first=1
    for T in $THREADS; do
        local pin; pin="$(pcpus_for "$T")"
        local tmp; tmp="$(mktemp)"
        echo "  $backend  T=$T  pinned to CPUs $pin"
        env "$tvar=$T" OMP_NUM_THREADS="$T" \
            taskset -c "$pin" "$bin" --suite gemm --sizes "$SIZES" \
            --label "${backend}-t${T}" --csv "$tmp" >/dev/null
        if [[ $first -eq 1 ]]; then
            cat "$tmp" > "$out"
            # bench_all writes <csv>.sysinfo next to the CSV it was given; the
            # per-T runs are merged into one file, so keep the first one.
            [[ -f "$tmp.sysinfo" ]] && mv "$tmp.sysinfo" "$out.sysinfo"
            first=0
        else
            tail -n +2 "$tmp" >> "$out"
            rm -f "$tmp.sysinfo"
        fi
        rm -f "$tmp"
    done
    SIDECARS+=("$out.sysinfo")
    echo "  -> $out"
}

# pcpus_for <T>: comma list of the first T physical-core ids.
pcpus_for() {
    local t="$1" out=""
    for ((i = 0; i < t && i < ${#PCPU_ARR[@]}; ++i)); do
        out="${out:+$out,}${PCPU_ARR[$i]}"
    done
    printf '%s' "$out"
}

# ── build phase ─────────────────────────────────────────────────────────────
# Every variant is built BEFORE any of them is measured. Building and measuring
# in turn meant each backend was timed on a machine still hot from compiling its
# own binary, while the next one compiled during the previous one's cooldown --
# a per-arm bias in a comparison whose entire purpose is comparing arms.
echo "=== building all variants (nothing is measured yet) ==="
BACKENDS=()   # label
BUILDDIRS=()  # build directory
TVARS=()      # threading environment variable
MKL_ARM=-1    # index of the mkl arm, which needs setvars sourced to RUN

configure_build build-scaling-native-fast \
    -DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON
BACKENDS+=(native-fast); BUILDDIRS+=(build-scaling-native-fast); TVARS+=(MTL5_NUM_THREADS)

configure_build build-scaling-openblas -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON
BACKENDS+=(openblas); BUILDDIRS+=(build-scaling-openblas); TVARS+=(OPENBLAS_NUM_THREADS)

# BLIS (BLA_VENDOR=FLAME, BLIS_NUM_THREADS); skipped if not found at configure.
if cmake -B build-scaling-blis-probe -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
        -DMTL5_WITH_BLAS=ON -DBLA_VENDOR=FLAME >/dev/null 2>&1; then
    rm -rf build-scaling-blis-probe
    configure_build build-scaling-blis -DMTL5_WITH_BLAS=ON -DBLA_VENDOR=FLAME
    BACKENDS+=(blis); BUILDDIRS+=(build-scaling-blis); TVARS+=(BLIS_NUM_THREADS)
else
    rm -rf build-scaling-blis-probe
    echo "=== blis: SKIPPED (no FLAME/BLIS BLAS found) ==="
fi

if [[ -f "$MKL_SETVARS" ]]; then
    # setvars.sh is sourced in a SUBSHELL, here and at run time. Sourcing it into
    # this shell would leave oneAPI's LD_LIBRARY_PATH ahead of every later run,
    # so the openblas and blis binaries could resolve MKL's libraries and be
    # labelled as themselves -- an arm measuring another arm.
    ( set +u +e
      # shellcheck disable=SC1090
      source "$MKL_SETVARS" >/dev/null 2>&1 || true
      set -u -e
      configure_build build-scaling-mkl -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON \
                      -DBLA_VENDOR=Intel10_64lp )
    BACKENDS+=(mkl); BUILDDIRS+=(build-scaling-mkl); TVARS+=(MKL_NUM_THREADS)
    MKL_ARM=$(( ${#BACKENDS[@]} - 1 ))
else
    echo "=== mkl: SKIPPED (no $MKL_SETVARS) ==="
fi

# ── measurement phase ───────────────────────────────────────────────────────
# Gate again now that the builds are done: this is the temperature and the load
# the numbers are actually taken under, and it is what the sidecars record.
#
# NOT interleaved, unlike run_blocking_ab. Interleaving needs several rounds per
# arm, and bench_all truncates its --csv on every invocation while
# analyze_scaling.py keeps the LAST row for a (backend, op, size, T) key rather
# than the best -- so rounds would silently overwrite instead of accumulating.
# Doing it properly means append support in bench_all and min-of-rounds in the
# analyzer; until then, building everything up front is what removes the largest
# order effect. Tracked in #442.
PREFLIGHT_KV=$(preflight_gate)
echo "=== measuring ==="
for i in "${!BACKENDS[@]}"; do
    echo "=== ${BACKENDS[$i]} (${TVARS[$i]}) ==="
    if [[ $i -eq $MKL_ARM ]]; then
        ( set +u +e
          # shellcheck disable=SC1090
          source "$MKL_SETVARS" >/dev/null 2>&1 || true
          set -u -e
          run_scaling_for "${BUILDDIRS[$i]}" "${BACKENDS[$i]}" "${TVARS[$i]}" )
        SIDECARS+=("$DATA/gemm_scaling_${BACKENDS[$i]}.csv.sysinfo")
    else
        run_scaling_for "${BUILDDIRS[$i]}" "${BACKENDS[$i]}" "${TVARS[$i]}"
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
    printf 'harness=run_scaling.sh\nharness_pinned_cpus=%s\nharness_interleaved=0\n' "$PCPUS" >> "$s"
done

echo
echo "Done. Analyze with:"
echo "  ./benchmarks/analyze_scaling.py $DATA/gemm_scaling_*.csv --plot $DATA/gemm_scaling.png"
