#!/usr/bin/env bash
# A vendor BLAS called from inside an MTL5 region, per vendor (#640).
#
# Builds bench_vendor_nested once per vendor library -- OpenBLAS (FindBLAS's
# default), BLIS (BLA_VENDOR=FLAME) and MKL (BLA_VENDOR=Intel10_64lp, the
# threaded MKL, if oneAPI is installed) -- then runs each with an MTL5 pool of
# THREADS and the vendor's own team set to THREADS too: the configuration in
# which an uncapped vendor call puts THREADS x THREADS threads on THREADS cores.
#
# Usage:
#   OUTDIR=benchmarks/data/<machine>-vendor-nested BENCH_PCPUS=0,2,4,6,8,10,12,14 \
#       benchmarks/run_vendor_nested.sh
# Environment:
#   OUTDIR       required; one directory per machine
#   BENCH_PCPUS  cores to pin to (one logical cpu per physical core); THREADS
#                defaults to their count
#   VENDORS      default "openblas blis mkl"; a vendor that cannot be found is skipped
#   SIZES BATCH REPS   passed to the benchmark (defaults 128,256,512 / 64 / 7)
#   MKL_SETVARS  oneAPI's setvars.sh (default /opt/intel/oneapi/setvars.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [ -z "${OUTDIR:-}" ]; then
    echo "OUTDIR is required: give this machine its own directory, e.g." >&2
    echo "  OUTDIR=benchmarks/data/<machine>-vendor-nested $0" >&2
    exit 2
fi
mkdir -p "$OUTDIR"

VENDORS=${VENDORS:-"openblas blis mkl"}
SIZES=${SIZES:-128,256,512}
BATCH=${BATCH:-64}
REPS=${REPS:-7}
MKL_SETVARS=${MKL_SETVARS:-/opt/intel/oneapi/setvars.sh}

PIN=()
if [ -n "${BENCH_PCPUS:-}" ]; then
    PIN=(taskset -c "$BENCH_PCPUS")
    NPIN=$(printf '%s' "$BENCH_PCPUS" | tr ',' '\n' | grep -c .)
    THREADS=${THREADS:-$NPIN}
else
    echo "WARNING: BENCH_PCPUS not set -- no pinning; on a hybrid CPU the teams" >&2
    echo "         land on E-cores too, and the numbers mix core classes." >&2
    THREADS=${THREADS:-$(nproc)}
fi

PREFLIGHT=$("$SCRIPT_DIR/preflight.sh" --threads "$THREADS" --repo "$ROOT" \
                --ignore-path "$OUTDIR") || {
    echo "preflight failed -- not measuring (ALLOW_DIRTY=1 accepts a dirty tree)" >&2
    exit 1
}

# Builds first, all of them, so no measurement runs beside a compiler.
configure_build() {   # vendor, then extra cmake arguments
    local v=$1; shift
    local dir=build-vendor-$v
    cmake -B "$dir" -DCMAKE_BUILD_TYPE=Release -DMTL5_BUILD_BENCHMARKS=ON \
          -DMTL5_BUILD_TESTS=OFF -DMTL5_BUILD_EXAMPLES=OFF -DMTL5_WITH_BLAS=ON \
          "$@" >"$OUTDIR/.cmake-$v.log" 2>&1 || return 1
    grep -q "BLAS/LAPACK vendor: $v" "$OUTDIR/.cmake-$v.log" || {
        echo "== $v: configured, but the vendor detected was not $v:" >&2
        grep 'BLAS/LAPACK vendor' "$OUTDIR/.cmake-$v.log" >&2 || true
        return 1
    }
    cmake --build "$dir" --target bench_vendor_nested -j4 >/dev/null
}

BUILT=()
for v in $VENDORS; do
    echo "== building $v"
    case $v in
        openblas) configure_build openblas -DBLA_VENDOR=OpenBLAS && BUILT+=(openblas) ;;
        blis)     configure_build blis -DBLA_VENDOR=FLAME && BUILT+=(blis) ;;
        mkl)
            if [ -f "$MKL_SETVARS" ] &&
               ( set +u +e; source "$MKL_SETVARS" >/dev/null 2>&1; set -u -e
                 configure_build mkl -DBLA_VENDOR=Intel10_64lp ); then
                BUILT+=(mkl)
            fi ;;
        *) echo "unknown vendor '$v'" >&2; exit 2 ;;
    esac
    printf '%s\n' "${BUILT[@]}" | grep -qx "$v" || echo "== $v: SKIPPED (not found)"
done
rm -f "$OUTDIR"/.cmake-*.log

for v in "${BUILT[@]}"; do
    csv=$OUTDIR/vendor_nested_$v.csv
    rm -f "$csv" "$csv.sysinfo"
    echo "== running $v: MTL5 pool $THREADS, vendor team $THREADS"
    # One process per arm: a vendor's idle team spins on after its last call,
    # and the next arm in the same process would be timed against it.
    arms="uncapped capped direct ddot"
    [ "$v" = openblas ] && arms="uncapped capped process1 direct ddot"
    run() {   # arm, then VAR=value settings
        local arm=$1; shift
        env MTL5_NUM_THREADS="$THREADS" "$@" ${PIN[@]+"${PIN[@]}"} \
            "build-vendor-$v/benchmarks/bench_vendor_nested" --arms "$arm" \
            --sizes "$SIZES" --batch "$BATCH" --reps "$REPS" --csv "$csv"
    }
    for arm in $arms; do
        case $v in
            openblas) run "$arm" OPENBLAS_NUM_THREADS="$THREADS" ;;
            blis)     run "$arm" BLIS_NUM_THREADS="$THREADS" ;;
            mkl)      ( set +u +e; source "$MKL_SETVARS" >/dev/null 2>&1; set -u -e
                        run "$arm" MKL_NUM_THREADS="$THREADS" ) ;;
        esac
    done
    { printf '%s\n' "$PREFLIGHT"
      "$SCRIPT_DIR/preflight.sh" --phase after --repo "$ROOT" --report-only; } >>"$csv.sysinfo"
done
