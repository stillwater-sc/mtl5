#!/usr/bin/env bash
# Characterize this machine with mtl5_bench: the one command (#721 M2).
#
#   git clone https://github.com/stillwater-sc/mtl5 && cd mtl5
#   benchmarks/platform.sh
#
# What it does, in order:
#   1. preflight.sh gates the session (dirty tree, competing load, governor,
#      thermals) and records the machine state. A failed gate stops here, before
#      ten minutes of building. --no-gate records without gating (CI, or a first
#      look at a machine), and says so in the data.
#   2. Configures and builds mtl5_bench once per VARIANT, each in its own clean
#      build directory -- the build is the backend (benchmarks/README.md):
#        native       generic C++, no SIMD library
#        native-fast  Highway SIMD + MTL5's blocked GEMM, -march=native where
#                     that is safe (not on SVE hardware: #720; not on macOS)
#        blas         system BLAS/LAPACK, if CMake finds one (else skipped)
#      Every variant is built before any is measured, so no measurement runs on
#      a machine still hot from compiling the next one.
#   3. preflight again, after the builds: the temperature that matters is the
#      one the measurements start at.
#   4. Runs every variant at every thread count, writing per run:
#        <variant>-t<T>.json + .sysinfo   (results + provenance, machine state
#                                          appended from preflight)
#        <variant>-t<T>.md                (capability card)
#      and summary.md, all the cards in one file, ready for
#      docs/benchmarks/systems.md.
#
# Options:
#   --class small|medium|large   problem sizes (default: medium)
#   --variants LIST              comma list (default: native,native-fast,blas;
#                                blas is skipped if no BLAS is found)
#   --threads LIST               comma list of thread counts (default: 1 and
#                                the number of online CPUs)
#   --jobs N                     parallel compile jobs (default: half the CPUs,
#                                at least 1 -- template-heavy translation units
#                                exhaust memory on small boards before CPUs)
#   --out DIR                    results directory (default:
#                                build-platform/results/<host>-<UTC time>)
#   --machine NAME               machine name for the cards (default: the
#                                device-tree board model on ARM boards, else
#                                the CPU brand)
#   --no-gate                    record the machine state but never stop on it
#
# Exit: 0 if every run passed every check; 1 if a gate, a build or a check
# failed; 2 on a usage error.
#
# Portable to bash 3.2 (macOS's /bin/bash): no mapfile, no associative arrays,
# and every expansion of a possibly-empty array is guarded, because 3.2 treats
# "${empty[@]}" as an unbound variable under set -u.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)
CLASS=medium
VARIANTS="native,native-fast,blas"
THREADS=""
JOBS=$(( ncpu / 2 > 0 ? ncpu / 2 : 1 ))
OUT=""
MACHINE=""
GATE=1

need_value() { [ "$2" -ge 2 ] || { echo "platform.sh: $1 requires a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
    case "$1" in
        --class)    need_value "$1" $#; CLASS=$2; shift 2 ;;
        --variants) need_value "$1" $#; VARIANTS=$2; shift 2 ;;
        --threads)  need_value "$1" $#; THREADS=$2; shift 2 ;;
        --jobs)     need_value "$1" $#; JOBS=$2; shift 2 ;;
        --out)      need_value "$1" $#; OUT=$2; shift 2 ;;
        --machine)  need_value "$1" $#; MACHINE=$2; shift 2 ;;
        --no-gate)  GATE=0; shift ;;
        -h|--help)  sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "platform.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
case "$CLASS" in small|medium|large) ;; *) echo "platform.sh: --class must be small, medium or large" >&2; exit 2 ;; esac
case "$JOBS" in ''|*[!0-9]*) echo "platform.sh: --jobs must be a number" >&2; exit 2 ;; esac
[ -n "$THREADS" ] || { THREADS=1; [ "$ncpu" -gt 1 ] && THREADS="1,$ncpu"; }
IFS=',' read -r -a THREAD_LIST <<< "$THREADS"
IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
max_threads=1
for t in "${THREAD_LIST[@]}"; do
    case "$t" in ''|*[!0-9]*|0) echo "platform.sh: bad thread count '$t'" >&2; exit 2 ;; esac
    [ "$t" -gt "$max_threads" ] && max_threads=$t
done
for v in "${VARIANT_LIST[@]}"; do
    case "$v" in native|native-fast|blas) ;; *) echo "platform.sh: unknown variant '$v'" >&2; exit 2 ;; esac
done

BUILD_ROOT="$ROOT/build-platform"
host=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo host)
[ -n "$OUT" ] || OUT="$BUILD_ROOT/results/${host}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT/logs" || { echo "platform.sh: cannot create $OUT" >&2; exit 1; }
OUT="$(cd "$OUT" && pwd)"

if [ -z "$MACHINE" ] && [ -r /proc/device-tree/model ]; then
    MACHINE=$(tr -d '\0' < /proc/device-tree/model)
fi

echo "mtl5 platform benchmark: class=$CLASS variants=$VARIANTS threads=$THREADS jobs=$JOBS"
echo "results: $OUT"

# -- 1. preflight, before the builds ------------------------------------------
preflight() {   # extra args...
    # No --ignore-path: build-platform/ is gitignored (build-*/), so it can never
    # make the tree dirty.
    local args=(--threads "$max_threads" --repo "$ROOT")
    [ "$GATE" -eq 1 ] || args+=(--report-only)
    "$SCRIPT_DIR/preflight.sh" "${args[@]}" "$@"
}
if ! preflight > "$OUT/logs/preflight-before-build.txt"; then
    echo "preflight failed -- not measuring. Fix the above, pass --no-gate to record" >&2
    echo "without gating, or set ALLOW_DIRTY=1 to accept a dirty tree (it is recorded)." >&2
    exit 1
fi

# -- 2. build every variant -----------------------------------------------------
has_sve=0
grep -qw sve /proc/cpuinfo 2>/dev/null && has_sve=1

variant_flags() {   # variant -> cmake args on stdout, one per line
    case "$1" in
        native) ;;
        native-fast)
            printf '%s\n' -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_FAST_GEMM=ON
            if [ "$has_sve" -eq 1 ]; then
                echo "  native-fast: SVE hardware -- not using -march=native, which demotes" >&2
                echo "  the SIMD batch to one lane with GCC (#720); baseline NEON instead." >&2
            elif [ "$(uname -s)" = Darwin ]; then
                echo "  native-fast: macOS -- baseline target, no -march=native." >&2
            else
                echo -DMTL5_NATIVE_ARCH=ON
            fi ;;
        blas) printf '%s\n' -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON ;;
    esac
}

BUILT=()
SKIPPED=()
failed=0
for v in "${VARIANT_LIST[@]}"; do
    dir="$BUILD_ROOT/$v"
    rm -rf "$dir"
    extra=()
    while IFS= read -r f; do [ -n "$f" ] && extra+=("$f"); done < <(variant_flags "$v")
    echo "== build $v"
    if ! cmake -S "$ROOT" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DMTL5_BUILD_BENCHMARKS=ON \
           -DMTL5_BUILD_TESTS=OFF -DMTL5_BUILD_EXAMPLES=OFF ${extra[@]+"${extra[@]}"} \
           > "$OUT/logs/$v.configure.log" 2>&1; then
        if [ "$v" = blas ]; then
            echo "   skipped: no BLAS/LAPACK found (see logs/$v.configure.log)"
            SKIPPED+=("$v: no BLAS/LAPACK found")
            continue
        fi
        echo "   configure FAILED (see $OUT/logs/$v.configure.log)" >&2
        SKIPPED+=("$v: configure failed"); failed=1; continue
    fi
    if ! cmake --build "$dir" --target mtl5_bench --parallel "$JOBS" > "$OUT/logs/$v.build.log" 2>&1; then
        echo "   build FAILED (see $OUT/logs/$v.build.log)" >&2
        SKIPPED+=("$v: build failed"); failed=1; continue
    fi
    BUILT+=("$v")
done
[ "${#BUILT[@]}" -gt 0 ] || { echo "nothing built -- no measurements" >&2; exit 1; }

# -- 3. preflight, after the builds ---------------------------------------------
if ! preflight > "$OUT/logs/preflight.txt"; then
    echo "preflight failed after the builds -- not measuring (see above)" >&2
    exit 1
fi

# -- 4. measure -----------------------------------------------------------------
RUNS=()
for v in "${BUILT[@]}"; do
    exe="$BUILD_ROOT/$v/benchmarks/mtl5_bench"
    for t in "${THREAD_LIST[@]}"; do
        base="$OUT/$v-t$t"
        echo "== run $v, $t thread(s)"
        args=(--class "$CLASS" --label "$v" --json "$base.json" --card "$base.md")
        [ -n "$MACHINE" ] && args+=(--machine "$MACHINE")
        # Every library's pool is pinned to the same count: OpenBLAS, MKL and BLIS
        # start one thread per core by default, so a "1-thread" BLAS run would
        # otherwise use all of them (a 1-thread blas3 measured 202% of the
        # 1-thread ceiling before this).
        if MTL5_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t BLIS_NUM_THREADS=$t \
           OMP_NUM_THREADS=$t "$exe" "${args[@]}" > "$OUT/logs/$v-t$t.log" 2>&1; then
            echo "   $(tail -1 "$OUT/logs/$v-t$t.log")"
        else
            echo "   FAILED: $(tail -1 "$OUT/logs/$v-t$t.log") (see logs/$v-t$t.log)" >&2
            failed=1
        fi
        # Machine state into the sidecar, so it is IN the data (#442).
        if [ -f "$base.json.sysinfo" ]; then
            { cat "$OUT/logs/preflight.txt"
              after=$("$SCRIPT_DIR/preflight.sh" --phase after 2>/dev/null) || after=""
              case "$after" in *thermal_after_c=*) echo "$after" ;; *) echo "thermal_after_c=unavailable" ;; esac
            } >> "$base.json.sysinfo"
            RUNS+=("$base")
        fi
    done
done

# -- summary --------------------------------------------------------------------
{
    echo "# mtl5_bench platform characterization"
    echo
    echo "Class \`$CLASS\`, variants ${BUILT[*]}, threads $THREADS, $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    if [ "${#SKIPPED[@]}" -gt 0 ]; then
        echo; echo "Not measured:"; printf -- '- %s\n' ${SKIPPED[@]+"${SKIPPED[@]}"}
    fi
    echo
    for r in ${RUNS[@]+"${RUNS[@]}"}; do [ -f "$r.md" ] && cat "$r.md"; done
} > "$OUT/summary.md"

echo
echo "summary: $OUT/summary.md"
exit "$failed"
