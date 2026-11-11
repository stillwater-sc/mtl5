#!/usr/bin/env bash
# Shared helpers for the platform-benchmark machine profiles (#737):
# benchmarks/machines/<machine>-platform.sh.
#
# A profile is the whole invocation for one machine: it identifies the machine,
# derives the CPUs to measure from the topology (never hard-coded), applies and
# records the clock policy, and calls benchmarks/platform.sh with --cpus and
# --clock-policy. The clock policy needs root; every change goes through
# `sudo -n`, so a profile run without the rights fails at once instead of
# prompting, and everything it changed is put back when it exits.
#
#   NO_CLOCK_POLICY=1   measure without applying the policy; the sidecars then
#                       record clock_policy=not applied, and the card has to be
#                       published as PRELIMINARY.
#   PLATFORM_OUT=DIR    results directory, passed to platform.sh as --out.
#   INTERLEAVE=1        after the session, still under the clock policy, run
#                       bench_interleave_rows (#780) on the first measured CPU:
#                       REPS runs (default 3) with the session's native build,
#                       and as many with Clang where clang++ is installed.
#                       Written to interleave-<compiler>-<i>.txt in the results.
#
# Usage, from a profile:
#     . "$(dirname "${BASH_SOURCE[0]}")/_platform_profile.sh"
#     CPUS=$(one_cpu_per_core 0 1 2 3)       # first SMT sibling of each core
#     apply_governor performance $CPUS       # restored on exit
#     run_platform <machine> "$CPUS" "<policy text>" "$@"

PROFILE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE_ROOT=$(cd "$PROFILE_DIR/../.." && pwd)
. "$PROFILE_DIR/_identify.sh"

# Commands run by the exit trap, last registered first.
_restore=()
_run_restore() {
    local i
    for (( i=${#_restore[@]}-1; i>=0; i-- )); do eval "${_restore[$i]}" || true; done
}
trap _run_restore EXIT
on_exit() { _restore+=("$1"); }

# Whether the clock policy can be applied: 1 under NO_CLOCK_POLICY, else each
# command must be runnable through sudo -n. `sudo -n -l CMD` asks for exactly
# that, so a sudoers rule that permits only these commands (benchmarks/README.md)
# is enough; `sudo -n true` would demand far more.
need_root() {   # what-for command...
    local what=$1 cmd
    shift
    if [ "${NO_CLOCK_POLICY:-0}" = 1 ]; then return 1; fi
    for cmd in "$@"; do
        sudo -n -l "$cmd" > /dev/null 2>&1 && continue
        echo "$(basename "$0"): applying the clock policy ($what) needs passwordless sudo for $cmd." >&2
        echo "  Install the sudoers rule in benchmarks/README.md (\"Machine profiles\")," >&2
        echo "  or set NO_CLOCK_POLICY=1 and publish the card as PRELIMINARY." >&2
        exit 2
    done
    return 0
}

# CPU ids, one per physical core: of the CPUs given (default: all online), the
# first of each (core, socket) pair in lscpu's order. Comma-separated.
one_cpu_per_core() {
    local want=" $* "
    lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | awk -F, -v want="$want" '
        (want == "  " || index(want, " " $1 " ")) && !seen[$2 "," $3]++ { ids = ids (ids ? "," : "") $1 }
        END { print ids }'
}

# The cpufreq policy directories that cover the given CPUs (comma or space list).
policies_of() {
    local c p seen=" "
    for c in $(printf '%s' "$*" | tr ',' ' '); do
        p=$(readlink -f "/sys/devices/system/cpu/cpu$c/cpufreq" 2>/dev/null) || continue
        [ -d "$p" ] || continue
        case "$seen" in *" $p "*) ;; *) seen="$seen$p "; echo "$p" ;; esac
    done
}

# Set the governor on every policy covering the CPUs; restore each on exit.
apply_governor() {   # governor cpus
    local gov=$1 p old
    shift
    need_root "cpufreq governor $gov" "$(command -v tee)" || return 0
    for p in $(policies_of "$*"); do
        old=$(cat "$p/scaling_governor")
        [ "$old" = "$gov" ] && continue
        echo "$gov" | sudo -n tee "$p/scaling_governor" > /dev/null || {
            echo "$(basename "$0"): could not set $p to $gov" >&2; exit 2; }
        on_exit "echo $old | sudo -n tee $p/scaling_governor > /dev/null"
    done
}

# "performance 1200-2700 MHz on cpus 0-5" for the record: the policies covering
# the CPUs, grouped by governor and frequency range.
describe_policies() {   # cpus
    local p
    for p in $(policies_of "$*"); do
        printf '%s %s-%s MHz|%s\n' "$(cat "$p/scaling_governor")" \
            $(( $(cat "$p/scaling_min_freq") / 1000 )) $(( $(cat "$p/scaling_max_freq") / 1000 )) \
            "$(cat "$p/related_cpus" | tr ' ' ',')"
    done | awk -F'|' '{ if (!($1 in c)) o[n++] = $1; c[$1] = c[$1] (c[$1] ? "," : "") $2 }
                      END { for (i = 0; i < n; i++) printf "%s%s on cpus %s", (i ? "; " : ""), o[i], c[o[i]] }'
}

# Run the platform benchmark: 1 thread and one per listed CPU, pinned.
run_platform() {   # machine cpus policy-text [platform.sh args...]
    local machine=$1 cpus=$2 policy=$3 n
    shift 3
    [ "${NO_CLOCK_POLICY:-0}" = 1 ] && policy="not applied (NO_CLOCK_POLICY=1)"
    n=$(printf '%s' "$cpus" | tr ',' '\n' | grep -c .)
    echo "== $machine: CPUs $cpus ($n), clock policy: $policy"
    local out_args=() status=0
    [ -n "${PLATFORM_OUT:-}" ] && out_args=(--out "$PLATFORM_OUT")
    "$PROFILE_ROOT/benchmarks/platform.sh" --class "${CLASS:-medium}" --cpus "$cpus" \
        --threads "1,$n" --machine "$machine" --clock-policy "$policy" \
        ${out_args[@]+"${out_args[@]}"} "$@" || status=$?
    if [ "${INTERLEAVE:-0}" = 1 ]; then
        run_interleave "${cpus%%,*}" "$policy" || status=1
    fi
    return "$status"
}

# The interleave-width sweep (#780) on one CPU, under the policy still applied.
run_interleave() {   # cpu policy-text
    local cpu=$1 policy=$2 out=${PLATFORM_OUT:-$PROFILE_ROOT/build-platform/results/interleave} i cxx dir
    local jobs=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))
    mkdir -p "$out"
    for cxx in g++ clang++; do
        command -v "$cxx" > /dev/null 2>&1 || continue
        dir="$PROFILE_ROOT/build-platform/interleave-$cxx"
        echo "== bench_interleave_rows ($cxx)"
        cmake -S "$PROFILE_ROOT" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$cxx" \
            -DMTL5_BUILD_BENCHMARKS=ON -DMTL5_BUILD_TESTS=OFF -DMTL5_BUILD_EXAMPLES=OFF \
            > "$out/interleave-$cxx.configure.log" 2>&1 &&
        cmake --build "$dir" --target bench_interleave_rows --parallel "$jobs" \
            > "$out/interleave-$cxx.build.log" 2>&1 || { echo "   build FAILED ($cxx)" >&2; return 1; }
        for i in $(seq 1 "${REPS:-3}"); do
            { echo "# bench_interleave_rows, $($cxx --version | head -1), cpu $cpu, clock policy: $policy"
              echo "# commit $(git -C "$PROFILE_ROOT" rev-parse HEAD), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
              taskset -c "$cpu" "$dir/benchmarks/bench_interleave_rows" --reps 21
            } > "$out/interleave-$cxx-$i.txt" 2>&1 || { echo "   run FAILED ($cxx #$i)" >&2; return 1; }
        done
    done
}
