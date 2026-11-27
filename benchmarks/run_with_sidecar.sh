#!/usr/bin/env bash
# Run one benchmark command that writes CSV and CSV.sysinfo, and complete the
# sidecar with the run's keys (#930). For the binaries that have no harness
# script of their own: the SuperLU and KLU scoreboards, a one-off bench_sparse.
#
#   [BENCH_CPUS=4] [BENCH_MACHINE=i7-12700k] [CLOCK_POLICY=...] \
#   benchmarks/run_with_sidecar.sh CSV CLASS -- COMMAND [ARGS...]
#
#   CSV      the file COMMAND writes (its binary writes CSV.sysinfo beside it,
#            the build's and the machine's keys: benchmarks/harness/sidecar.hpp)
#   CLASS    what the run measured, for the sidecar's class key
#   COMMAND  run pinned to BENCH_CPUS with taskset when that is set
#
# In order: preflight (report only: this wrapper records the machine state, it
# does not gate on it), the command, then the preflight record, benchmark (the
# command's file name), class, bench_cpus and clock_policy into CSV.sysinfo.
# Exit: the command's status, or 1 when the sidecar still lacks a required
# key (benchmarks/sidecar_keys.txt); 2 on bad usage.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 4 ] && [ "$3" = -- ] || { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d' >&2; exit 2; }
CSV=$1 CLASS=$2; shift 3
mkdir -p "$(dirname "$CSV")"
PREFLIGHT_KV=$("$SCRIPT_DIR/preflight.sh" --repo "$SCRIPT_DIR/.." --ignore-path "$(dirname "$CSV")" \
                --report-only 2>/dev/null || true)
pin=()
[ -n "${BENCH_CPUS:-}" ] && pin=(taskset -c "$BENCH_CPUS")
rc=0
"${pin[@]+"${pin[@]}"}" "$@" || rc=$?
[ "$rc" -eq 0 ] || { echo "run_with_sidecar.sh: the command failed (exit $rc)" >&2; exit "$rc"; }
[ -f "$CSV.sysinfo" ] || { echo "run_with_sidecar.sh: $CSV.sysinfo was not written -- does $(basename "$1") write a sidecar with --csv?" >&2; exit 1; }
printf '%s\n' "$PREFLIGHT_KV" >> "$CSV.sysinfo"
. "$SCRIPT_DIR/sidecar.sh"
sidecar_finish "$CSV.sysinfo" "benchmark=$(basename "$1")" "class=$CLASS" \
               "bench_cpus=${BENCH_CPUS:-unpinned}" "harness=run_with_sidecar.sh"
