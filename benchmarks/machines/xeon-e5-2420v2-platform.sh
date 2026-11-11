#!/usr/bin/env bash
# Platform benchmark (#737) on the Xeon E5-2420 v2 (Ivy Bridge-EP, 6 cores,
# SMT): one thread per physical core, under the `performance` governor.
#
# One CPU per core, read from lscpu: the SMT siblings (6-11 on this board)
# are excluded, since two threads on one core share its FMA units.
#
# Usage:  benchmarks/machines/xeon-e5-2420v2-platform.sh [platform.sh args...]
#         CLASS=large benchmarks/machines/xeon-e5-2420v2-platform.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_platform_profile.sh"
require_machine "E5-2420 v2" "Xeon E5-2420 v2"

CPUS=$(one_cpu_per_core)
[ "$(printf '%s' "$CPUS" | tr ',' '\n' | grep -c .)" -eq 6 ] || { echo "expected 6 cores, got '$CPUS'" >&2; exit 2; }

apply_governor performance "$(seq -s, 0 $(( $(nproc --all) - 1 )))"   # siblings too: they share the core clock
turbo=on
if [ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = 1 ] && turbo=off
elif [ -r /sys/devices/system/cpu/cpufreq/boost ]; then
    [ "$(cat /sys/devices/system/cpu/cpufreq/boost)" = 0 ] && turbo=off
fi
run_platform xeon-e5-2420v2 "$CPUS" "governor performance on every CPU, turbo $turbo; $(describe_policies "$CPUS")" "$@"
