#!/usr/bin/env bash
# Platform benchmark (#737) on the i7-12700K: the 8 P-cores, one thread per
# physical core, under the `performance` governor.
#
# The P-cores are read from /sys/devices/cpu_core/cpus (the hybrid PMU's list;
# the E-cores are under cpu_atom), then reduced to the first SMT sibling of each
# core: 0,2,...,14 on this board. E-cores and siblings are excluded, as in
# i7-12700k.sh, because a team that mixes them runs at the slowest member's pace.
#
# Usage:  benchmarks/machines/i7-12700k-platform.sh [platform.sh args...]
#         CLASS=large benchmarks/machines/i7-12700k-platform.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_platform_profile.sh"
require_machine "i7-12700K" "i7-12700K"

[ -r /sys/devices/cpu_core/cpus ] || { echo "no /sys/devices/cpu_core/cpus: not a hybrid Intel part?" >&2; exit 2; }
pcores=$(cat /sys/devices/cpu_core/cpus)            # e.g. 0-15
lo=${pcores%-*}; hi=${pcores#*-}
CPUS=$(one_cpu_per_core $(seq "$lo" "$hi"))
[ "$(printf '%s' "$CPUS" | tr ',' '\n' | grep -c .)" -eq 8 ] || { echo "expected 8 P-cores, got '$CPUS'" >&2; exit 2; }

apply_governor performance "$CPUS"
turbo=on
[ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo 0)" = 1 ] && turbo=off
run_platform i7-12700k "$CPUS" "governor performance on the P-cores, turbo $turbo; $(describe_policies "$CPUS")" "$@"
