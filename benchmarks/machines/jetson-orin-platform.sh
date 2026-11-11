#!/usr/bin/env bash
# Platform benchmark (#737) on the Jetson Orin Nano: all six Cortex-A78AE cores,
# at a fixed nvpmodel power mode with jetson_clocks.
#
# The power mode decides the clock caps, so it decides what the numbers mean
# (see jetson-orin-nano.sh). JETSON_NVPMODEL=<id> selects one; by default the
# current mode is kept and made explicit, so the session never changes a mode
# the operator chose. jetson_clocks pins the DVFS clocks at the mode's maximum;
# the previous clocks are stored first and restored on exit.
#
# Usage:  benchmarks/machines/jetson-orin-platform.sh [platform.sh args...]
#         JETSON_NVPMODEL=0 benchmarks/machines/jetson-orin-platform.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_platform_profile.sh"
require_machine "Orin" "Jetson Orin"

CPUS=$(one_cpu_per_core)
[ "$(printf '%s' "$CPUS" | tr ',' '\n' | grep -c .)" -eq 6 ] || { echo "expected 6 cores, got '$CPUS'" >&2; exit 2; }

mode_name() { nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode:[[:space:]]*//p' | head -1; }
mode_id()   { nvpmodel -q 2>/dev/null | sed -n '2p' | tr -d '[:space:]'; }
policy="not applied"
if need_root "nvpmodel and jetson_clocks"; then
    if [ -n "${JETSON_NVPMODEL:-}" ] && [ "$JETSON_NVPMODEL" != "$(mode_id)" ]; then
        old=$(mode_id)
        sudo -n nvpmodel -m "$JETSON_NVPMODEL" > /dev/null
        on_exit "sudo -n nvpmodel -m $old > /dev/null"
    fi
    store=$(mktemp)
    sudo -n jetson_clocks --store "$store" > /dev/null
    on_exit "sudo -n jetson_clocks --restore $store > /dev/null; rm -f $store"
    sudo -n jetson_clocks > /dev/null
    policy="nvpmodel $(mode_name) (mode $(mode_id)), jetson_clocks; $(describe_policies "$CPUS")"
fi
run_platform jetson-orin "$CPUS" "$policy" "$@"
