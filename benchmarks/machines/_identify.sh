#!/usr/bin/env bash
# Shared machine-identity guard for the per-machine run profiles (#439).
#
# WHY THIS IS A SHARED FILE. Eleven profiles carried a copy of this check, and
# the copies had drifted: the `-int` ones read /proc/device-tree/model first and
# worked everywhere, while the `-nc-*` ones used
#
#     grep -m1 'model name' /proc/cpuinfo || tr -d '\0' < /proc/device-tree/model
#
# `A || B` runs B only when A FAILS. On ARM, `grep 'model name' /proc/cpuinfo`
# SUCCEEDS and returns a generic "ARMv8 Processor rev 1 (v8l)" -- no board
# identity at all -- so the device-tree fallback never ran on the one
# architecture that needs it. Every nc-* run on the Jetson had to be forced past
# its own guard, which means the guard was not protecting anything there.
#
# READS BOTH SOURCES AND MATCHES THE UNION, rather than picking one and hoping:
#   /proc/device-tree/model   the board (ARM SBCs: "NVIDIA Jetson Orin Nano ...")
#   /proc/cpuinfo model name  the CPU   (x86: "Intel(R) Xeon(R) CPU E5-2420 v2")
# Neither is present on every machine and neither alone identifies every one.
#
# Usage, from a profile:
#     . "$(dirname "${BASH_SOURCE[0]}")/_identify.sh"
#     require_machine "Orin"          # substring, case-sensitive
#     echo "$MTL5_MACHINE_ID"         # what it matched against
#
# FORCE=1 overrides, and the message says so -- an operator who knows the guard
# is wrong should not be stuck, but should have to say it out loud.

mtl5_machine_id() {
    local dt="" cpu="" id=""
    [ -r /proc/device-tree/model ] && dt="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
    cpu="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^[[:space:]]*//')"
    # Apple silicon and some BSDs have neither; sysctl is the usual last resort.
    if [ -z "$dt" ] && [ -z "$cpu" ] && command -v sysctl >/dev/null 2>&1; then
        cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    fi
    id="$dt${dt:+ / }$cpu"
    printf '%s' "${id:-unknown}"
}

MTL5_MACHINE_ID="$(mtl5_machine_id)"

# require_machine <substring> [<friendly name>]
require_machine() {
    local want="$1" friendly="${2:-$1}"
    case "$MTL5_MACHINE_ID" in
        *"$want"*) return 0 ;;
    esac
    echo "This profile is for a $friendly; this machine reports:" >&2
    echo "  $MTL5_MACHINE_ID" >&2
    if [ "${FORCE:-0}" = "1" ]; then
        echo "  FORCE=1 set -- proceeding anyway." >&2
        return 0
    fi
    echo "Refusing to write $friendly data into a shared results directory: a" >&2
    echo "mislabelled CSV silently overwrites another machine's committed run (#439)." >&2
    echo "Set FORCE=1 if this really is one." >&2
    exit 2
}
