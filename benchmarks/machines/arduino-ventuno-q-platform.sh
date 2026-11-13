#!/usr/bin/env bash
# Platform benchmark (#737) on the Arduino Ventuno Q (Qualcomm Monaco): the four
# Cortex-A78C cores, under the `performance` governor on their cpufreq
# policies. The A55s are excluded.
#
# The A78C cores are found by MIDR part number (0xd4b) in /proc/cpuinfo, as in
# arduino-ventuno-q.sh, not assumed. Its kernel exposes no cache sizes, so the
# card's last-level-cache probe reads 0 (a caveat for the card, #737).
#
# Usage:  benchmarks/machines/arduino-ventuno-q-platform.sh [platform.sh args...]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_platform_profile.sh"
require_machine "Monaco" "Arduino Ventuno Q (Qualcomm Monaco)"

CPUS=$(awk -F: '/^processor/ { cpu = $2 + 0 }
                /^CPU part/  { gsub(/[[:space:]]/, "", $2); if ($2 == "0xd4b") ids = ids (ids ? "," : "") cpu }
                END { print ids }' /proc/cpuinfo)
[ "$(printf '%s' "$CPUS" | tr ',' '\n' | grep -c .)" -eq 4 ] || { echo "expected 4 A78C cores (part 0xd4b), got '$CPUS'" >&2; exit 2; }

apply_governor performance "$CPUS"
# GCC 13 resolves -march=native to baseline Armv8.0-a on this board (#806): it
# passes no -march or -mcpu on to cc1, so native-fast lost DotProd and the quad
# GEMM ran Highway's emulation. The flags spell out what /proc/cpuinfo lists on
# every core (asimddp, asimdhp, lrcpc on Armv8.2) and tune for the A78C cores
# checked above. Not -mcpu=cortex-a78c: that also enables PAuth, which this
# kernel doesn't report, and it measured within 4% of these.
run_platform arduino-ventuno-q "$CPUS" "governor performance on the A78C policies; $(describe_policies "$CPUS")" \
    --arch "-march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c" "$@"
