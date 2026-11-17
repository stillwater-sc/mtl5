#!/usr/bin/env bash
# The column sweep against the row kernel on this machine (#884).
#
# Builds benchmarks/bench_column_sweep.cpp with each compiler found (g++,
# clang++) in two variants, with CMake Release's optimization (-O3 -DNDEBUG):
#
#   generic  no arch flags: the portable build;
#   native   -march=native, or NATIVE_FLAGS when the compiler under-detects
#            the host (the Ventuno: see machines/arduino-ventuno-q-platform.sh).
#
# It compiles the one file directly against include/ (the library is header
# only), so a machine needs only a compiler. Each binary runs pinned to one CPU
# (CPU, default 0) with the governor of that CPU's policy at `performance`, and
# on a Jetson with jetson_clocks, both through `sudo -n` and put back on exit;
# without root they are recorded as not applied.
#
# Output: OUT (default build-platform/column-sweep/<host>-<UTC time>) holds
# <compiler>-<variant>.csv, each with its .csv.sysinfo sidecar, and the
# session's sysinfo.txt.
#
# Usage: CPU=0 NATIVE_FLAGS='-march=...' benchmarks/run_column_sweep.sh [bench args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU="${CPU:-0}"
OUT="${OUT:-$ROOT/build-platform/column-sweep/$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
NATIVE="${NATIVE_FLAGS:--march=native}"
# One core: the public mult would otherwise thread over a pool of every CPU,
# all pinned to the one measured.
export MTL5_NUM_THREADS=1

restore=()
cleanup() { for ((i = ${#restore[@]} - 1; i >= 0; --i)); do eval "${restore[$i]}" || true; done; }
trap cleanup EXIT

clock_policy="none"
gov="/sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor"
if [ -r "$gov" ]; then
    old=$(cat "$gov")
    pol=$(readlink -f "/sys/devices/system/cpu/cpu$CPU/cpufreq")
    if echo performance | sudo -n tee "$pol/scaling_governor" > /dev/null 2>&1; then
        restore+=("echo $old | sudo -n tee $pol/scaling_governor > /dev/null")
        clock_policy="governor performance on $(basename "$pol") (was $old)"
    else
        clock_policy="governor $old (sudo -n refused; not changed)"
    fi
fi
if command -v jetson_clocks > /dev/null 2>&1; then
    store="$OUT/jetson_clocks.store"
    if sudo -n jetson_clocks --store "$store" > /dev/null 2>&1 && sudo -n jetson_clocks > /dev/null 2>&1; then
        restore+=("sudo -n jetson_clocks --restore $store > /dev/null")
        clock_policy="$clock_policy; jetson_clocks"
        if command -v nvpmodel > /dev/null 2>&1; then
            clock_policy="$clock_policy ($(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode:[[:space:]]*//p' | head -1))"
        fi
    fi
fi

{
    echo "host=$(hostname -s)"
    if [ -r /proc/device-tree/model ]; then echo "model=$(tr -d '\0' < /proc/device-tree/model)"; fi
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //')"
    echo "cpu_part=$(awk -v c="$CPU" '/^processor/{p=$3} /^CPU part/{if (p==c) print $4}' /proc/cpuinfo)"
    echo "kernel=$(uname -sr)"
    echo "bench_cpu=$CPU"
    echo "clock_policy=$clock_policy"
    echo "commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || cat "$ROOT/COMMIT" 2>/dev/null || echo unknown)"
    echo "flags=-std=c++20 -O3 -DNDEBUG; native: $NATIVE"
} > "$OUT/sysinfo.txt"

for cxx in g++ clang++; do
    command -v "$cxx" > /dev/null 2>&1 || continue
    echo "compiler_$cxx=$($cxx --version | head -1)" >> "$OUT/sysinfo.txt"
    for variant in generic native; do
        flags=(-std=c++20 -O3 -DNDEBUG)
        # shellcheck disable=SC2206
        [ "$variant" = native ] && flags+=($NATIVE)
        bin="$OUT/bench_column_sweep-$cxx-$variant"
        echo "building $cxx $variant" >&2
        "$cxx" "${flags[@]}" -I"$ROOT/include" "$ROOT/benchmarks/bench_column_sweep.cpp" -o "$bin"
        echo "running $cxx $variant on cpu $CPU" >&2
        taskset -c "$CPU" "$bin" "$@" > "$OUT/$cxx-$variant.csv"
        rm -f "$bin"
        # The CSV's sidecar (benchmarks/check_sidecars.sh): the session's
        # record, and this build's compiler and flags.
        { cat "$OUT/sysinfo.txt"; echo "compiler=$($cxx --version | head -1)"; echo "build_flags=${flags[*]}"; } \
            > "$OUT/$cxx-$variant.csv.sysinfo"
    done
done
echo "results: $OUT" >&2
