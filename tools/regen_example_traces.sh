#!/usr/bin/env bash
# regen_example_traces.sh -- rebuild the pedagogical examples and rewrite the
# JSON traces that the docs site animates (#933).
#
#   tools/regen_example_traces.sh [build-dir]      (default: build)
#
# The build directory must be configured with MTL5_BUILD_EXAMPLES=ON. Each
# example runs with --trace and writes docs-site/src/data/traces/<name>.json.
# Run this after a library change that moves an example's numbers, then run
# `npm test` in docs-site/ and commit the traces with the change.
#
# Builds with -j4 at most: one build at a time on shared machines.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${1:-build}"
case "$build" in /*) ;; *) build="$root/$build" ;; esac
out="$root/docs-site/src/data/traces"

# target name -> trace file name. Add an example here when it gains a trace.
examples=(
    "example_leontief_economy:leontief_economy"
    "example_kirchhoff_network:kirchhoff_network"
    "example_durer_magic_square:durer_magic_square"
    "example_bandwidth_reduction:bandwidth_reduction"
    "example_heat_equation_1d:heat_equation_1d"
)

if [ ! -f "$build/CMakeCache.txt" ]; then
    echo "error: $build is not a configured CMake build directory" >&2
    exit 1
fi

targets=()
for e in "${examples[@]}"; do targets+=("${e%%:*}"); done
cmake --build "$build" --target "${targets[@]}" -j4

mkdir -p "$out"
for e in "${examples[@]}"; do
    target="${e%%:*}"
    name="${e##*:}"
    exe="$(find "$build" -type f \( -name "$target" -o -name "$target.exe" \) | head -n 1)"
    if [ -z "$exe" ]; then
        echo "error: $target was not built" >&2
        exit 1
    fi
    "$exe" --trace "$out/$name.json" > /dev/null
    echo "  $name.json"
done
echo "traces written to ${out#"$root"/}"
