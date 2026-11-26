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
    "example_convection_diffusion:convection_diffusion"
    "example_laplacian_2d:laplacian_2d"
    "example_smoother_convergence:smoother_convergence"
    "example_least_squares_qr:least_squares_qr"
    "example_solve_three_ways:solve_three_ways"
    "example_vibrating_string:vibrating_string"
    "example_pca_svd:pca_svd"
    "example_sparse_formats:sparse_formats"
    "example_structured_views:structured_views"
)

if [ ! -f "$build/CMakeCache.txt" ]; then
    echo "error: $build is not a configured CMake build directory" >&2
    exit 1
fi

# One configuration, used for both the build and the lookup: a multi-config
# tree (Visual Studio, Ninja Multi-Config) holds one binary per configuration,
# and the traces must come from the one just built. CONFIG overrides; the
# default is the tree's CMAKE_BUILD_TYPE, else Release.
config="${CONFIG:-$(sed -n 's/^CMAKE_BUILD_TYPE:[A-Z]*=//p' "$build/CMakeCache.txt")}"
config="${config:-Release}"

targets=()
for e in "${examples[@]}"; do targets+=("${e%%:*}"); done
cmake --build "$build" --config "$config" --target "${targets[@]}" -j4

mkdir -p "$out"
for e in "${examples[@]}"; do
    target="${e%%:*}"
    name="${e##*:}"
    # Multi-config trees put it in examples/<config>/, single-config in examples/.
    exe=""
    for dir in "$build/examples/$config" "$build/examples"; do
        for cand in "$dir/$target" "$dir/$target.exe"; do
            if [ -z "$exe" ] && [ -f "$cand" ] && [ -x "$cand" ]; then exe="$cand"; fi
        done
    done
    if [ -z "$exe" ]; then
        echo "error: $target ($config) not found in $build/examples" >&2
        exit 1
    fi
    "$exe" --trace "$out/$name.json" > /dev/null
    echo "  $name.json"
done
echo "traces written to ${out#"$root"/}"
