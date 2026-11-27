# Sourced by the benchmark harness scripts: the run's half of a sidecar (#930).
#
#   . "$SCRIPT_DIR/sidecar.sh"
#   sidecar_finish FILE benchmark=bench_all class=blas-sweep bench_cpus=4 ...
#
# The binary writes the build and the machine's identity into FILE
# (benchmarks/harness/sidecar.hpp); the harness has already appended its
# preflight record. sidecar_finish adds what only the run knows, then checks
# FILE against benchmarks/sidecar_keys.txt:
#   - each key=value argument, unless FILE already has that key;
#   - machine: $BENCH_MACHINE (a machine profile sets it from its file name),
#     else the cpu_brand the binary recorded;
#   - clock_policy: $CLOCK_POLICY, else "not applied";
#   - bench_cpus: "unpinned", unless an argument or FILE gives it.
# A key that doesn't apply still gets a value, never an omission (#930).
# Returns 1, naming the keys, when FILE still lacks a required key.

_SIDECAR_KEYS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sidecar_keys.txt"

# Every function here is safe under the callers' set -euo pipefail.

# sidecar_has FILE KEY: FILE has a line KEY=...
sidecar_has() { grep -q "^$2=" "$1" 2>/dev/null; }

# sidecar_set FILE key=value...: append each key FILE does not have yet.
sidecar_set() {
    local f=$1 kv; shift
    for kv in "$@"; do
        sidecar_has "$f" "${kv%%=*}" || printf '%s\n' "$kv" >> "$f"
    done
}

# sidecar_put FILE key=value...: set each key, replacing a value already there
# (one line per key: a key repeated with two values is ambiguous, and
# mtl5-python drops it). For what the harness knows better than the binary,
# such as the thread counts of a CSV that several runs wrote.
sidecar_put() {
    local f=$1 kv k tmp; shift
    for kv in "$@"; do
        k=${kv%%=*}
        tmp=$(mktemp) || return 1
        grep -v "^$k=" "$f" > "$tmp" 2>/dev/null
        printf '%s\n' "$kv" >> "$tmp"
        cat "$tmp" > "$f" && rm -f "$tmp"
    done
}

# sidecar_missing FILE: the required keys FILE lacks, or has empty (except
# those marked empty-ok), one per line.
sidecar_missing() {
    local f=$1 key flag v
    while read -r key flag _; do
        case "$key" in ''|'#'*) continue ;; esac
        if ! sidecar_has "$f" "$key"; then echo "$key"; continue; fi
        v=$(sed -n "s/^$key=//p" "$f" | head -1)
        if [ -z "$v" ] && [ "$flag" != empty-ok ]; then echo "$key"; fi
    done < "$_SIDECAR_KEYS"
    return 0   # the list is the answer; under set -e a false last test must not end the caller
}

# sidecar_finish FILE key=value...
sidecar_finish() {
    local f=$1 brand missing; shift
    [ -f "$f" ] || { echo "sidecar: $f does not exist (did the binary write it?)" >&2; return 1; }
    sidecar_set "$f" "$@"
    brand=$(sed -n 's/^cpu_brand=//p' "$f" | head -1)
    sidecar_set "$f" "machine=${BENCH_MACHINE:-${brand:-unknown}}" \
                     "clock_policy=${CLOCK_POLICY:-not applied}" "bench_cpus=unpinned"
    missing=$(sidecar_missing "$f" | tr '\n' ' ')
    if [ -n "$missing" ]; then
        echo "sidecar: $f lacks required keys: $missing(benchmarks/sidecar_keys.txt)" >&2
        return 1
    fi
}
