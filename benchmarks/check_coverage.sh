#!/usr/bin/env bash
# Every public operation header is benchmarked by mtl5_bench, or says why not
# (#721 M7).
#
# The platform benchmark's promise is that it measures ALL of MTL5's linear
# algebra, so a new machine's capability card is complete. A registry only
# stays complete if adding an operation without registering it fails
# something. This is that something: benchmarks/platform/coverage.txt maps
# every header in the scanned directories to the entry that measures it, or to
# a written exemption, and this script checks the map against the tree and
# against the binary.
#
# It fails when:
#   - a scanned header has no line in the manifest (the case this exists for);
#   - a manifest line names a header that is gone, or one outside every scan,
#     so the manifest cannot rot into a list of things that no longer exist;
#   - a header appears twice, or a line has an unknown kind;
#   - a bench / via / class target matches no line of `mtl5_bench --list`, so
#     renaming or deleting an entry cannot silently orphan a header;
#   - an exemption gives no reason.
#
# Usage:  benchmarks/check_coverage.sh path/to/mtl5_bench
#         benchmarks/check_coverage.sh --list-file entries.txt   # a saved --list
# Exit:   0 all good, 1 a violation, 2 the check itself could not run
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/benchmarks/platform/coverage.txt"
INCLUDE="$ROOT/include/mtl"

usage() { echo "usage: $0 path/to/mtl5_bench | --list-file FILE" >&2; exit 2; }
[ $# -ge 1 ] || usage

# The registry, one area/op/dtype per line. Taken from the binary so the check
# is against what actually runs, not against a second copy of the names.
if [ "$1" = "--list-file" ]; then
    [ $# -eq 2 ] && [ -r "$2" ] || usage
    entries="$(tr -d '\r' < "$2")"
else
    [ -x "$1" ] || { echo "check_coverage: $1 is not an executable" >&2; exit 2; }
    entries="$("$1" --list | tr -d '\r')" || { echo "check_coverage: $1 --list failed" >&2; exit 2; }
fi
[ -n "$entries" ] || { echo "check_coverage: the registry listing is empty" >&2; exit 2; }
[ -r "$MANIFEST" ] || { echo "check_coverage: cannot read $MANIFEST" >&2; exit 2; }
[ -d "$INCLUDE" ]  || { echo "check_coverage: no $INCLUDE" >&2; exit 2; }

errors=0
fail() { echo "check_coverage: $*" >&2; errors=$((errors + 1)); }

# A target matches a listing line (area/op/dtype) only on a NAME BOUNDARY:
#   - the whole line, or the target followed by `/` or `+` in the line
#     (`krylov/gmres30` matches krylov/gmres30+ilu0/f64);
#   - a target that itself ends in `/` or `+` matches as a prefix
#     (`blas3/gemm/` matches blas3/gemm/f64 and blas3/gemm/f32);
#   - a trailing `*` asks for a family prefix explicitly
#     (`krylov/idr_s*` matches idr_s1..idr_s8).
# A bare prefix used to match anywhere, so `krylov/gmres30` still passed after
# its entry was renamed to, say, krylov/gmres30x (#825).
has_entry() {
    local target="$1" e
    while IFS= read -r e; do
        case "$target" in
            *\*)    case "$e" in "${target%\*}"*) return 0 ;; esac ;;
            */|*+)  case "$e" in "$target"*) return 0 ;; esac ;;
            *)      case "$e" in "$target"|"$target"/*|"$target"+*) return 0 ;; esac ;;
        esac
    done <<< "$entries"
    return 1
}

scans=""        # newline-separated scan paths
mapped=""       # newline-separated manifest headers, for the duplicate and
                # reverse checks (plain strings: bash 3.2 has no assoc arrays)
lineno=0
while IFS= read -r raw || [ -n "$raw" ]; do
    lineno=$((lineno + 1))
    line="${raw%%#*}"
    line="$(printf '%s' "$line" | tr -d '\r')"
    # Split on whitespace: header, kind, and the rest (target or reason).
    read -r header kind rest <<< "$line"
    [ -n "${header:-}" ] || continue
    if [ "$header" = "scan" ]; then
        [ -n "${kind:-}" ] || { fail "line $lineno: scan with no path"; continue; }
        [ -e "$INCLUDE/$kind" ] || fail "line $lineno: scan path include/mtl/$kind does not exist"
        scans="$scans$kind"$'\n'
        continue
    fi
    if printf '%s' "$mapped" | grep -Fxq -- "$header"; then
        fail "line $lineno: $header is listed twice"
    fi
    mapped="$mapped$header"$'\n'
    [ -f "$INCLUDE/$header" ] || fail "line $lineno: $header does not exist under include/mtl"
    case "${kind:-}" in
        bench|via|class)
            target="${rest%% *}"
            if [ -z "$target" ]; then
                fail "line $lineno: $header ($kind) names no entry"
            elif ! has_entry "$target"; then
                fail "line $lineno: $header ($kind) -> '$target' matches no mtl5_bench entry"
            fi ;;
        exempt)
            [ -n "${rest:-}" ] || fail "line $lineno: $header is exempt with no reason" ;;
        *)
            fail "line $lineno: $header has unknown kind '${kind:-}' (bench, via, class or exempt)" ;;
    esac
done < "$MANIFEST"

[ -n "$scans" ] || { echo "check_coverage: the manifest has no scan lines" >&2; exit 2; }

# Every header in every scan must be mapped. `find | sort`, not a glob, so a
# scan that matches nothing is visible rather than silently empty.
scanned=""
while IFS= read -r s; do
    [ -n "$s" ] || continue
    if [ -d "$INCLUDE/$s" ]; then
        found="$(cd "$INCLUDE" && find "$s" -type f -name '*.hpp' | LC_ALL=C sort)"
        [ -n "$found" ] || fail "scan $s contains no headers"
        scanned="$scanned$found"$'\n'
    elif [ -f "$INCLUDE/$s" ]; then
        scanned="$scanned$s"$'\n'
    fi
done <<< "$scans"

unmapped=0
while IFS= read -r h; do
    [ -n "$h" ] || continue
    if ! printf '%s' "$mapped" | grep -Fxq -- "$h"; then
        fail "include/mtl/$h has no line in benchmarks/platform/coverage.txt"
        unmapped=$((unmapped + 1))
    fi
done <<< "$scanned"

# And the reverse: a mapped header outside every scan is dead weight that
# looks like coverage.
while IFS= read -r h; do
    [ -n "$h" ] || continue
    printf '%s' "$scanned" | grep -Fxq -- "$h" || fail "$h is in the manifest but under no scan line"
done <<< "$mapped"

if [ "$errors" -ne 0 ]; then
    if [ "$unmapped" -ne 0 ]; then
        echo "" >&2
        echo "A new public header needs a line in benchmarks/platform/coverage.txt:" >&2
        echo "  bench <entry> if mtl5_bench measures it (add the entry: benchmarks/platform/entries_*.hpp)," >&2
        echo "  via / class <entry> if an existing entry already measures its kernel," >&2
        echo "  exempt <reason> if it is a query, a predicate, I/O or plumbing." >&2
    fi
    echo "check_coverage: $errors problem(s)" >&2
    exit 1
fi

n_headers=$(printf '%s' "$scanned" | grep -c .)
n_exempt=$(grep -Ec '^[^#[:space:]]+[[:space:]]+exempt([[:space:]]|$)' "$MANIFEST")
n_entries=$(printf '%s\n' "$entries" | grep -c .)
echo "check_coverage: OK -- $n_headers headers mapped ($n_exempt exempt) against $n_entries entries"
