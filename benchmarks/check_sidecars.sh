#!/usr/bin/env bash
# Every committed CSV must have its .sysinfo beside it (#477).
#
# A benchmark result without provenance is a number nobody can check: not the
# commit it was built from, not the compiler flags that decided which kernel ran,
# not the machine state it was measured under. #442 established that contract and
# closed on the three harnesses it named; the sidecar is written by the BINARY,
# so a producer that was never taught to write one keeps producing CSVs that
# cannot be traced. Twenty-eight such files reached the repository before anyone
# noticed, and they were found by auditing an open issue rather than by anything
# failing.
#
# This is the thing that fails instead.
#
# A RATCHET, NOT A GATE ON HISTORY. The pre-contract files are listed in
# .sidecar-exempt and cost nothing here; what the check forbids is ADDING another
# one. The list only ever shrinks, and the checker also fails when an exempt file
# HAS gained a sidecar -- otherwise the list would quietly rot into a permanent
# amnesty for files that were fixed years earlier.
#
# Usage:  benchmarks/check_sidecars.sh          # from anywhere in the tree
# Exit:   0 all good, 1 a violation, 2 the scan itself could not run
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

DATA="benchmarks/data"
EXEMPT="$DATA/.sidecar-exempt"

[ -d "$DATA" ] || { echo "check_sidecars: no $DATA directory" >&2; exit 2; }

# Read the exemption list, stripping comments and blanks. Missing file is fine
# (it means nothing is exempt), unreadable is not.
declare -A exempt=()
if [ -e "$EXEMPT" ]; then
    [ -r "$EXEMPT" ] || { echo "check_sidecars: $EXEMPT is not readable" >&2; exit 2; }
    while IFS= read -r line; do
        line="${line%%#*}"                       # strip trailing comment
        line="$(printf '%s' "$line" | tr -d '[:space:]')"
        [ -n "$line" ] && exempt["$line"]=1
    done < "$EXEMPT"
fi

missing=()   # CSV with no sidecar and no exemption -- the violation this exists for
stale=()     # exempt entry that now HAS a sidecar, or names a file that is gone

# `find | sort` rather than a glob: the data directory is nested per machine, and
# an unmatched glob would silently scan nothing.
mapfile -t csvs < <(find "$DATA" -type f -name '*.csv' | sort)
if [ "${#csvs[@]}" -eq 0 ]; then
    echo "check_sidecars: found no CSVs under $DATA -- the scan is broken, not the tree" >&2
    exit 2
fi

for csv in "${csvs[@]}"; do
    if [ -f "$csv.sysinfo" ]; then
        [ -n "${exempt[$csv]:-}" ] && stale+=("$csv  (has a sidecar now -- drop the exemption)")
    else
        [ -n "${exempt[$csv]:-}" ] || missing+=("$csv")
    fi
done

# An exemption naming a file that no longer exists is also stale.
for path in "${!exempt[@]}"; do
    [ -f "$path" ] || stale+=("$path  (no such CSV -- drop the exemption)")
done

rc=0
if [ "${#missing[@]}" -gt 0 ]; then
    rc=1
    echo "Committed CSVs with no .sysinfo sidecar:"
    printf '  %s\n' "${missing[@]}"
    cat <<'MSG'

A result without provenance cannot be traced to a commit, a build configuration
or a machine state, which is what makes it checkable later. Either:

  * produce the CSV through a harness that writes the sidecar (see
    benchmarks/README.md), and commit the two together; or
  * if the file predates the contract and its provenance genuinely was never
    captured, add it to benchmarks/data/.sidecar-exempt with a reason.

Adding an exemption is a deliberate act and should be visible in review.
MSG
fi

if [ "${#stale[@]}" -gt 0 ]; then
    rc=1
    echo
    echo "Stale entries in $EXEMPT:"
    printf '  %s\n' "${stale[@]}"
    echo
    echo "The exemption list only ever shrinks. Delete these lines."
fi

if [ "$rc" -eq 0 ]; then
    echo "OK: ${#csvs[@]} committed CSVs, ${#exempt[@]} grandfathered, none missing a sidecar."
fi
exit "$rc"
