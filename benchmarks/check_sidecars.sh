#!/usr/bin/env bash
# Every committed CSV must have its .sysinfo beside it (#477), and every sidecar
# every key in benchmarks/sidecar_keys.txt (#930; the second half of this script).
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
# JSON too: mtl5_bench (#721) writes its results as JSON with the same sidecar.
mapfile -t csvs < <(find "$DATA" -type f \( -name '*.csv' -o -name '*.json' \) | sort)
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

# -- the keys (#930) ---------------------------------------------------------------
# A sidecar must also SAY enough: every key in benchmarks/sidecar_keys.txt, the
# list mtl5-python's ingest requires (REQUIRED_KEYS), non-empty unless marked
# empty-ok, no key with two different values (which one describes the run?
# mtl5-python drops such a key), and no line that is neither key=value, a
# #-comment nor blank (the ingest counts each as a fault). A derived report -- a sidecar with derived_by=,
# such as a tracking comparison (benchmarks/tracking/) -- describes no run of
# its own; its provenance is the sessions it names, so it needs those keys.
#
# The same ratchet as the exemptions: sidecars that were already incomplete are
# listed in benchmarks/data/.sidecar-incomplete and cost nothing here. A NEW
# incomplete sidecar fails, and so does a listed one that is complete now, or
# gone: the list only ever shrinks.
KEYS="benchmarks/sidecar_keys.txt"
INCOMPLETE="$DATA/.sidecar-incomplete"
[ -r "$KEYS" ] || { echo "check_sidecars: cannot read $KEYS" >&2; exit 2; }
required=$(awk '!/^[[:space:]]*(#|$)/ {printf "%s ", $1}' "$KEYS")
empty_ok=$(awk '!/^[[:space:]]*(#|$)/ && $2 == "empty-ok" {printf "%s ", $1}' "$KEYS")
declare -A grandfathered=()
if [ -e "$INCOMPLETE" ]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(printf '%s' "$line" | tr -d '[:space:]')"
        [ -n "$line" ] && grandfathered["$line"]=1
    done < "$INCOMPLETE"
fi
# The problems with one sidecar, space-separated: missing or empty keys, and
# conflict:<key> for a key with two values.
sidecar_problems() {
    awk -v req="$required" -v ok="$empty_ok" '
        BEGIN { n = split(req, R, " "); m = split(ok, O, " "); for (i = 1; i <= m; ++i) E[O[i]] = 1 }
        /^[[:space:]]*(#|$)/ { next }
        index($0, "=") == 0 { bad = bad "line" NR " "; next }
        {
            k = substr($0, 1, index($0, "=") - 1); v = substr($0, index($0, "=") + 1)
            gsub(/^[ \t\r]+|[ \t\r]+$/, "", k); gsub(/^[ \t\r]+|[ \t\r]+$/, "", v)
            if (k in V) { if (V[k] != v) C[k] = 1 } else V[k] = v
        }
        END {
            if ("derived_by" in V) { n = split("derived_by baseline_session new_session", R, " "); delete E }
            for (i = 1; i <= n; ++i) if (!(R[i] in V) || (V[R[i]] == "" && !(R[i] in E))) printf "%s ", R[i]
            for (k in C) printf "conflict:%s ", k
            if (bad != "") printf "not-key=value:%s", bad
        }' "$1"
}
incomplete=()    # a sidecar with a problem, not grandfathered
fixed=()         # grandfathered, but complete now, or gone
n_checked=0
for csv in "${csvs[@]}"; do
    [ -f "$csv.sysinfo" ] || continue
    n_checked=$((n_checked + 1))
    probs=$(sidecar_problems "$csv.sysinfo")
    if [ -n "$probs" ]; then
        [ -n "${grandfathered[$csv.sysinfo]:-}" ] || incomplete+=("$csv.sysinfo: $probs")
    elif [ -n "${grandfathered[$csv.sysinfo]:-}" ]; then
        fixed+=("$csv.sysinfo  (complete now -- drop it from the list)")
    fi
done
for path in "${!grandfathered[@]}"; do
    [ -f "$path" ] || fixed+=("$path  (no such sidecar -- drop it from the list)")
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

if [ "${#incomplete[@]}" -gt 0 ]; then
    rc=1
    echo
    echo "Sidecars that lack required keys (benchmarks/sidecar_keys.txt) or give a key two values:"
    printf '  %s\n' "${incomplete[@]}"
    cat <<'MSG'

mtl5-python keeps a run whose sidecar lacks one of these out of every trend and
comparison (#930). Produce it through its harness script, which completes the
sidecar (benchmarks/sidecar.sh), or, for a binary with no script, through
benchmarks/run_with_sidecar.sh. A key that does not apply still gets a value:
pool_size=1, bench_cpus=unpinned, clock_policy=not applied.
MSG
fi

if [ "${#fixed[@]}" -gt 0 ]; then
    rc=1
    echo
    echo "Stale entries in $INCOMPLETE:"
    printf '  %s\n' "${fixed[@]}"
    echo
    echo "The list only ever shrinks. Delete these lines."
fi

if [ "$rc" -eq 0 ]; then
    echo "OK: ${#csvs[@]} committed CSVs, ${#exempt[@]} grandfathered, none missing a sidecar."
    echo "OK: $n_checked sidecars, ${#grandfathered[@]} grandfathered as incomplete, every other one has every required key."
fi
exit "$rc"
