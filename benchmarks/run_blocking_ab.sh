#!/usr/bin/env bash
# Cache-blocking A/B (#426, #432): does GEMM go faster with kc/mc derived from
# the DETECTED cache hierarchy than with the hardcoded Haswell-class defaults?
#
# Both arms are the same source. `bench_blocking_ab_detected` is built with
# MTL5_ENABLE_CACHE_DETECTION; `bench_blocking_ab_default` is what MTL5 ships.
# Nothing else differs, so a difference in throughput is a difference in kc/mc.
#
# Detection is opt-in precisely because this harness found it losing by up to 45%
# on an i7-12700K (see simd/blocking.hpp). Re-run this on new hardware before
# concluding anything for that machine.
#
# Protocol, matching the discipline the rest of docs/benchmarks/ uses:
#   * PINNED    -- one logical id per physical core. On a hybrid CPU this also
#                  decides which cache hierarchy is detected at all (#432), so an
#                  unpinned run here is not merely noisy, it is ambiguous.
#   * INTERLEAVED -- arms alternate within one session, and the ORDER within a
#                  round alternates too, so warm-up and thermal drift do not
#                  accrue to one arm. A ratio is only defensible when both sides
#                  come from the same session.
#   * MIN of N  -- the binary reports the minimum of --reps runs per point.
#   * SHAPES DERIVED ONCE -- the shape list comes from the detected arm and is
#                  passed verbatim to both, so the arms are never compared on
#                  different shapes (their own mc/nc differ, so each would
#                  otherwise pick its own).
#
# Environment:
#   BUILD_DIR    build tree containing the benchmarks (default: build-release)
#   BENCH_PCPUS  comma list of logical ids, one per physical core, in the order
#                to use. MUST match your topology -- see `lscpu -e=CPU,CORE`.
#                  i7-12700K P-cores : 0,2,4,6,8,10,12,14   (E-cores excluded)
#                  Xeon E5-2420 v2   : 0,1,2,3,4,5
#                Default: 0,2,4,6,8,10,12,14
#   THREADS      thread counts to sweep (default "1 8")
#   REPS         repetitions per point (default 5)
#   ROUNDS       interleaved A/B rounds (default 3)
#   DTYPE        double | float (default double)
#   SHAPES       override the derived shape list, "m,n,k;m,n,k". Use for a quick
#                smoke run; prefer the derived list for real measurements, since
#                it is what puts the jc loop in play on THIS machine.
#   OUTDIR       REQUIRED. Where the CSVs land, one directory PER MACHINE, e.g.
#                benchmarks/data/i7-12700k. There is deliberately no default:
#                the CSVs are named by arm, not by machine, and this script
#                deletes them before writing -- so a shared default silently
#                destroys another machine's committed results, which is exactly
#                what happened once.
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build-release}
BENCH_PCPUS=${BENCH_PCPUS:-0,2,4,6,8,10,12,14}
THREADS=${THREADS:-"1 8"}
REPS=${REPS:-5}
ROUNDS=${ROUNDS:-3}
DTYPE=${DTYPE:-double}
if [ -z "${OUTDIR:-}" ]; then
    echo "OUTDIR is required: give this machine its own directory, e.g." >&2
    echo "  OUTDIR=benchmarks/data/<machine> $0" >&2
    echo "Existing machine directories:" >&2
    ls -d benchmarks/data/*/ 2>/dev/null | sed 's|^|  |' >&2
    exit 2
fi

# The arms to run, in the order they lead the first round. All are the same
# source; they differ only in which detected cache levels feed the blocking:
#
#   default    none  -- the compile-time model MTL5 ships
#   detected   L1+L2 -- kc from L1 and mc from L2
#   kconly     L1    -- kc detected, mc from the default model
#   mconly     L2    -- mc detected, kc from the default model
#   ccap       L2    -- mconly PLUS the runtime C-strip bound on mc (#453)
#   ccap2      L2    -- as ccap, but the bound charges the C strip ALONE
#
# kconly/mconly exist because the four-machine result (#430) implicated kc and
# exonerated mc without ever varying them separately: the Ryzen run happened to
# move mc alone and cost nothing, while both machines whose kc moved lost. Run
# all four in ONE session and that becomes a measurement rather than a
# cross-machine inference:  ARMS="default detected kconly mconly"
ARMS=${ARMS:-"detected default"}

ARM_BINS=""
for a in $ARMS; do
    b="$BUILD_DIR/benchmarks/bench_blocking_ab_$a"
    if [ ! -x "$b" ]; then
        echo "missing $b" >&2
        echo "known arms: default detected kconly mconly ccap ccap2" >&2
        echo "build with:  cmake --preset release && cmake --build $BUILD_DIR --target \\" >&2
        for x in $ARMS; do echo "                 bench_blocking_ab_$x \\" >&2; done
        echo "                 -j4" >&2
        exit 1
    fi
    ARM_BINS="$ARM_BINS $b"
done
NARMS=$(echo "$ARMS" | wc -w)
if [ "$NARMS" -lt 2 ]; then
    echo "ARMS needs at least two arms to compare, got '$ARMS'" >&2; exit 2
fi

# The rotation below only cancels position effects when every arm leads an EQUAL
# number of rounds, which needs ROUNDS to be a multiple of the arm count. At 5
# rounds over 4 arms the first arm leads twice and one arm never leads -- a
# residual bias of exactly the size the rotation exists to remove. Round up
# rather than refuse: the operator asked for at least this many rounds, and the
# sidecar records what actually ran.
if [ $((ROUNDS % NARMS)) -ne 0 ]; then
    ROUNDS_ASKED=$ROUNDS
    ROUNDS=$(( ((ROUNDS + NARMS - 1) / NARMS) * NARMS ))
    echo "NOTE: ROUNDS=$ROUNDS_ASKED over $NARMS arms cannot balance the rotation;" >&2
    echo "      running $ROUNDS rounds so each arm leads $((ROUNDS / NARMS))." >&2
fi
mkdir -p "$OUTDIR"

arm_bin() { echo "$BUILD_DIR/benchmarks/bench_blocking_ab_$1"; }
arm_csv() { echo "$OUTDIR/blocking_ab_$1.csv"; }

# Pin to the first T ids of BENCH_PCPUS. On a hybrid part this is also what fixes
# WHICH cache hierarchy gets detected, so it is applied to every run including
# the single-threaded ones.
pin_for() {
    local t=$1
    echo "$BENCH_PCPUS" | cut -d, -f1-"$t"
}

# Every requested thread count must be positive and must fit BENCH_PCPUS. `cut`
# silently returns the whole (shorter) list when asked for more fields than it
# has, so an oversized THREADS would run more workers than pinned cores while the
# CSV still recorded the larger count -- a wrong number that looks like a result.
NPCPUS=$(echo "$BENCH_PCPUS" | tr ',' '\n' | grep -c .)
TMAX=0
for t in $THREADS; do
    if ! [ "$t" -ge 1 ] 2>/dev/null; then
        echo "THREADS: '$t' is not a positive integer" >&2; exit 2
    fi
    if [ "$t" -gt "$NPCPUS" ]; then
        echo "THREADS=$t exceeds the $NPCPUS cpu id(s) in BENCH_PCPUS ($BENCH_PCPUS)." >&2
        echo "Set BENCH_PCPUS to one logical id per physical core on this machine." >&2
        exit 2
    fi
    [ "$t" -gt "$TMAX" ] && TMAX=$t
done

# Preflight BEFORE anything is measured (#442): a dirty tree, a competing build
# or a machine that is already hot invalidates the whole session, and finding
# that out afterwards means the CSVs are already written. Its key=value output is
# appended to both sidecars below, so the machine state travels with the data
# instead of living in the operator's memory of the session.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! PREFLIGHT_KV=$("$SCRIPT_DIR/preflight.sh" --threads "$TMAX" --repo "$SCRIPT_DIR/.." \
                        --ignore-path "$OUTDIR"); then
    echo "preflight failed -- not measuring. Fix the above, or set ALLOW_DIRTY=1 if" >&2
    echo "you accept a dirty tree (it is then recorded as such in the sidecar)." >&2
    exit 1
fi

# Shapes: derived ONCE, from the FIRST arm, at the largest thread count (the
# wide/short shapes depend on the thread budget). Every arm then gets this list,
# so the arms are never compared on different shapes -- each would otherwise pick
# its own from its own mc/nc.
FIRST_ARM=$(echo "$ARMS" | awk '{print $1}')
SHAPES=${SHAPES:-$(taskset -c "$(pin_for "$TMAX")" "$(arm_bin "$FIRST_ARM")" \
                       --suggest-shapes --threads "$TMAX" --dtype "$DTYPE")}
echo "arms:   $ARMS   (shapes derived from '$FIRST_ARM')"
echo "shapes: $SHAPES"

for a in $ARMS; do rm -f "$(arm_csv "$a")" "$(arm_csv "$a").sysinfo"; done

run_arm() {   # arm cpus threads
    MTL5_NUM_THREADS=$3 taskset -c "$2" "$(arm_bin "$1")" \
        --label "$1" --dtype "$DTYPE" --threads "$3" --reps "$REPS" \
        --shapes "$SHAPES" --csv "$(arm_csv "$1")"
}

# Rotate the arm order by round, so every arm leads an equal share of rounds.
# Running one arm first every round would fold warm-up, frequency ramp and
# thermal drift into the ratio in a fixed direction; rotating cancels it to first
# order, and generalises the two-arm alternation to any number of arms.
rotated() {   # round -> the arm list rotated left by (round-1) mod NARMS
    local shift_by=$(( ($1 - 1) % NARMS )) i=0 head="" tail=""
    for a in $ARMS; do
        if [ "$i" -lt "$shift_by" ]; then tail="$tail $a"; else head="$head $a"; fi
        i=$((i + 1))
    done
    echo "$head$tail"
}

for round in $(seq 1 "$ROUNDS"); do
    for t in $THREADS; do
        cpus=$(pin_for "$t")
        order=$(rotated "$round")
        echo "== round $round, T=$t, cpus=$cpus, order:$order"
        for a in $order; do run_arm "$a" "$cpus" "$t"; done
    done
done

# Thermal AFTER the session, alongside the before reading: a run that ended hot
# and a configuration that is simply slow produce the same GFLOP/s, and only the
# pair of temperatures distinguishes them. Appended to the sidecars the binaries
# wrote, which is why this happens once at the end rather than per run -- each
# invocation truncates its own sidecar.
# If the postflight read fails, the MEASUREMENTS are still good -- they are
# already on disk, and discarding a completed session over a failed thermometer
# read would destroy data to punish a missing probe. What must not happen is the
# key going missing silently, so the failure is written into the sidecar as
# `unavailable` and said out loud. `unavailable` is a fact; an absent key is a
# reader's guess.
if ! AFTER_KV=$("$SCRIPT_DIR/preflight.sh" --phase after); then
    echo "WARNING: postflight thermal read failed; recording thermal_after_c=unavailable." >&2
    echo "         The measurements themselves are unaffected." >&2
    AFTER_KV="thermal_after_c=unavailable"
fi
# Guard the same hole from the other side: a preflight that exits 0 while
# printing nothing would leave the key absent just as effectively.
case "$AFTER_KV" in
    *thermal_after_c=*) ;;
    *) AFTER_KV="thermal_after_c=unavailable" ;;
esac

# Did we measure the code the tree is on? The binary records the commit it was
# BUILT from (mtl/build_info.hpp); preflight records where the tree is NOW. They
# differ whenever someone edits, forgets to rebuild, and measures -- an easy
# error, and one that hides completely in the numbers. It happened while writing
# this very script: the first run recorded a binary two commits behind.
TREE_COMMIT=$(printf '%s\n' "$PREFLIGHT_KV" | sed -n 's/^tree_git_commit=//p')
BUILD_COMMIT=$(sed -n 's/^git_commit=//p' "$(arm_csv "$FIRST_ARM").sysinfo" 2>/dev/null | head -1)
STALE=unknown
if [ -n "$TREE_COMMIT" ] && [ -n "$BUILD_COMMIT" ] && \
   [ "$TREE_COMMIT" != unknown ] && [ "$BUILD_COMMIT" != unknown ]; then
    if [ "$TREE_COMMIT" = "$BUILD_COMMIT" ]; then
        STALE=0
    else
        STALE=1
        echo "WARNING: binary was built from $BUILD_COMMIT but the tree is on $TREE_COMMIT." >&2
        echo "         These numbers describe the BUILT code; rebuild if that is not what you meant." >&2
    fi
fi

for a in $ARMS; do
    s="$(arm_csv "$a").sysinfo"
    if [ -f "$s" ]; then
        printf '%s\n%s\n' "$PREFLIGHT_KV" "$AFTER_KV" >> "$s"
        # The INVOCATION, not just the harness name. Which cpus were pinned,
        # which arms ran and which machine profile chose them are what make a
        # committed CSV re-runnable; without them the numbers depend on a command
        # line that lives only in someone's shell history.
        printf 'binary_stale=%s\nharness=run_blocking_ab.sh\nharness_profile=%s\n' \
               "$STALE" "${BENCH_PROFILE:-none}" >> "$s"
        printf 'harness_rounds=%s\nharness_reps=%s\nharness_arms=%s\n' \
               "$ROUNDS" "$REPS" "$(echo "$ARMS" | tr ' ' ',')" >> "$s"
        printf 'harness_pcpus=%s\nharness_threads=%s\nharness_dtype=%s\n' \
               "$BENCH_PCPUS" "$(echo "$THREADS" | tr ' ' ',')" "$DTYPE" >> "$s"
    fi
done

echo
echo "wrote (+ .sysinfo sidecars):"
for a in $ARMS; do echo "  $(arm_csv "$a")"; done
# Every arm against the baseline. `default` is the baseline when it is present --
# it is what MTL5 ships, so every other arm is a proposed change to it.
BASE=default
echo "$ARMS" | grep -qw "$BASE" || BASE=$FIRST_ARM
# Arms that derived the SAME kc/mc measure the same thing, so comparing them is
# a check on the SESSION rather than on the code -- and the analyzer fails it
# when they disagree systematically. Surface those pairs, because they are the
# only way to find out that a machine's own drift is larger than the effect
# under test (it disqualified a 15 W Jetson session that looked fine, #430).
# By HEADER NAME, not column number. The CSV has grown columns twice (`pool`,
# then the mc_used group), so kc sits at field 9 in the pre-`pool` files still
# committed under benchmarks/data and at field 10 in current ones -- a positional
# read silently returns (mc,nc) for the older layout. It happens to group
# correctly today, because a session only ever reads files it just wrote, which
# is exactly the kind of accident that stops being true later.
blocking_of() {
    awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
             NR == 2 { if (col["kc"] && col["mc"]) print $col["kc"] "," $col["mc"]; exit }' \
        "$(arm_csv "$1")" 2>/dev/null
}
CHECKS=""
for a in $ARMS; do
    for b in $ARMS; do
        [ "$a" \< "$b" ] || continue
        [ "$(blocking_of "$a")" = "$(blocking_of "$b")" ] || continue
        CHECKS="$CHECKS  benchmarks/analyze_blocking_ab.py $(arm_csv "$a") $(arm_csv "$b")\n"
    done
done
if [ -n "$CHECKS" ]; then
    echo
    echo "consistency checks -- these arms derived IDENTICAL kc/mc, so they must agree;"
    echo "the analyzer fails them if the machine drifted more than the effect under test:"
    printf "%b" "$CHECKS"
fi

echo "compare with:"
for a in $ARMS; do
    [ "$a" = "$BASE" ] && continue
    echo "  benchmarks/analyze_blocking_ab.py $(arm_csv "$a") $(arm_csv "$BASE")"
done
