#!/usr/bin/env bash
# Wall-clock tracking: measure main on every benchmark machine and publish (#944).
#
#   benchmarks/tracking/track.sh [options]
#
# Run from a clean checkout. In order:
#   1. Fetches origin and resolves --ref (default origin/main) to a commit, then
#      checks that commit out, detached, in a scratch worktree under
#      build-platform/track/. The sessions measure that commit, whatever this
#      checkout has checked out.
#   2. Runs one platform session per machine in hosts.txt, all at once (one
#      session per host): benchmarks/remote_session.sh for a remote host, the
#      machine profile itself for "local". The scripts are the measured commit's
#      own, so the sessions follow its methodology. Results land in
#      build-platform/remote/<machine>/<tag>/.
#   3. For each session that finished with every check passed:
#      - publishes its data with publish_platform.cmake -DDATA_ONLY=ON, into
#        benchmarks/data/<machine>/mtl5_bench/<session>/ of THIS checkout. The
#        card page stays on its paired sessions (#737): one tracking session
#        has no repeatability band;
#      - compares it with the machine's previous published session
#        (track_sessions.py compare), into benchmarks/data/<machine>/tracking/
#        <session>.md and .json.
#   4. Regenerates docs/benchmarks/tracking.md (track_sessions.py page).
#   5. --pr: commits all of it on a new branch bench/track-<tag> off the
#      measured commit, pushes it and opens a pull request. Without --pr the
#      files stay in this checkout for review.
#
# Options:
#   --ref REF        what to measure (default origin/main)
#   --hosts FILE     the machine list (default benchmarks/tracking/hosts.txt)
#   --only M[,M..]   only these machines (names as in hosts.txt)
#   --class C        mtl5_bench class (default medium, the class the cards use)
#   --tag T          session tag (default the UTC time, YYYYMMDDTHHMMSSZ)
#   --pr             commit, push and open a pull request (see 5.)
#   --no-publish     run and fetch the sessions; publish and compare nothing
#
# A machine that cannot be reached, or whose session fails, is reported and
# skipped; the others still publish. Exit 0 when every selected machine
# published; 1 when some did not; 2 on bad usage or an unusable checkout.
#
# Run on demand, by hand, when the lab machines are powered on: their power is
# managed manually, so nothing schedules this (benchmarks/tracking/README.md).
set -uo pipefail
# One { ... } block, parsed before it runs (as in remote_session.sh): --pr
# switches this checkout's branch, which may change this file under it.
{
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 2; }
REF=origin/main HOSTS="$SCRIPT_DIR/hosts.txt" ONLY="" CLASS=medium
TAG=$(date -u +%Y%m%dT%H%M%SZ) PR=0 PUBLISH=1
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF=$2; shift 2 ;;
        --hosts) HOSTS=$2; shift 2 ;;
        --only) ONLY=$2; shift 2 ;;
        --class) CLASS=$2; shift 2 ;;
        --tag) TAG=$2; shift 2 ;;
        --pr) PR=1; shift ;;
        --no-publish) PUBLISH=0; shift ;;
        -h|--help) usage ;;
        *) echo "track.sh: unknown argument '$1'" >&2; usage ;;
    esac
done
[[ "$TAG" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "track.sh: --tag must be YYYYMMDDTHHMMSSZ (it names the published session)" >&2; exit 2; }
[ -r "$HOSTS" ] || { echo "track.sh: cannot read $HOSTS" >&2; exit 2; }
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "track.sh: $ROOT has uncommitted changes; run from a clean, dedicated checkout" >&2; exit 2
fi
log() { echo "[$(date -u +%H:%M:%S) track] $*"; }

# -- 1. the commit -------------------------------------------------------------------
git -C "$ROOT" fetch -q origin || { log "git fetch failed"; exit 2; }
SHA=$(git -C "$ROOT" rev-parse --verify -q "$REF^{commit}") || { log "cannot resolve $REF"; exit 2; }
TRACK="$ROOT/build-platform/track"
SRC="$TRACK/src-$TAG"
DEST="$ROOT/build-platform/remote"
LOGS="$TRACK/logs-$TAG"
mkdir -p "$TRACK" "$LOGS" || exit 2
git -C "$ROOT" worktree add -q --detach "$SRC" "$SHA" || { log "cannot create the worktree $SRC"; exit 2; }
# An interrupted run stops its sessions before their worktree goes: the local
# profile (its exit trap restores the clock policy) and each remote_session.sh,
# which releases its host's lock if the remote session has not started yet;
# one that has started runs on and releases the lock itself.
stop_sessions() {
    local pid
    for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
        kill -0 "$pid" 2>/dev/null || continue
        pkill -TERM -P "$pid" 2>/dev/null
        kill -TERM "$pid" 2>/dev/null
    done
    for pid in "${PIDS[@]+"${PIDS[@]}"}"; do wait "$pid" 2>/dev/null; done
    return 0
}
cleanup() { stop_sessions; git -C "$ROOT" worktree remove --force "$SRC" 2>/dev/null; return 0; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
log "measuring $(git -C "$ROOT" log --oneline -1 "$SHA"), class $CLASS, tag $TAG"

# -- 2. the sessions ------------------------------------------------------------------
# hosts.txt: "<machine> <ssh host or local> <profile>", # comments.
# Two passes: every row is checked before any session starts, so a bad row
# never leaves started sessions behind an exit that removes their worktree.
declare -a MACHINES=() PIDS=()
declare -A PROFILE_OF=() HOST_OF=()
while read -r machine host profile _; do
    case "$machine" in ''|'#'*) continue ;; esac
    if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$machine,"* ]]; then continue; fi
    if [ -z "$profile" ]; then log "$HOSTS: '$machine' has no profile"; exit 2; fi
    case "$profile" in "$machine"-platform.sh|"$machine"-platform.ps1) ;;
        *) log "$HOSTS: $machine's profile $profile is not $machine-platform.sh/.ps1"; exit 2 ;; esac
    if [ -n "${PROFILE_OF[$machine]+x}" ]; then log "$HOSTS: $machine is listed twice"; exit 2; fi
    [ -f "$SRC/benchmarks/machines/$profile" ] || { log "$profile is not a profile at $SHA; skipping $machine"; continue; }
    if [ -e "$DEST/$machine/$TAG" ]; then log "$DEST/$machine/$TAG exists; pick another --tag"; exit 2; fi
    MACHINES+=("$machine")
    PROFILE_OF[$machine]=$profile
    HOST_OF[$machine]=$host
done < "$HOSTS"
[ "${#MACHINES[@]}" -gt 0 ] || { log "no machine selected"; exit 2; }

for machine in "${MACHINES[@]}"; do
    host=${HOST_OF[$machine]} profile=${PROFILE_OF[$machine]}
    if [ "$host" = local ]; then
        mkdir -p "$DEST/$machine"
        ( cd "$SRC" && env CLASS="$CLASS" INTERLEAVE=0 PLATFORM_OUT="$DEST/$machine/$TAG" \
              "benchmarks/machines/$profile" ) > "$LOGS/$machine.log" 2>&1 < /dev/null &
    else
        "$SRC/benchmarks/remote_session.sh" "$host" "$profile" --tag "$TAG" --class "$CLASS" \
            --dest "$DEST" > "$LOGS/$machine.log" 2>&1 < /dev/null &
    fi
    PIDS+=($!)
    log "$machine: started on $host ($profile); log $LOGS/$machine.log"
done

declare -A RC=()
for i in "${!MACHINES[@]}"; do
    wait "${PIDS[$i]}"; RC[${MACHINES[$i]}]=$?
    log "${MACHINES[$i]}: session exit ${RC[${MACHINES[$i]}]}"
done
[ "$PUBLISH" -eq 1 ] || { log "--no-publish: sessions are in $DEST/<machine>/$TAG"; exit 0; }

# -- 3.-5. publish, compare, page ----------------------------------------------------------
# Done with the measured commit's tree; publishing writes into this checkout.
cleanup; trap - EXIT
if [ "$PR" -eq 1 ]; then
    git -C "$ROOT" switch -q -c "bench/track-$TAG" "$SHA" || { log "cannot create the branch bench/track-$TAG"; exit 2; }
fi
PY=$(command -v python3 || command -v python) || { log "no python3"; exit 2; }
published=() missing=()
for machine in "${MACHINES[@]}"; do
    res="$DEST/$machine/$TAG"
    if [ "${RC[$machine]}" -ne 0 ] || [ ! -f "$res/summary.md" ]; then
        log "$machine: not published (session exit ${RC[$machine]}; see $LOGS/$machine.log)"
        missing+=("$machine"); continue
    fi
    data="$ROOT/benchmarks/data/$machine/mtl5_bench"
    # The previous session: the latest one already published, before this one.
    prev=$(ls -1 "$data" 2>/dev/null | grep -E '^[0-9]{8}T[0-9]{6}Z$' | awk -v t="$TAG" '$0 < t' | tail -1)
    if ! cmake -DRESULTS="$res" -DMACHINE="$machine" -DSESSION="$TAG" -DDATA_ONLY=ON \
            -P "$ROOT/benchmarks/publish_platform.cmake" >> "$LOGS/$machine.log" 2>&1; then
        log "$machine: publish_platform refused the session (see $LOGS/$machine.log)"
        missing+=("$machine"); continue
    fi
    if [ -n "$prev" ]; then
        "$PY" "$ROOT/benchmarks/tracking/track_sessions.py" compare "$data/$prev" "$data/$TAG" \
            --markdown "$ROOT/benchmarks/data/$machine/tracking/$TAG.md" \
            --json "$ROOT/benchmarks/data/$machine/tracking/$TAG.json" > /dev/null \
            || log "$machine: published, but the comparison with $prev failed"
        log "$machine: published $TAG, compared with $prev"
    else
        log "$machine: published $TAG, the machine's first session (nothing to compare)"
    fi
    published+=("$machine")
done
( cd "$ROOT" && "$PY" benchmarks/tracking/track_sessions.py page ) || log "the tracking page was not written"

if [ "$PR" -eq 1 ] && [ "${#published[@]}" -gt 0 ]; then
    paths=(docs/benchmarks/tracking.md)
    for m in "${published[@]}"; do
        paths+=("benchmarks/data/$m/mtl5_bench/$TAG")
        [ -d "$ROOT/benchmarks/data/$m/tracking" ] && paths+=("benchmarks/data/$m/tracking")
    done
    msg="bench: wall-clock tracking session $TAG at $(git -C "$ROOT" rev-parse --short=12 "$SHA")"
    body="Tracking session $TAG of \`main\` at $SHA, published by benchmarks/tracking/track.sh (#944).

Published: ${published[*]}.
Not published: ${missing[*]:-none}.

The comparisons are on docs/benchmarks/tracking.md. This is a report: nothing in it gates a merge."
    git -C "$ROOT" add -- "${paths[@]}" \
        && git -C "$ROOT" commit -q -m "$msg" -m "$body" \
        && git -C "$ROOT" push -q -u origin "bench/track-$TAG" \
        && gh pr create --repo stillwater-sc/mtl5 --head "bench/track-$TAG" --base main --title "$msg" --body "$body" \
        || { log "the pull request was not created"; exit 1; }
fi
log "published: ${published[*]:-none}; not published: ${missing[*]:-none}"
[ "${#missing[@]}" -eq 0 ]
exit $?
}
