#!/usr/bin/env bash
# Run one platform-benchmark session on a lab machine over ssh, and fetch it (#737).
#
#   benchmarks/remote_session.sh HOST PROFILE [options] [-- profile/platform.sh args]
#
# HOST is an ssh destination; PROFILE is a machine profile's file name under
# benchmarks/machines/ (e.g. xeon-e5-2420v2-platform.sh). In order:
#   1. Pushes this checkout's HEAD to ~/mtl5-campaign on HOST (created on first
#      use; an existing clone elsewhere is never touched) and checks it out
#      detached, so the session measures exactly this commit. The local tree
#      must be clean: a session of uncommitted code is not reproducible.
#   2. --reboot: reboots HOST first (sudo -n reboot), waits for ssh, then lets
#      it settle (--settle, default 300 s) -- #737's "after a cold boot".
#   3. Runs the profile detached on HOST (setsid nohup), so a dropped ssh
#      connection does not end the session; polls until it finishes.
#   4. Copies the results directory back to DEST/<HOST>/<session> (default
#      DEST build-platform/remote) and prints the profile's exit status.
#
# Options:
#   --tag NAME      session name (default: UTC time); also the remote --out
#   --reboot        cold-boot HOST before the session
#   --settle SEC    seconds to wait after the reboot (default 300)
#   --interleave    also run the #780 interleave sweep (INTERLEAVE=1)
#   --class C       CLASS for the profile (default medium)
#   --dest DIR      local destination root (default build-platform/remote)
#   --poll SEC      seconds between completion checks (default 60)
#
# Exit: the profile's exit status (0 all checks passed), 3 if HOST could not
# be reached or the session could not be started, 2 on bad usage.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 2; }
[ $# -ge 2 ] || usage
HOST=$1 PROFILE=$2
shift 2
TAG=$(date -u +%Y%m%dT%H%M%SZ) REBOOT=0 SETTLE=300 INTERLEAVE=0 CLASS=medium
DEST="$ROOT/build-platform/remote" POLL=60 EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG=$2; shift 2 ;;
        --reboot) REBOOT=1; shift ;;
        --settle) SETTLE=$2; shift 2 ;;
        --interleave) INTERLEAVE=1; shift ;;
        --class) CLASS=$2; shift 2 ;;
        --dest) DEST=$2; shift 2 ;;
        --poll) POLL=$2; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "remote_session.sh: unknown argument '$1'" >&2; usage ;;
    esac
done
[ -x "$ROOT/benchmarks/machines/$PROFILE" ] || { echo "remote_session.sh: no profile benchmarks/machines/$PROFILE" >&2; exit 2; }
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "remote_session.sh: the local tree has uncommitted changes; commit them first" >&2; exit 2
fi
SHA=$(git -C "$ROOT" rev-parse HEAD)
RDIR=mtl5-campaign
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30)
log() { echo "[$(date -u +%H:%M:%S) $HOST] $*"; }

# -- 1. the commit -------------------------------------------------------------
"${SSH[@]}" "$HOST" "git init -q $RDIR 2>/dev/null; cd $RDIR && git config receive.denyCurrentBranch ignore" \
    || { log "cannot reach $HOST"; exit 3; }
git -C "$ROOT" push -q -f "$HOST:$RDIR" "$SHA:refs/heads/campaign" || { log "push failed"; exit 3; }
"${SSH[@]}" "$HOST" "cd $RDIR && git checkout -q --detach -f $SHA && git clean -q -fdx -e build-platform && git log --oneline -1" \
    || { log "checkout failed"; exit 3; }
log "at $(git -C "$ROOT" log --oneline -1 "$SHA")"

# -- 2. cold boot ----------------------------------------------------------------
if [ "$REBOOT" -eq 1 ]; then
    log "rebooting"
    "${SSH[@]}" "$HOST" "sudo -n reboot" > /dev/null 2>&1
    sleep 20
    for _ in $(seq 1 60); do "${SSH[@]}" "$HOST" true 2>/dev/null && break; sleep 15; done
    "${SSH[@]}" "$HOST" true 2>/dev/null || { log "did not come back"; exit 3; }
    log "up after reboot ($("${SSH[@]}" "$HOST" uptime -p)); settling ${SETTLE}s"
    sleep "$SETTLE"
fi

# -- 3. the session ----------------------------------------------------------------
OUTREL="build-platform/results/campaign-$TAG"
q_extra=$(printf ' %q' ${EXTRA[@]+"${EXTRA[@]}"})
"${SSH[@]}" "$HOST" "cd $RDIR && mkdir -p $OUTREL && setsid nohup env CLASS=$CLASS INTERLEAVE=$INTERLEAVE \
    PLATFORM_OUT=\$PWD/$OUTREL benchmarks/machines/$PROFILE$q_extra > $OUTREL/session.log 2>&1 < /dev/null; \
    echo \$? > $OUTREL/session.exit" > /dev/null 2>&1 < /dev/null &
sleep 5
"${SSH[@]}" "$HOST" "test -f $RDIR/$OUTREL/session.log" || { log "session did not start"; exit 3; }
log "session $TAG started ($PROFILE, class $CLASS$([ "$INTERLEAVE" -eq 1 ] && echo ", interleave"))"
while :; do
    sleep "$POLL"
    st=$("${SSH[@]}" "$HOST" "cat $RDIR/$OUTREL/session.exit 2>/dev/null") && [ -n "$st" ] && break
done
log "session finished, exit $st"

# -- 4. fetch -------------------------------------------------------------------------
mkdir -p "$DEST/$HOST"
rsync -a -e "ssh -o BatchMode=yes" "$HOST:$RDIR/$OUTREL/" "$DEST/$HOST/$TAG/" || { log "fetch failed"; exit 3; }
log "results in $DEST/$HOST/$TAG"
tail -3 "$DEST/$HOST/$TAG/session.log" | sed "s/^/[$HOST]   /"
exit "$st"
