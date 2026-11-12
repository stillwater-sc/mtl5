#!/usr/bin/env bash
# Run one platform-benchmark session on a lab machine over ssh, and fetch it (#737).
#
#   benchmarks/remote_session.sh HOST PROFILE [options] [-- profile/platform.sh args]
#
# HOST is an ssh destination; PROFILE is a machine profile's file name under
# benchmarks/machines/ (e.g. xeon-e5-2420v2-platform.sh, or a Windows host's
# ryzen-7-8845hs-platform.ps1). In order:
#   1. Takes the host's lock (~/mtl5-campaign.lock: one session per host),
#      pushes this checkout's HEAD to ~/mtl5-campaign on HOST (created on first
#      use; an existing clone elsewhere is never touched) and checks it out
#      detached, so the session measures exactly this commit. The local tree
#      must be clean: a session of uncommitted code is not reproducible.
#   2. --reboot: reboots HOST first (sudo -n reboot), waits until its boot id
#      has changed, then lets it settle (--settle, default 300 s) -- #737's
#      "after a cold boot".
#   3. Runs the profile detached on HOST (setsid nohup), so a dropped ssh
#      connection does not end the session; polls until it finishes.
#   4. Copies the results directory back to DEST/<machine>/<session> (default
#      DEST build-platform/remote) and prints the profile's exit status.
#      <machine> is the profile's name without -platform.sh / -platform.ps1:
#      the machine's name in docs/benchmarks/systems.md and under
#      benchmarks/data/, whatever HOST is called on the network (sw-13 runs
#      xeon-e5-2420v2-platform.sh, so its sessions land in xeon-e5-2420v2/).
#
# A .ps1 PROFILE means a Windows HOST, which needs OpenSSH Server with
# PowerShell as its default shell (HKLM:\SOFTWARE\OpenSSH DefaultShell: under
# cmd.exe, git over ssh receives the repository path with its quotes), Git for
# Windows on sshd's PATH, and an administrator account (the clock policy and
# the reboot). The same steps then run as PowerShell: the session is started
# with Win32_Process.Create, outside the ssh session's job object (which
# Windows ends with the connection, so nohup has no equivalent), the reboot is
# Restart-Computer checked by LastBootUpTime, and the fetch is scp.
#
# Options:
#   --tag NAME      session name (default: UTC time); also the remote --out
#   --reboot        cold-boot HOST before the session
#   --settle SEC    seconds to wait after the reboot (default 300)
#   --interleave    also run the #780 interleave sweep (INTERLEAVE=1)
#   --class C       CLASS for the profile (default medium)
#   --dest DIR      local destination root (default build-platform/remote)
#   --poll SEC      seconds between completion checks (default 60)
#   --timeout SEC   give up waiting for the session after SEC (default 43200,
#                   12 h; a large session on the Xeon takes about 3 h)
#
# Exit: the profile's exit status (0 all checks passed), 3 if HOST could not
# be reached or the session could not be started, 4 if the session did not
# finish within --timeout (its results stay on HOST), 2 on bad usage.
#
# Until the detached session has started, this script owns HOST's lock and
# releases it however it ends (an error, Ctrl-C, a kill); from then on the
# session releases it.
set -uo pipefail
# The whole script is one { ... } block, so bash parses all of it before running
# any: a session runs for hours, and bash otherwise reads a script from the file
# as it goes, so switching branches or editing this file under a running
# session made its poller execute the new file from the old offset.
{
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 2; }
[ $# -ge 2 ] || usage
HOST=$1 PROFILE=$2
shift 2
TAG=$(date -u +%Y%m%dT%H%M%SZ) REBOOT=0 SETTLE=300 INTERLEAVE=0 CLASS=medium
DEST="$ROOT/build-platform/remote" POLL=60 TIMEOUT=43200 EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG=$2; shift 2 ;;
        --reboot) REBOOT=1; shift ;;
        --settle) SETTLE=$2; shift 2 ;;
        --interleave) INTERLEAVE=1; shift ;;
        --class) CLASS=$2; shift 2 ;;
        --dest) DEST=$2; shift 2 ;;
        --poll) POLL=$2; shift 2 ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "remote_session.sh: unknown argument '$1'" >&2; usage ;;
    esac
done
case "$PROFILE" in *.ps1) WIN=1 ;; *) WIN=0 ;; esac
# The local results directory is named after the machine, not the ssh host.
MACHINE=${PROFILE%-platform.*}
if [ "$MACHINE" = "$PROFILE" ] || ! [[ "$MACHINE" =~ ^[a-z0-9][a-z0-9.-]*$ ]]; then
    echo "remote_session.sh: '$PROFILE' is not a <machine>-platform.sh / .ps1 profile" >&2; exit 2
fi
if [ "$WIN" -eq 1 ]; then
    [ -f "$ROOT/benchmarks/machines/$PROFILE" ] || { echo "remote_session.sh: no profile benchmarks/machines/$PROFILE" >&2; exit 2; }
else
    [ -x "$ROOT/benchmarks/machines/$PROFILE" ] || { echo "remote_session.sh: no profile benchmarks/machines/$PROFILE" >&2; exit 2; }
fi
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "remote_session.sh: the local tree has uncommitted changes; commit them first" >&2; exit 2
fi
SHA=$(git -C "$ROOT" rev-parse HEAD)
RDIR=mtl5-campaign
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30)
log() { echo "[$(date -u +%H:%M:%S) $HOST] $*"; }
# Windows OpenSSH prints a post-quantum key-exchange warning on stderr; noise here.
rsh() { "${SSH[@]}" "$HOST" "$@" 2> >(grep -v '^\*\*' >&2); }

# The profile's kind must match the host's: PowerShell prints Windows_NT for
# $env:OS, a POSIX shell prints ":OS".
os=$("${SSH[@]}" "$HOST" 'echo $env:OS' 2>/dev/null | tr -d '\r') || { log "cannot reach $HOST"; exit 3; }
if [ "$WIN" -eq 1 ] && [ "$os" != Windows_NT ]; then
    log "$PROFILE is a Windows profile, but $HOST's ssh shell is not PowerShell (see the header)"; exit 2
elif [ "$WIN" -eq 0 ] && [ "$os" = Windows_NT ]; then
    log "$HOST is a Windows host; give it a .ps1 profile"; exit 2
fi
[ "$WIN" -eq 1 ] && exec_windows=1 || exec_windows=0

# The remote lock and the local temporary file are cleaned up however this
# script ends. OWN_LOCK is 1 from taking the lock until the detached session
# has started, which then owns it.
OWN_LOCK=0 JOB=""
# Every long wait is nap, not sleep: bash runs a trap only after the
# foreground command ends, so a Ctrl-C or kill during "sleep 300" would wait
# out the 300 s; wait returns on the signal at once.
NAP=""
nap() { sleep "$1" & NAP=$!; wait "$NAP"; NAP=""; }
cleanup() {
    [ -n "$NAP" ] && kill "$NAP" 2>/dev/null
    [ "$OWN_LOCK" -eq 1 ] && unlock
    [ -n "$JOB" ] && rm -f "$JOB"
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# Wait for the session's exit status, at most --timeout. The probe (a function)
# prints the status, or nothing while there is none; a failing probe means ssh
# failed.
wait_session() {   # probe-function where-the-results-are
    local probe=$1 where=$2 deadline last="no check yet" rc
    deadline=$(( $(date +%s) + TIMEOUT ))
    while :; do
        nap "$POLL"
        if st=$("$probe"); then
            [ -n "$st" ] && return 0
            last="no session.exit yet"
        else
            rc=$?; last="ssh failed (exit $rc)"
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            log "no session.exit after ${TIMEOUT}s (last check: $last); results left on $HOST ($where)"
            exit 4
        fi
    done
}

# == Windows hosts ==================================================================
# The same four steps as below, as PowerShell. Every remote command ends in an
# explicit exit status: powershell -c reports only whether the last command
# succeeded, and a native command's failure isn't a PowerShell error.
if [ "$exec_windows" -eq 1 ]; then
    LOCK=mtl5-campaign.lock
    rsh "try { New-Item -ItemType Directory -ErrorAction Stop \$HOME\\$LOCK | Out-Null; exit 0 } catch { exit 1 }" || {
        log "another session holds $HOST:~/$LOCK (remove it by hand if that session is gone)"; exit 3; }
    unlock() { rsh "Remove-Item -Force -ErrorAction SilentlyContinue \$HOME\\$LOCK" > /dev/null; }
    OWN_LOCK=1
    rsh "if (-not (Test-Path $RDIR\\.git)) { git init -q $RDIR }; git -C $RDIR config receive.denyCurrentBranch ignore; exit \$LASTEXITCODE" \
        || { log "cannot set up ~/$RDIR"; unlock; exit 3; }
    git -C "$ROOT" push -q -f "$HOST:$RDIR" "$SHA:refs/heads/campaign" 2> >(grep -v '^\*\*' >&2) || { log "push failed"; unlock; exit 3; }
    rsh "git -C $RDIR checkout -q --detach -f $SHA; if (\$LASTEXITCODE) { exit 1 }; git -C $RDIR clean -q -fdx -e build-platform; if (\$LASTEXITCODE) { exit 1 }; git -C $RDIR log --oneline -1" \
        || { log "checkout failed"; unlock; exit 3; }
    log "at $(git -C "$ROOT" log --oneline -1 "$SHA")"

    OUTREL="build-platform/results/campaign-$TAG"
    WOUT="$RDIR\\build-platform\\results\\campaign-$TAG"
    rsh "New-Item -ItemType Directory -Force $RDIR\\build-platform\\results | Out-Null; try { New-Item -ItemType Directory -ErrorAction Stop $WOUT | Out-Null; exit 0 } catch { exit 1 }" \
        || { log "session $TAG already exists on $HOST (~/$OUTREL); pick another --tag"; unlock; exit 2; }

    if [ "$REBOOT" -eq 1 ]; then
        bootq="(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')"
        # A missing first reading would make any later one look like a new boot.
        boot0=$(rsh "$bootq" | tr -d '\r') && [ -n "$boot0" ] \
            || { log "cannot read the boot time; not rebooting"; unlock; exit 3; }
        log "rebooting"
        # Its status is not checked: the connection may drop as the host goes
        # down. A restart that didn't happen shows as an unchanged boot time below.
        rsh "Restart-Computer -Force" > /dev/null 2>&1
        nap 30
        boot1=$boot0
        for _ in $(seq 1 60); do
            boot1=$(rsh "$bootq" 2>/dev/null | tr -d '\r') && [ -n "$boot1" ] && [ "$boot1" != "$boot0" ] && break
            nap 15
        done
        if [ -z "$boot1" ] || [ "$boot1" = "$boot0" ]; then
            log "did not reboot (boot time unchanged, or the host did not come back)"; unlock; exit 3
        fi
        log "up after reboot (booted $boot1); settling ${SETTLE}s"
        nap "$SETTLE"
    fi

    # Remote commands carry no double quotes: Windows sshd hands them to
    # PowerShell on a command line, whose argument parsing would strip them.
    # (So a profile path with spaces is not supported; ~ has none here.)
    #
    # The session script: the profile, its exit status and the unlock in ONE
    # process, started outside the ssh session so a dropped connection can't
    # end it or leave it without its session.exit.
    pq() { printf "'%s'" "${1//\'/\'\'}"; }   # a PowerShell single-quoted literal
    ps_extra=""
    for a in "${EXTRA[@]+"${EXTRA[@]}"}"; do ps_extra="$ps_extra $(pq "$a")"; done
    job=$(mktemp)
    JOB=$job
    cat > "$job" <<EOF
\$ErrorActionPreference = 'Continue'
\$out = Join-Path \$HOME '$WOUT'
\$env:CLASS = '$CLASS'; \$env:INTERLEAVE = '$INTERLEAVE'; \$env:PLATFORM_OUT = \$out
Set-Location (Join-Path \$HOME '$RDIR')
\$rc = 1
try {
    & powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks\\machines\\$PROFILE$ps_extra *>&1 | Out-File -Encoding ascii (Join-Path \$out 'session.log')
    \$rc = \$LASTEXITCODE
} finally {
    Set-Content -Encoding ascii (Join-Path \$out 'session.exit.tmp') "\$rc"
    Move-Item -Force (Join-Path \$out 'session.exit.tmp') (Join-Path \$out 'session.exit')
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path \$HOME '$LOCK')
}
EOF
    scp -q -o BatchMode=yes "$job" "$HOST:$RDIR/$OUTREL/session.ps1" 2> >(grep -v '^\*\*' >&2) || { log "session did not start (copy)"; unlock; exit 3; }
    rsh "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ' + [char]34 + (Join-Path \$HOME '$WOUT\\session.ps1') + [char]34; CurrentDirectory = (Join-Path \$HOME '$RDIR') }; if (\$r.ReturnValue -ne 0) { exit 1 }; exit 0" \
        || { log "session did not start"; unlock; exit 3; }
    OWN_LOCK=0   # started: the session releases the lock now (an interrupt from here on must not)
    nap 10
    # Launched, so the lock stays with the session even if it is slow to start
    # or ssh blips: unlocking here could let a second session in beside it.
    # wait_session's --timeout bounds a session that never really started.
    rsh "if (Test-Path $WOUT\\session.log) { exit 0 } else { exit 1 }" \
        || log "no session.log yet; waiting for the session anyway (it holds the lock; --timeout bounds the wait)"
    log "session $TAG started ($PROFILE, class $CLASS$([ "$INTERLEAVE" -eq 1 ] && echo ", interleave"))"
    probe() { rsh "if (Test-Path $WOUT\\session.exit) { Get-Content $WOUT\\session.exit }; exit 0" 2>/dev/null | tr -d '\r '; }
    wait_session probe "~/$RDIR/$OUTREL"
    log "session finished, exit $st"

    mkdir -p "$DEST/$MACHINE"
    [ -e "$DEST/$MACHINE/$TAG" ] && { log "$DEST/$MACHINE/$TAG exists; results left on $HOST (~/$OUTREL)"; exit 3; }
    scp -q -r -o BatchMode=yes "$HOST:$RDIR/$OUTREL" "$DEST/$MACHINE/" 2> >(grep -v '^\*\*' >&2) \
        && mv "$DEST/$MACHINE/campaign-$TAG" "$DEST/$MACHINE/$TAG" || { log "fetch failed"; exit 3; }
    log "results in $DEST/$MACHINE/$TAG"
    tail -3 "$DEST/$MACHINE/$TAG/session.log" | tr -d '\r' | sed "s/^/[$HOST]   /"
    exit "$st"
fi

# -- 1. the lock and the commit ---------------------------------------------------
# One session per host at a time: two would share ~/$RDIR, and the second
# checkout would change the first one's sources under its build. mkdir is the
# lock (atomic); the detached session removes it when it finishes.
LOCK=mtl5-campaign.lock
"${SSH[@]}" "$HOST" "mkdir $LOCK" 2>/dev/null || {
    if "${SSH[@]}" "$HOST" true 2>/dev/null; then
        log "another session holds $HOST:~/$LOCK (remove it by hand if that session is gone)"
    else
        log "cannot reach $HOST"
    fi
    exit 3
}
unlock() { "${SSH[@]}" "$HOST" "rmdir $LOCK" 2>/dev/null; }
OWN_LOCK=1
"${SSH[@]}" "$HOST" "git init -q $RDIR 2>/dev/null; cd $RDIR && git config receive.denyCurrentBranch ignore" \
    || { log "cannot set up ~/$RDIR"; unlock; exit 3; }
git -C "$ROOT" push -q -f "$HOST:$RDIR" "$SHA:refs/heads/campaign" || { log "push failed"; unlock; exit 3; }
"${SSH[@]}" "$HOST" "cd $RDIR && git checkout -q --detach -f $SHA && git clean -q -fdx -e build-platform && git log --oneline -1" \
    || { log "checkout failed"; unlock; exit 3; }
log "at $(git -C "$ROOT" log --oneline -1 "$SHA")"

# A reused --tag would leave the old session.exit for the poller to find.
OUTREL="build-platform/results/campaign-$TAG"
"${SSH[@]}" "$HOST" "cd $RDIR && mkdir -p build-platform/results && mkdir $OUTREL" 2>/dev/null \
    || { log "session $TAG already exists on $HOST (~/$RDIR/$OUTREL); pick another --tag"; unlock; exit 2; }

# -- 2. cold boot ----------------------------------------------------------------
# Reachable again is not rebooted: the boot id has to change.
if [ "$REBOOT" -eq 1 ]; then
    # A missing first reading would make any later one look like a new boot.
    boot0=$("${SSH[@]}" "$HOST" cat /proc/sys/kernel/random/boot_id) && [ -n "$boot0" ] \
        || { log "cannot read the boot id; not rebooting"; unlock; exit 3; }
    log "rebooting"
    "${SSH[@]}" "$HOST" "sudo -n reboot" > /dev/null 2>&1
    nap 20
    boot1=$boot0
    for _ in $(seq 1 60); do
        boot1=$("${SSH[@]}" "$HOST" cat /proc/sys/kernel/random/boot_id 2>/dev/null) && [ -n "$boot1" ] && [ "$boot1" != "$boot0" ] && break
        nap 15
    done
    if [ -z "$boot1" ] || [ "$boot1" = "$boot0" ]; then
        log "did not reboot (boot id unchanged, or the host did not come back)"; unlock; exit 3
    fi
    log "up after reboot ($("${SSH[@]}" "$HOST" uptime -p)); settling ${SETTLE}s"
    nap "$SETTLE"
fi

# -- 3. the session ----------------------------------------------------------------
# The profile, its exit status and the unlock run in ONE detached process, so a
# dropped ssh connection can't leave the session without its session.exit.
q_extra=""
[ "${#EXTRA[@]}" -gt 0 ] && q_extra=$(printf ' %q' "${EXTRA[@]}")
job="env CLASS=$CLASS INTERLEAVE=$INTERLEAVE PLATFORM_OUT=\"\$HOME/$RDIR/$OUTREL\" \
benchmarks/machines/$PROFILE$q_extra > $OUTREL/session.log 2>&1 < /dev/null; \
echo \$? > $OUTREL/session.exit.tmp; mv $OUTREL/session.exit.tmp $OUTREL/session.exit; rmdir \"\$HOME/$LOCK\""
# "cd X && cmd &" would background the whole list, whose subshell keeps ssh's
# output open, so ssh would wait for the session; "cd X; cmd &" doesn't.
"${SSH[@]}" "$HOST" "cd $RDIR || exit 1; setsid nohup bash -c $(printf %q "$job") > /dev/null 2>&1 < /dev/null &" \
    || { log "session did not start"; unlock; exit 3; }
OWN_LOCK=0   # started: the session releases the lock now (an interrupt from here on must not)
nap 5
# Launched, so the lock stays with the session (see the Windows path above).
"${SSH[@]}" "$HOST" "test -f $RDIR/$OUTREL/session.log" \
    || log "no session.log yet; waiting for the session anyway (it holds the lock; --timeout bounds the wait)"
log "session $TAG started ($PROFILE, class $CLASS$([ "$INTERLEAVE" -eq 1 ] && echo ", interleave"))"
probe() { "${SSH[@]}" "$HOST" "cat $RDIR/$OUTREL/session.exit 2>/dev/null; true"; }
wait_session probe "~/$RDIR/$OUTREL"
log "session finished, exit $st"

# -- 4. fetch -------------------------------------------------------------------------
mkdir -p "$DEST/$MACHINE"
rsync -a -e "ssh -o BatchMode=yes" "$HOST:$RDIR/$OUTREL/" "$DEST/$MACHINE/$TAG/" || { log "fetch failed"; exit 3; }
log "results in $DEST/$MACHINE/$TAG"
tail -3 "$DEST/$MACHINE/$TAG/session.log" | sed "s/^/[$HOST]   /"
exit "$st"
}
