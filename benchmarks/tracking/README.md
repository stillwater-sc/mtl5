# Wall-clock tracking on the benchmark machines (#944)

The per-PR gate (`../regression/`) counts instructions. They are deterministic,
so a 2% threshold holds, but they do not see caches, SIMD width or threading.
This directory covers that part: it times the platform benchmark on the
benchmark machines and compares each session with that machine's previous one.
It reports and gates nothing: a timing on a real machine moves by a few percent
between sessions of the same commit, and #454 showed what a wall-clock gate on
shared runners does.

| file | what |
|---|---|
| `track.sh` | the driver: measures `origin/main` on every machine at once, publishes and compares |
| `hosts.txt` | the machines: name, ssh host (or `local`), profile |
| `track_sessions.py` | `compare` two sessions of one machine across commits; `page` writes `docs/benchmarks/tracking.md` |
| `test_track_sessions.py` | its tests, run by `bench-regression.yml` |

## One run

From a clean checkout used only for this:

```bash
benchmarks/tracking/track.sh                           # every machine in hosts.txt, class medium
benchmarks/tracking/track.sh --only i7-12700k,jetson-orin
benchmarks/tracking/track.sh --pr                      # and open the PR that publishes it
```

`track.sh` does the following, in order:
1. **Resolves `--ref`** (default `origin/main`) after a fetch, and checks it
   out in a scratch worktree under `build-platform/track/`. The sessions
   measure that commit with that commit's own scripts and profiles, whatever
   this checkout has checked out.
2. **Runs one session per machine, all at once.** A remote machine goes through
   `remote_session.sh`. The `local` one runs its profile directly. Each machine
   applies its own profile's pinning and clock policy (#737), as for a card.
   A medium session takes 15 to 50 minutes depending on the machine. Logs go
   to `build-platform/track/logs-<tag>/<machine>.log`.
3. **Publishes each session** that finished with every check passed. It uses
   `publish_platform.cmake -DDATA_ONLY=ON`, which copies the session into
   `benchmarks/data/<machine>/mtl5_bench/<tag>/` and stops there. The card page
   stays on the paired sessions it was published from: one session has no
   repeatability band to put on it.
4. **Compares each session** with the machine's previous published session. The
   report goes to `benchmarks/data/<machine>/tracking/<tag>.md` and `.json`.
5. **Regenerates** [`docs/benchmarks/tracking.md`](../../docs/benchmarks/tracking.md),
   which shows the latest report and the history for every machine.
6. **`--pr` commits and opens a pull request.** It commits the data, the reports
   and the page on a new branch `bench/track-<tag>`, pushes it and opens a pull
   request. Data-only changes run no build CI; `bench-data.yml` checks the
   sidecars. Without `--pr` the files stay in the checkout for review.

A machine that cannot be reached, or whose session fails a check, is reported and
skipped. The rest still publish, and the exit status is 1.

## The comparison

`track_sessions.py compare BASELINE NEW` matches runs by file name
(`native-t1.json`, ...) and entries by area, op, type and shape, as
`compare_sessions.py` does. The difference is the question:
`compare_sessions.py` measures the noise between two sessions of one commit,
and this measures the change between two commits.

- **The run's ratio** is the median, over its entries, of new time / old time,
  following the rule of `analyze_gate.py` (#327). One entry's ratio moves several
  percent between sessions of one commit. The median of a run's ratios repeats
  to about 0.5%: the cards' repeatability notes give median ratios of 0.998
  to 1.005 per run. A run outside ±2% (`--run-band`) is marked *slower* or
  *faster*.
- **Entries outside ±10%** (`--entry-band`) are listed, slowest first. Between
  two sessions of one commit, 5 to 12% of entries move by more than 5%. So an
  entry on the list is worth a look, not yet a finding.
- **Not compared:** an entry that is in only one session (added or removed
  between the commits), failed its check in either, or has no timing.
- **What changed besides the commit:** the report lists the sidecar fields that
  differ, such as the compiler, the flags, the kernel, the pin list or the
  clock policy. A shift that arrives with a new compiler is not necessarily a
  change in MTL5.

The exit status is 0 whenever the report was written, whatever it found. It is 1
when the sessions are of different machines or classes, or share no run.

## Scheduling

The decision in #944 is a local scheduled run, not a self-hosted GitHub runner.
Use a dedicated clone on the machine that drives the lab, one that can reach
every host in `hosts.txt` over ssh without a password:

```bash
git clone git@github.com:stillwater-sc/mtl5 ~/mtl5-track
```

Then add a crontab entry. Weekly at 03:00 on Sunday:

```cron
0 3 * * 0  cd ~/mtl5-track && benchmarks/tracking/track.sh --pr >> ~/mtl5-track.log 2>&1
```

Each run leaves the clone on its `bench/track-<tag>` branch. The next run
branches from the newly fetched `origin/main`, so this is harmless. `--pr`
needs `gh` to be logged in. Nightly also works (`0 3 * * *`), at the cost of a
PR a night.

## A new machine

1. Give it a profile in `benchmarks/machines/` and a row in
   `docs/benchmarks/systems.md`.
2. Add a line here in `hosts.txt`.

The first session has nothing to compare with. It is published, and the
comparisons start with the second.
