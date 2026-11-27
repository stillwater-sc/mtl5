# Session: a sidecar has to say enough

**Date**: 2026-11-27 (night; the evening's log is `2026-11-27-a-count-is-not-a-seed.md`)
**Duration**: Two PRs and an issue filed, plus docs
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

The order agreed for #885, #928, #929 and #930 was:
1. #928 and #929 together, the per-entry version and the sample spread;
2. then #930, complete sidecars;
3. then #885, the lab campaign.

This session did the first two, with CodeRabbit's free reviews only. It also filed #971.

**Merged:**

| PR | Issue | What |
|---|---|---|
| #973 | #928, #929 | Every `mtl5_bench` result carries a definition `version` and its sample quantiles; `check_results.py` |
| #975 | #930 | Every harness writes all 27 keys mtl5-python requires; `check_sidecars.sh` checks them |

**Filed:** #971, MINRES with a preconditioner beyond about 2^±700.

## Decisions

- **#928 and #929 in one PR.** Both add fields to `record` in `benchmarks/platform/run.hpp` and to the JSON writer in `mtl5_bench.cpp`. Two PRs would have conflicted and cost two CI cycles.
- **Always write a value (#930).** The user chose this over per-producer exemptions. mtl5-python's ingest has no exemption table, so an exemption in mtl5 would still have left those runs incomplete. A key that doesn't apply is written as `pool_size=1`, `bench_cpus=unpinned` or `clock_policy=not applied`. Recorded in memory.
- **A session from before #928 is compared as before.** A reader that treats a missing `version` as 0, as #928 specifies for mtl5-python, would make the first tracking comparison after #973 compare nothing. `track_sessions.py` and `compare_sessions.py` therefore treat a version change as a new series only when both sessions record a version.

## Findings

- **The definition changes since #721 can all be dated.**
  - `symv`'s byte model (#745) and TFQMR's matrix (#752) changed under the same id, so both start at version 2.
  - `ic_0` and `ildl` (#750) changed their shape string as well, so their ids changed and they stay at 1.
  - All eleven commits in `benchmarks/data` postdate both changes.
- **mtl5-python's `REQUIRED_KEYS` is the authority.** The issue described "the full key set". Reading `mtl5/bench/sysinfo.py` in the local mtl5-python clone gave the exact 27 keys, the two that may be empty, and two further rules: a key with two values is dropped, and a non-`key=value` line is a fault. `check_sidecars.sh` enforces all of them.
- **The tools found faults of their own kind.**
  - `run_int_bench.sh` appended `bench_all`'s console banner. Every key was present, but mtl5-python rated the run incomplete because of the two non-`key=value` lines. This turned up only because the fresh sidecars were run through mtl5-python's parser, not just through this repository's check.
  - `interleave_table.py` dropped four required keys on purpose, because the card session's values didn't describe the sweep's binary. Under "always write a value" they get the sweep's own values.
  - The KLU and SuperLU scoreboards had no producer of sidecars at all: theirs were written by hand.
- **Review found what the tests had not** (CodeRabbit on #973 and #975):
  - a NaN timing passed `check_results.py`'s order checks, and a string sample crashed it;
  - three harnesses skipped a sidecar the binary had failed to write, and exited 0;
  - `run_blocking_ab.ps1` recorded every core it was given, not the ones it pinned;
  - `sidecar.ps1`'s .NET write resolved a relative path against the process directory. It was reproduced on the Ryzen 9 NUC, both with and without the fix.
- **MINRES's range** (#971) is about 2^±700, from a 3/2 power: the unnormalized Lanczos vector's preconditioner solve.

## Process notes

- **What was run end to end:**
  - every harness that could be run in minutes on the i7;
  - the Windows column sweep on the Ryzen 9 NUC;
  - `run_sweeps.sh` and `run_scaling.sh` only as their finishing block, on real `bench_all` sidecars, since each builds four or five library variants.

  `run_nc_bench`, `run_vendor_nested` and three `.ps1` harnesses were not run. The PR says so.
- **The one-build rule held twice** while kpu-sim's `ninja -j 16` was running. A harness that builds at `--parallel $(nproc)` (`run_int_bench.sh`) was pre-built at -j4, so that its own build step was a no-op.
- **The safety check refused two test commands, and neither ran.**
  - An `rm -f $S/*` with an unguarded variable. It was rerun as `"${S:?}"/*`.
  - An `eval` with nested substitutions. It was rewritten as a plain script file.
- **`gh pr edit` failed** on the deprecated Projects (classic) GraphQL field. The PR body was updated through `gh api -X PATCH` instead.
- **CodeRabbit's allowance** ran out after every fix push, for 50 to 53 minutes, with offers of $0.75 to $3.00 declined. The user merged #973 and #975 by hand during the waits.
