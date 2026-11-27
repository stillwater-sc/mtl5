# Session: the tracking run finds what the gate cannot

**Date**: 2026-11-27 (afternoon; the morning's log is `2026-11-27-the-range-is-part-of-the-answer.md`)
**Duration**: One PR (#962), one tracking session on six machines, two issues filed
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Finish #944 with its item 3: wall-clock tracking of `main` on the benchmark machines, as a report and not a gate. The CodeRabbit allowance was free reviews only.

**Merged:**

| PR | Issue | What |
|---|---|---|
| #962 | #944 (item 3) | `benchmarks/tracking/`; the instruction-count gate on arm64; per-configuration acceptances; `/bigobj` for `mtl5_bench` |

**Filed:**
- #963, MINRES about 45% slower per iteration since #869;
- #964, the native SOR sweep on AArch64 alternating between two speeds.

**Open:** #965, the two Ryzen NUC tracking sessions, opened by `track.sh --pr`.

## Decisions

- **On demand, not scheduled.** The plan was a weekly cron job. The user pointed out that the lab machines are power-cycled by hand, so nothing could count on them being up. `track.sh` is run manually when they are on. The README's cron section became "When to run it", and the decision is recorded in memory.
- **The instruction-count gate on arm64, in the same PR.** The user had taken the automated check to cover emulated platforms. It ran on Linux x64 only, and callgrind there instruments the real binary; it emulates nothing. The user asked for an arm64 job in #962.
- **Tracking sessions publish data only.** `publish_platform.cmake -DDATA_ONLY=ON` adds the session to `benchmarks/data/` and leaves the card page alone. A card is published from a session pair with a repeatability band (#737), and one tracking session has none.

## What the first run found

At f98777e, on all six machines (the two NUCs at 57512ad, once `/bigobj` had merged):

- **Every run on four machines was steady.** On the i7, Xeon, Jetson and Ventuno Q, each run's median ratio was within ±0.6% of 1. On the NUCs, one run in twelve, the Ryzen 7's `blas-t8`, was just outside the ±2% band, at 1.028.
- **MINRES, every machine.** `minres+diag` was 1.43–1.55× slower under GCC and 1.85–1.93× under MSVC, with the same 371 iterations.
  - **The commit.** A standalone timing with the headers from three commits isolated it to #869 (35dc6a9): 20.3 ms before it, 29.8 ms at it, 29.9 ms on main.
  - **The cause.** #869's overflow-safe M-norm makes four passes and 2n divisions per iteration, where one dot product was enough except near the range's ends.
  - **Why the gate missed it.** The instruction-count gate counts CG and BiCGSTAB, not MINRES.
  - Filed as #963.
- **SOR on ARM.** `sor1.5` was 1.7–1.8× slower on the two ARM machines only.
  - **Not one commit.** The published sessions show the native build's sweep alternating between two speeds from commit to commit. It went from slow to fast between bfa1c28 and 967d20a, when nothing under `itl/smoother/` changed.
  - **Gauss-Seidel too.** The Ventuno's native-fast Gauss-Seidel did the same once.
  - Filed as #964, as a code-generation question for GCC on AArch64.
- **`sparse_cholesky.factor` on a 3-D grid** was about 30× faster everywhere (#868).
- **The Ryzen NUCs could not build two of their three variants.** Under Visual Studio 2026 (MSVC 19.52), `mtl5_bench.cpp`'s native-fast and blas builds failed with C1128: the object file exceeded COFF's section limit. CI's MSVC 19.44 is still under it. `/bigobj` fixed it, checked by building native-fast on the Ryzen 9 NUC.

## Findings that changed what was built

- **`accepted.json` could not have worked.** It held one `max_ir` per kernel. Native and native-fast already count a kernel differently, and arm64 made four configurations. `max_ir` is now a map from configuration to count, and a bare number is refused. The file was empty, so nothing broke.
- **Every committed JSON needs a sidecar.** A tracking report's `.json` under `benchmarks/data/` would have failed `check_sidecars.sh`. The report now writes a `.sysinfo` naming the two sessions it was derived from, rather than taking an exemption.
- **CodeRabbit's one finding on #962 was real.** `track.sh` started each session as soon as its `hosts.txt` row was checked. A bad row later in the file then exited the script, and the exit trap removed the worktree that the local session was building from. The rows are now checked in a first pass, and an interrupted run stops its sessions before removing the worktree. It was checked by hand: a bad row, a duplicate row, and a TERM 3 s into a run on sw-13, which left no process, no lock and no session behind.

## Process notes

- **The arm64 counts are not x64's.** `two_norm` counts 27× and `lower_solve_level` 41× their x64 counts. The gate compares each configuration only with its own baseline, so this doesn't affect gating, but the cause is not known.
- **The tracking run in numbers.**
  - Medium sessions took 12 minutes on the i7 and 12–13 on the NUCs. The Xeon, Jetson and Ventuno Q had all finished within 26 minutes. All of a machine's runs happen at once, so a full pass takes as long as its slowest machine.
  - Each session adds about 650 KB under `benchmarks/data/`.
- **CodeRabbit's allowance.** It ran out once, for 45 minutes. A $0.25 offer was declined. The user merged #962 during the wait.
- **pkill.** `pkill -f 'pr-watch.sh 962'` matched its own command line and ended the command (exit 144), the same mistake as earlier in the session. The memory note on pgrep self-matching applies to `pkill -f` too.
