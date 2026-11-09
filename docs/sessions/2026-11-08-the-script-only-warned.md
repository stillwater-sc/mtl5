# Session: the script only warned

**Date**: 2026-11-08 (into 2026-11-09 UTC)
**Duration**: One long working session, with a context compaction partway through
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Close the solver defects that `mtl5_bench` had surfaced (#725, #731, #734).
Then turn the first platform sessions into committed data and card pages on the
docs site, with a script instead of hand copying.

**Merged:** #750, #751, #752, #753, #759, #761, #763, and the benchmark data
PRs #760, #762, #764 and #765. #758 merged ahead of #759; see below.

**Filed:** #754 (OpenBLAS threaded `ddot`), #756 (preflight and idle MSBuild
nodes).

## The through-line

Each benchmark machine was published by hand at first, and every hand step had
a matching way to go wrong.

- **Copying a results directory.** The first card pages were copied in without
  their raw JSON. Nothing linked them from the docs site, because nobody had
  added them to `FILE_MAP`.
- **Re-running a machine.** With no rule for what a second run should become,
  the Ryzen 9 8945HS got a second page, `ryzen-9-8945hs-2.md` (#758).
- **Card text.** Integer shapes such as `|v| <= 15` added table columns, since
  `mtl5_bench` didn't escape `|`.

`publish_platform.cmake` (#759) does all of that in one checked step:

- validates the session;
- copies it append-only to `benchmarks/data/<machine>/mtl5_bench/<UTC session>/`;
- writes the card page and registers it in `FILE_MAP`.

The first real publish, on the i7-12700K, then hit the one step the script
still left to a person: it only warned that `systems.md` didn't link the
page. #763 has it fill in that cell. Each automation removed one hand step and
showed the next.

## Work completed

### Solver and interface fixes

- **#750:** `ic_0` and `ildl` apply, and `ilu_0` and `ilut` setup, in O(nnz),
  bit-identical to before. At n = 16,384, the `ic_0` apply fell from 129 ms to
  0.17 ms.
- **#751 (#725):** `superlu.hpp` renames SuperLU's own BLAS prototypes while
  its header is parsed, so `<mtl/mtl.hpp>` compiles with BLAS and SuperLU
  both enabled. The OpenBLAS LAPACK lane in CI now links SuperLU.
- **#752 (#731):** TFQMR gets back its missing u recurrence. It no longer
  stalls at a 5–9% residual on unsymmetric systems. When the quasi-residual
  converges, it now recomputes b − A·x and restarts if that hasn't converged.
- **#753 (#734):** IDR(s) holds one ω per cycle of s + 1 steps (Sonneveld and
  van Gijzen 2008, Alg. 3.1). The iteration counts now fall with s: 71, 63,
  59 and 54 at s = 1, 2, 4 and 8 on the medium entries.

Two tests needed platform fixes:

- **Bit-identity tests:** they compare two template instantiations, and are
  pinned to `-ffp-contract=off`. Under contraction the two can round
  differently.
- **Float IDR(s) tolerance:** relaxed, because float's attainable accuracy is
  set by the drift between the updated and the true residual.

### Benchmark publishing

- **#759:**
  - `publish_platform.cmake` and its CI smoke step, run on Windows, macOS and
    Linux;
  - `|` escaped in card cells;
  - the card pages renamed `*-mtl5-bench.md` and added to `FILE_MAP`;
  - a "mtl5_bench card" column in the Systems table.
- **#761:**
  - `ci.yml` and `regression.yml` ignore `benchmarks/data/**`.
  - The sidecar check moved to a new `bench-data.yml`, which runs only when
    data changes.
  - A data PR now runs the sidecar check, the docs build and CodeQL. #762 and
    #764 confirmed it.
- **#763:**
  - The script sets the machine's card cell, `[Card](<machine>-mtl5-bench.md)`,
    adding "(preliminary)" when the page is preliminary.
  - The cell follows the page: publishing an older session leaves both alone.
  - The CI publish step had found its session with `ls` of the data
    directory. Once a real i7-12700K session was committed, that returned two
    names. It now takes the stamp from the results directory's name.
- **Data:** four machines published, all class `medium` and preliminary. The
  sessions are the Ryzen 7 2700X (#760), i7-12700K (#762), Ryzen 9 8945HS
  (#764) and Xeon E5-2420 v2 (#765).

### Issues filed

- **#754:** OpenBLAS's threaded `ddot` costs about 3 ms per call on a loaded
  machine, so BLAS-build Krylov solves run about 30× slower than native.
- **#756:** `preflight.ps1` fails on idle MSBuild worker nodes. Node reuse
  keeps them alive for about 15 minutes after a build made outside the script,
  and they block every platform run until they exit. Hit on a Windows lab
  machine.

## The parts that went wrong

- **#758 merged ahead of #759.** #759 was written to replace #758, but #758
  merged first. Its content on main was identical to what #759 expected to
  start from. So #759 merged main in and applied its corrections on top:
  - the `-2` page renamed;
  - links fixed;
  - the `FILE_MAP` entry added;
  - pipes escaped.

  Nothing was lost. Saying that one PR supersedes another doesn't stop the
  other from merging.
- **A rebase force-push was denied.** The permission classifier refused it on
  a feature branch. Merging main into the branch and pushing normally gave
  the same tree.
- **The session stamp.** The script first named a session after the time
  `summary.md` was written, which is the end of the run. It now uses the
  results directory's start stamp. That's why the Ryzen 9 session is
  `20261108T231039Z`, while the hand-made page said `20261108T231900Z`.
- **CMake `-P` scripting.**
  - It has no `{n}` regex quantifiers, so the digits are spelled out.
  - `-D` must come before `-P`: anything after it is a script argument.
  - `file(READ)` drops CRs, so CRLF handling written for the systems.md edit
    could never apply; it was removed.
- **The watcher stopped too early.** An empty check conclusion means the
  check is still running, and `grep -q` under `pipefail` exits early with
  SIGPIPE. Both made it report "done" too soon.

## Lessons

- **A warning in a publishing script is a hand step that hasn't been
  automated yet.** If the script knows what the fix is, it can apply it.
- **A check that lists a directory assumes how many things are in it.**
  `session=$(ls ...)` was right only until real data arrived.
- **Path filters have to follow the readers.** `benchmarks/data/` could be
  ignored by the build matrix only because its one reader, the sidecar check,
  moved to a workflow that triggers on it.
- **Get a bit-identity test's compile flags right.** Under FP contraction,
  two instantiations of the same code can legitimately round differently.

## Follow-ups

- **#737:** the benchmark campaign. Phase 0 pinning would make the cards
  final instead of preliminary. The i7-12700K session ran on 20 threads, E-cores
  included, though its Systems-table row says 8 P-cores.
- **#754:** a single-threaded or threshold-gated BLAS Level 1 for Krylov
  solvers.
- **#756:** make preflight tell idle, reusable MSBuild nodes from a running
  build. Also add `mtl5_bench` to the busy-process gate in `preflight.ps1` and
  `preflight.sh`, so a second platform run can't start during a measurement.
- **#749:** the generic `trmv` and `trsv` are bound by FMA latency. The plan
  is to interleave rows, keeping results bit-identical.
- **stillwater-sc/mtl5-python#121:** ingest the committed JSON for graphing.
