# Session: it ran the same loop

**Date**: 2026-11-17
**Duration**: One working day, following the session of
[2026-11-16 into 2026-11-17](2026-11-17-a-budget-over-all-the-solves.md)
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

- **#894:** the `eigen()` decision #891 left open. The user chose option
  (a): drop the second eigenvalue run.
- **#884:** the column sweep was slower than the row kernel on small and
  short, wide column-major matrices. It was measured on the Jetson Orin, the
  Ryzen 7 8845HS, the Arduino Ventuno Q and the Xeon E5-2420 v2, plus the
  local i7-12700K.

**Merged:** #895 (#894), #896 (#884).

**Filed:** #894, split out of issue 888 at the user's request.

**Closed:** #872, #886, #887, #888.

## The through-line

Most of the time on #884 went into what the measurement was measuring.
Several times a number said "this kernel is slower" when the kernel was not
what had changed.

- **The issue's own numbers** came from `-O2` scratch builds that were
  probably not pinned to one core. Under CMake Release's `-O3`, GCC's code
  for the row kernel on a column-major A with `-march=native` is 3.4x slower
  than its generic code (2.2 us against 0.65 us at 64 x 64 f64). On the i7
  an unpinned run can also land on an E-core. Neither was in the issue.
- **"The sweep is 25-35% slower at n = 1024-4096"** was not the kernel. At
  that stride the new code ran the same one-column loop as #777's. In the
  bench's translation unit, GCC kept the eight-column instantiation out of
  line, and the same loop measured 20-25% slower inside it. Timed
  standalone, interleaved, the two were equal. The public calls now pick
  the one-column instantiation at the call site, so they run #777's
  function itself.
- **The public calls looked slow at large sizes** because they threaded over
  a pool of every CPU, all pinned to the one being measured. The runners now
  set `MTL5_NUM_THREADS=1`.
- **CI failed every LAPACK and Highway lane** on the new test, not on the
  kernels. Those builds hand the public `mult` and `trmv` to BLAS or the
  SIMD kernels, which don't promise the one-row loop's bits. A local
  LAPACK build hadn't reproduced it, because it didn't enable BLAS
  dispatch.

## Work completed

### `eigen()` from its own Schur run (#894, #895)

- **The change:** `eigen()` returns the Schur-form Francis run's eigenvalues,
  each with its own vector. It no longer runs `eigenvalue()` a second time
  or pairs the two by nearest distance.
- **Behaviour change:** the order can differ from `eigenvalue()`'s. It
  differs by 1.18 index by index on `test_eigen_general.cpp`'s 3x3.
- **Tests:** that test now compares the two as a set. The accumulator test
  counts one Hessenberg reduction instead of two.
- **Recorded:** the decision is in memory (`project_eigen_order_decision`).

### The column sweep (#884, #896)

- **Eight columns per pass** (`sweep_columns<8>`).
- **Short strips in registers:** a strip of at most 16 rows takes
  `sweep_fixed_rows<R>`, with GCC's loop vectorizer off for it. With AVX,
  GCC vectorized the column loop through shuffles, at half speed on the
  Xeon.
- **The triangular diagonal blocked** eight columns at a time. Up to
  n = 2048 the whole matrix is one strip, so the unrolled path had never
  run there.
- **Where it measured better, it stayed as it was:**
  - one column per pass at 8 KB or more between columns, and under MSVC;
  - the row kernels for `gemv` m <= 2, `trmv` n <= 32 and the forward
    substitution n <= 64.
- **The cutoffs** were chosen by simulating the policy from round 2's data.
  Round 3 confirmed them through the public calls.

**Results.** The public calls against #777's sweep, as a geometric mean over
the shapes:

| Kernel | Speedup |
|---|---|
| `gemv` | 1.15-2.33x |
| `trmv` | 1.17-1.71x |
| Forward substitution | 1.11-1.59x |

The bits are the same with contraction off.

**Still slower, under 0.5 us per call:**
- the forward substitution under Clang at n = 96-256;
- `trmv` on the Xeon at n = 32-96.

**Still slower, larger:** `gemv` with 48-64 rows by 4096 columns on the
Jetson and the Ventuno under GCC, up to 19 us.

**Published.** Round 4's sessions are under
`benchmarks/data/<machine>/column_sweep/`, and
`docs/benchmarks/column-sweep.md` has the tables, the method and the list
of slower cases.

**Tooling:**
- **`bench_column_sweep`:** times four things per shape: #777's sweep, the
  new sweep, the row kernel and the public call.
- **`run_column_sweep.sh` and `.ps1`:** build it at Release flags, generic
  and native, pin it with one thread, and write each CSV's sidecar.
- **`column_sweep_table.py`:** builds the page's tables from the committed
  data.

CodeRabbit's eight findings on #896 were all valid and are all fixed. Seven
of them came in while the CI failure was being fixed, and they were found
only when the merge gate counted the unresolved threads:

- `column_sweep_table.py` missed a 16% slowdown at its 0.85 cut;
- the PowerShell runner's affinity mask wrapped at CPU 32 and couldn't pin
  CPU 31 in a 32-bit host;
- `ExitCode` read `$null` without a cached handle;
- a restore command lost its quoting.

## Open

- **Clang's forward substitution at n = 96-256:** the blocked diagonal
  compiles less well there than the one-column loop. Raising the cutoff
  for Clang alone would cost GCC.
- **GCC's `-O3 -march=native` code for the row kernels on a column-major A
  on x86** is 2-5x slower than its generic code. The sweep now avoids it,
  but a row-major A uses those kernels too, and that case was not measured
  here.
