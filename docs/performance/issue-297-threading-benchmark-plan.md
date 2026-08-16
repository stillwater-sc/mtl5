# Benchmarking Plan: On-Node Threading Rollout (#297)

Measure the benefit of the concurrency + scheduling work delivered in #297
(batches 1–10): dense LU/Cholesky/QR, the level-scheduled sparse Cholesky/LDLᵀ/LU
and supernodal triangular solves, the BLIS multi-loop (2D) GEMM, and the
element-wise expression sweeps. This document is two halves: a **benchmarking
plan** (what/why/how to measure) and an **implementation plan** (the concrete
harness changes).

The guiding principle from the existing scaling study
(`docs/performance/multicore-scaling-investigation.md`) still holds: **the experiment
is as easy to get wrong as the code.** We reuse its physical-core pinning
methodology rather than reinvent it.

---

## Part A — Benchmarking plan

### A.1 Questions to answer

1. **Speedup** — for each threaded kernel, how does wall-clock time scale from
   1 → N physical cores? Report speedup `t(1)/t(T)` and parallel efficiency
   `speedup/T`.
2. **Zero-overhead-at-T=1** — does the threaded path cost anything vs the
   pre-#297 serial code when `MTL5_NUM_THREADS=1`? (The design claims a no-op
   `parallel_for`/`parallel_ewise` at T=1.) This is a *regression* guard, not a
   scaling number.
3. **Shape sensitivity** — for GEMM, does the multi-loop (2D) grid actually help
   the wide/short and tall/thin shapes that ic-only parallelism could not fill?
4. **Scheduling payoff** — for the sparse solves, how much of the theoretical
   parallelism does level scheduling realize, and how does it track the
   problem's level structure (few wide levels vs many narrow levels)?
5. **Amortization** — for the sparse solves, is the schedule build cost paid back
   over repeated solves (the transient / mp-spice usage: one analyze+factor, many
   solves)?
6. **Correctness under load** — confirm the shipped invariant holds on the bench
   inputs: the T>1 result is **bit-identical** (`==`) to T=1 (not merely close).

### A.2 System under test (the #297 surface)

| Group | Kernel(s) | Entry point | Bench suite |
|-------|-----------|-------------|-------------|
| Dense L3 | multi-loop GEMM (2D grid) | `mult` / `detail::gemm_blocked` | `gemm` (extend: rectangular) |
| Dense factor | LU, Cholesky, Householder QR | `lu_factor`, `cholesky_factor`, `qr_factor` | `lapack` (exists) |
| Sparse solve | Cholesky, LDLᵀ, LU triangular solves | `*_numeric(...).solve()` | **new** `sparse-solve` |
| Sparse solve | supernodal LDLᵀ / LU triangular solves | `supernodal_*` `.solve()` | **new** `sparse-solve` |
| Element-wise | vector/matrix expression sweeps | `y = a+b`, `C = A+B`, `+=`/`-=` | **new** `ewise` |

Not in scope here (already covered by the #221 study or unchanged): raw
GEMV/axpy/dot/nrm2/SpMV scaling — reuse the existing `l1`/`l2` suites as sanity
controls but do not re-report them as new #297 results.

### A.3 Metrics & how they are computed

- **Wall-clock median** of K timed iterations after W warmups (existing
  `harness/timer.hpp` already does median/min/max/mean/stddev). Median, not mean,
  to resist scheduler/turbo jitter.
- **Speedup** `S(T) = median_ns(T=1) / median_ns(T)` and **efficiency**
  `E(T) = S(T)/T`, computed by `analyze_scaling.py` from the `<backend>-t<T>`
  labels.
- **GFLOP/s** where a flop count is meaningful (GEMM, dense factor). For the
  sparse solves and element-wise sweeps report **speedup + throughput**
  (elements/s or nnz/s) rather than GFLOP/s (the flop density is low / structural).
- **T=1 overhead ratio** — `median_ns(#297 build, T=1) / median_ns(pre-#297 tag,
  T=1)` per kernel. Target ≈ 1.00 (±noise).
- **Bit-identity flag** — each bench case, after timing, re-runs the kernel at
  T=1 and asserts `==` against the T=N output; emit a `bitexact=1/0` column. A `0`
  invalidates that row.

### A.4 Methodology (reused from the #221 study)

- **Pin to N *distinct physical* cores**, one logical id per core — no SMT
  siblings, no E-cores. `run_scaling.sh` already does this via `BENCH_PCPUS` +
  `taskset -c`; the new suites plug into the same driver.
- **SMT control** — repeat the top thread count on `taskset -c 0..2N-1` (N cores
  hyperthreaded) to show the SMT contribution separately and prove the affinity
  mask is honest.
- **Warmup + calibration** — W≥3 warmups; auto-scale iteration count so each
  case runs ≥ ~1 s of wall time (`harness/timer.hpp::calibrate`).
- **Fixed clocks where possible** — note turbo/`scaling_governor`; prefer
  `performance` governor and record the CPU model + governor in the CSV header /
  results doc. Turbo drift is why we pin and take medians.
- **One process per (kernel, T)** — the pool is sized once from
  `MTL5_NUM_THREADS` at first use, so thread count is a per-process axis (never
  change it mid-process).
- **Report the topology** — every results table names the CPU, physical-core
  count, governor, and compiler/flags. Numbers without topology are not
  comparable.

### A.5 Inputs / data sets

- **Dense GEMM** — square `{512, 1024, 2048, 4096}` plus **rectangular** to
  exercise the 2D grid: wide/short `m∈{64,256}, n∈{4096,8192}`, tall/thin
  `m∈{4096,8192}, n∈{64,256}`, fixed `k≈1024`. Random `double` and `float`.
- **Dense factor** — SPD (`AᵀA + nI`) for Cholesky; general random for LU/QR;
  `n ∈ {512, 1024, 2048, 4096}`.
- **Sparse solves** — real matrices via the existing `fetch_*` scripts
  (SuiteSparse). Pick a spread of **level structures**: a 2-D/3-D Laplacian
  (many narrow levels → limited parallelism), a banded/arrow system (few wide
  levels → good parallelism), and a couple of application matrices (circuit /
  structural). Build symbolic+numeric **once**; time **only** `solve()` (and a
  repeated-solve loop) — never fold factorization time into the solve number.
- **Element-wise** — vectors `n ∈ {1e5, 1e6, 1e7}`; matrices square `{1024,4096}`
  and wide/tall to expose the row-only sweep's shape gap (#313).

### A.6 Baselines & controls

- **Primary baseline: `MTL5_NUM_THREADS=1`** (same binary) — the honest measure
  of *our* speedup.
- **External references where they exist** (context, not the headline): OpenBLAS
  / BLIS / MKL for GEMM and dense factor (via `BLA_VENDOR`, already in
  `run_scaling.sh`); SuiteSparse KLU/CHOLMOD/UMFPACK for the sparse *solve* phase
  (`MTL5_WITH_SUITESPARSE_*`). These bound "how far off a tuned library are we,"
  but the #297 story is our own 1→N scaling.
- **SMT control** and a **pre-#297 tag** build for the T=1 overhead check.

### A.7 Pitfalls to actively avoid

1. **Dishonest affinity** (the #221 bug) — SMT siblings masquerading as cores.
   Mitigated by `BENCH_PCPUS` one-per-core + an explicit SMT control row.
2. **Folding build into solve** — the sparse win is in the *solve*; timing
   factorization+solve together hides it. Time the solve phase in isolation, and
   separately report schedule-build cost + break-even solve count.
3. **Too-small problems** — below the `parallel_for`/`parallel_ewise` grain the
   kernel runs serially by design; a "no speedup" at tiny n is expected, not a
   bug. Sweep sizes across the grain threshold and say so.
4. **Cold cache / first-touch NUMA** — warmups fix cache; on multi-socket note
   first-touch (single-socket dev box sidesteps it — record which).
5. **Turbo/frequency drift** — single-thread turbos higher than all-core, which
   *deflates* apparent speedup; report governor and take medians; optionally cap
   turbo for the headline table.
6. **Measuring a debug build** — Release + `-DMTL5_NATIVE_ARCH=ON` for the native
   kernels, matching `run_scaling.sh`.
7. **Grain mismatch on rare shapes** — e.g. the row-only matrix ewise sweep
   (#313) won't scale a 1×N expression; call it out rather than mislabel it.

### A.8 Success criteria

- Dense GEMM (square, N=2048+): efficiency comparable to the #221 result
  (~0.8–0.86 on 8 cores) and **positive scaling on wide/short shapes** where
  ic-only previously flat-lined (the multi-loop payoff).
- Dense LU/Cholesky/QR: measurable speedup at n≥1024 (bounded by the serial
  panel work — report where the ceiling is, don't oversell).
- Sparse solves: speedup that tracks level width; the wide-level cases show real
  gains, the deep-narrow cases are honestly reported as scheduling-limited.
- Element-wise: near-linear on large vectors / square matrices (memory-bandwidth
  bound ceiling noted).
- **All rows `bitexact=1`.** T=1 overhead ratio ≈ 1.0.

---

## Part B — Implementation plan

Reuse the harness (`benchmarks/harness/{timer,reporter,generators,runner}.hpp`,
`bench_all.cpp`, `run_scaling.sh`, `analyze_scaling.py`). Add suites, don't fork
the harness.

### Phase 0 — Reuse & verify what exists (0.5 day)
- Confirm `--suite lapack` (`bench_lu`/`bench_qr`/`bench_cholesky`) hits the
  **native threaded** factorization path (not an external LAPACK) and scales
  under `MTL5_NUM_THREADS` via `run_scaling.sh`. Add `qr` to the driver's suite
  list.
- Add a **`bitexact` re-check** to `harness/timer.hpp` (optional `verify` functor
  per case) and a `bitexact` CSV column in `reporter.hpp`.
- Add a `--threads-report` note capturing `MTL5_NUM_THREADS` + affinity into the
  CSV header for provenance.

### Phase 1 — Rectangular GEMM for the multi-loop grid (0.5 day)
- Extend `bench_gemm` (runner.hpp) to accept non-square `(m,n,k)` shapes and add
  a `gemm-rect` suite / `--shapes` spec (wide, tall, square).
- Driver: sweep the rectangular shapes across threads; the payoff row is
  wide/short speedup vs the ic-only baseline (reconstruct the ic-only baseline by
  forcing a 1×N grid, or compare against the pre-#311 tag).

### Phase 2 — Sparse triangular-solve scaling suite (2 days) — the core new work
- New `benchmarks/harness/sparse_suite.hpp` (or extend runner.hpp) with
  `bench_sparse_solve(rep, label, matrices)`:
  1. Load a matrix (Matrix Market via `io/`, from the `fetch_*` sets).
  2. `symbolic` + `numeric` factor **once** (untimed).
  3. Time `num.solve(x, b)` (median of K) — the level-scheduled path.
  4. Also time a **repeated-solve loop** (e.g. 50 solves) to model transient use.
  5. Emit schedule-build time separately (call the builder directly) → break-even.
  6. `verify`: `==` vs the dense serial reference solve.
- Cover `sparse_cholesky`, `sparse_ldlt`, `sparse_lu`, `supernodal_ldlt`,
  `supernodal_lu` (each has `factorL()`/`solve()` accessors from the #305–#309
  encapsulation).
- New `bench_sparse` target in `benchmarks/CMakeLists.txt` (mirrors `bench_klu`),
  or a `--suite sparse-solve` in `bench_all` gated on the data being present.
- Optional external reference: time the SuiteSparse solve phase (CHOLMOD/UMFPACK)
  under the same protocol when `MTL5_WITH_SUITESPARSE_*` is on.

### Phase 3 — Element-wise sweep suite (0.5 day)
- `bench_ewise` in runner.hpp: `y = a + b` (vector) and `C = A + B` / `C += A + B`
  (matrix), across vector sizes and matrix shapes (square + wide/tall to document
  #313). Throughput = elements/median_ns. `verify` `==` vs serial.
- Wire `--suite ewise` into `bench_all`.

### Phase 4 — Driver + analysis (0.5 day)
- Generalize `run_scaling.sh` → parameterize the suite (`SUITE=gemm|lapack|
  sparse-solve|ewise`) and its sizes/shapes/matrices, keeping the physical-core
  pinning and SMT control. Or add a thin `run_scaling_297.sh` that loops the four
  suites.
- Extend `analyze_scaling.py` to group by `(suite, kernel, backend, T)` and emit
  a per-kernel speedup/efficiency table + one combined plot; add a `bitexact`
  assertion (fail the analysis if any row is `0`).

### Phase 5 — Results write-up (0.5 day)
- `docs/performance/issue-297-threading-results.md`: topology, method (link this
  plan + the #221 study), per-kernel tables (speedup/efficiency/GFLOP/s or
  throughput), the multi-loop wide/short win, the sparse level-structure story,
  the T=1 overhead check, and the honest ceilings/limitations (serial panels,
  memory-bandwidth bound, #313 wide-matrix gap). Add to `FILE_MAP` in
  `docs-site/sync-content.mjs`.

### Deliverables / files
- `benchmarks/harness/timer.hpp`, `reporter.hpp` — `verify`/`bitexact` support
- `benchmarks/harness/runner.hpp` — rectangular `bench_gemm`, `bench_ewise`
- `benchmarks/harness/sparse_suite.hpp` — new sparse-solve suite
- `benchmarks/bench_all.cpp` — `gemm-rect`, `ewise`, `sparse-solve` suites
- `benchmarks/CMakeLists.txt` — `bench_sparse` target (if separate)
- `benchmarks/run_scaling.sh` (or `run_scaling_297.sh`), `analyze_scaling.py`
- `docs/performance/issue-297-threading-results.md` — the report

### Effort & sequencing
~4.5 engineer-days. Phase 2 (sparse) is the bulk and the most novel; Phases 0/1/3
lean almost entirely on existing code. Land as small PRs per phase (harness
change → suite → driver → results), each with a `bench` scope commit; none touch
library code, so they carry low risk and don't need the threaded/TSan CI lanes
(the correctness of the kernels is already proven there — this measures them).

### Risk notes
- **No multi-core CI measurement.** Perf numbers are dev-box artifacts, not CI
  gates (GitHub runners are noisy/shared and SMT-opaque). CI keeps proving
  *correctness* (bit-identity + TSan); this suite proves *benefit*, run on a
  known machine and recorded in the results doc.
- **Sparse data availability** — the `fetch_*` scripts pull SuiteSparse matrices;
  the suite must degrade gracefully (skip + warn) when a matrix is absent, as
  `bench_klu`/`bench_superlu` already do.
- Findings may feed the open optimization issues **#310** (schedule reuse in
  refactor) and **#313** (flatten the matrix ewise sweep) with concrete numbers.
