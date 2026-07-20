# On-Node Threading and Multi-Core Scaling

MTL5 parallelizes its hot kernels across CPU cores with a small persistent thread
pool (no OpenMP/TBB — just the C++ standard concurrency runtime). Threading is
**off by default** and opt-in per process.

## Enabling it

Set the environment variable before running:

```bash
MTL5_NUM_THREADS=8 ./your_program
```

`MTL5_NUM_THREADS` is read **once** (clamped to the hardware concurrency) and
sizes a single process-wide pool. Unset or `=1` means fully serial with **zero
overhead** (no worker threads are created), so the default build behaves exactly
as before.

## What is parallelized

| Kernel | Where | Partition | Bit-identical to serial? |
|---|---|---|---|
| **GEMM** (`mult`, native blocked) | `detail/gemm_blocked.hpp` | ic-blocks of C | **Yes** |
| **GEMV** (row- and col-major) | `operation/mult.hpp` | output rows | **Yes** |
| **axpy**, **scal** | `operation/{axpy,scale}.hpp` | element range | **Yes** |
| **dot**, **two_norm** | `operation/{dot,norms}.hpp` | chunked reduction | No¹ |
| **SpMV** (`compressed2D * vector`) | `mat/operators.hpp` | output rows | **Yes** |

¹ The reduction kernels sum per-chunk partials, so the result is **deterministic
for a fixed thread count** but not bit-identical to the serial summation (the
grouping/associativity differs). The library's tolerances (`WithinRel`) account
for this. Every other kernel produces each output element from exactly one chunk,
so its result is bit-identical regardless of `MTL5_NUM_THREADS`.

Because the iterative solvers (`mtl::itl` — CG, GMRES, BiCGSTAB, MINRES, …) and
the eigensolvers are written against `A * x`, `dot`, and `axpy`, they inherit the
SpMV/L1 threading with no solver-code changes.

## How it works

- **`detail::thread_pool`** — a persistent pool of worker threads. `run(count,
  task)` runs `task(tid)` for `tid ∈ [0,count)` (the caller runs tid 0), blocking
  until all finish. One condition-variable handoff per parallel region — no
  thread is spawned per call.
- **`parallel_for(n, grain, body)`** — contiguous chunking of `[0,n)` for the
  element-wise/per-row kernels.
- **`parallel_reduce<T>(n, grain, map)`** — chunked reduction combined with
  `operator+` in chunk order, for `dot`/`nrm2`.
- A **grain threshold** keeps small problems serial so the handoff never costs
  more than the work; nested/concurrent regions fall back to serial (no
  oversubscription or deadlock).

## Scaling results (indicative)

> **Indicative only.** Measured on a shared **Intel i7-12700K** (8 P-cores +
> 4 E-cores), single-run, `taskset`-pinned to logical CPUs 0–7 — which on this
> hybrid part spans P-core SMT siblings, so the 8-thread column is
> SMT/bandwidth-limited rather than 8 distinct physical cores. Numbers are
> directional (±10–15%), not an authoritative benchmark. Reproduce with
> `MTL5_NUM_THREADS=<n>` on a quiesced, P-core-pinned rig.

fp64 throughput (GFLOP/s) and speedup vs 1 thread:

| Kernel | 1T | 2T | 4T | 8T | speedup @4T | @8T |
|---|---|---|---|---|---|---|
| **GEMM** (N=2048) | 58.7 | 109.5 | 196.6 | 252.3 | **3.35×** | 4.30× |
| **GEMV** (N=8192) | 9.9 | 13.3 | 17.2 | 16.9 | 1.74× | 1.71× |
| **dot** (N=8M) | 4.6 | 7.4 | 8.7 | 8.8 | 1.89× | 1.92× |
| **axpy** (N=8M) | 3.5 | 5.3 | 6.3 | 6.2 | 1.79× | 1.76× |
| **scal** (N=8M) | 2.6 | 4.4 | 4.9 | 5.6 | 1.86× | 2.14× |
| **SpMV** (490k-row 2D Laplacian) | 1.00× | 1.42× | 1.74× | 1.73× | 1.74× | 1.73× |

### Reading the results

- **GEMM is compute-bound** and scales with cores (3.35× at 4T, 4.3× at 8T) — its
  arithmetic intensity keeps the cores fed.
- **The Level-1/2 kernels and SpMV are memory-bandwidth-bound.** They reach
  ~1.7–2× and then **plateau**: a few cores already saturate the memory channels,
  so adding threads cannot help. This is the expected roofline behavior, not a
  defect — the win from threading them is the ~2× bandwidth a couple of cores
  extract over one, and (for the solvers) overlapping that with the compute-bound
  GEMM/preconditioner work.

## Not yet threaded

- **Transposed SpMV** (scatters into `y`, which would race across rows).
- **Sparse triangular solve** — inherently sequential (forward/back
  substitution); parallelizing it needs level scheduling.
- **Dense factorizations** (LU/QR/Cholesky) on the in-house path — these need
  blocked algorithms over the L3 building blocks first.

## Determinism guidance

For reproducible-to-the-bit results across runs, keep `MTL5_NUM_THREADS` fixed
(or `=1`): every kernel except `dot`/`nrm2` is bit-identical across thread counts,
and even those are deterministic for a fixed thread count. If you need
serial-exact reductions, run those with `MTL5_NUM_THREADS=1`.
