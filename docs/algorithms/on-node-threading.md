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

## Measurement methodology (pin to *distinct physical cores*)

Getting an honest scaling curve requires pinning each software thread to its own
**physical core** — not to SMT siblings and not to slower efficiency cores. This
matters especially on hybrid CPUs. On the i7-12700K used below, logical CPUs pair
up as SMT siblings on 8 P-cores and the 4 E-cores sit at the top:

```text
lscpu -e=CPU,CORE,MAXMHZ
# P-cores: logical 0,1 -> core0 ; 2,3 -> core1 ; ... 14,15 -> core7   (~4.9-5.0 GHz)
# E-cores: logical 16..19 -> cores 8..11                              (3.8 GHz)
```

So the **one-logical-CPU-per-physical-P-core** set is `0,2,4,6,8,10,12,14`. A
naive `taskset -c 0-7` instead spans only **four** physical cores (0–3) doubled by
SMT — an easy way to mismeasure "8 threads" as 4 hyperthreaded cores. Pin
explicitly:

```bash
MTL5_NUM_THREADS=8 taskset -c 0,2,4,6,8,10,12,14 ./bench   # 8 distinct P-cores
```

## Scaling results (indicative)

> **Indicative only.** Single-run on a shared **Intel i7-12700K**, fp64, each
> thread pinned to a distinct physical P-core (`0,2,4,6,8,10,12,14`), no SMT
> siblings, no E-cores. Directional (±10–15%), not an authoritative benchmark.

fp64 throughput (GFLOP/s) and speedup vs 1 thread:

| Kernel | 1T | 2T | 4T | 8T | speedup @4T | @8T |
|---|---|---|---|---|---|---|
| **GEMM** (N=2048) | 57.4 | 110.3 | 205.1 | 362.5 | 3.57× | **6.32×** |
| **GEMM** (N=4096) | 57.4 | — | — | 393.7 | — | **6.85×** |
| **GEMV** (N=8192) | 9.4 | 13.2 | 17.1 | 20.6 | 1.82× | 2.19× |
| **dot** (N=8M) | 4.6 | 7.6 | 8.6 | 11.1 | 1.89× | 2.43× |
| **axpy** (N=8M) | 3.4 | 5.3 | 6.4 | 6.9 | 1.87× | 2.02× |
| **scal** (N=8M) | 2.6 | 4.2 | 5.6 | 6.6 | 2.12× | 2.48× |
| **SpMV** (490k-row 2D Laplacian) | 1.00× | 1.33× | 1.79× | 1.92× | 1.79× | 1.92× |

### Reading the results

- **GEMM is compute-bound and scales with cores** — **6.32× at 8 cores** (N=2048),
  rising to **6.85×** at N=4096 (~86% parallel efficiency). Scaling *improves*
  with N, which rules out a memory-bandwidth ceiling: larger tiles do more
  arithmetic per parallel region, so the fixed per-region synchronization
  amortizes better. At small N (e.g. N=1024, ~5.1×) that per-region handoff is the
  main limiter; the remaining few percent is the all-core turbo drop.
- **The Level-1/2 kernels and SpMV are memory-bandwidth-bound.** They reach
  ~2–2.5× and then **plateau**: a couple of P-cores already saturate the (two
  DDR5 channel) memory bandwidth, so more threads cannot help. This is the
  expected roofline behavior, not a defect — the value is the ~2× a few cores
  extract over one, and (for the solvers) overlapping it with the compute-bound
  GEMM/preconditioner work.

> **Measurement caveat learned the hard way.** An earlier draft of this table
> reported GEMM at only 4.3× on "8 threads" — that run used `taskset -c 0-7`,
> which is four physical cores hyperthreaded, not eight cores. Pinning to distinct
> physical cores (above) shows the real 6.3–6.9×. Always verify the affinity mask
> against the core topology before drawing a scaling conclusion. The full
> investigation — hypotheses, controls, and attribution — is written up as a
> performance-engineering case study:
> [Multi-Core Scaling of MTL5's Threaded Kernels](../design/multicore-scaling-investigation.md).

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
