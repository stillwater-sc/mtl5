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
sizes a single process-wide pool. Unset or `=1` means fully serial: **no worker
threads are created** and each kernel runs its original serial path (a cheap
pool-size check is all the threading layer adds), so the default build behaves as
before.

Or size it from code, before anything uses the pool:

```cpp
#include <mtl/parallel/thread_pool.hpp>

int main(int argc, char** argv) {
    mtl::parallel::configure(8);   // overrides MTL5_NUM_THREADS
    // ...
}
```

`configure(n)` is clamped to the hardware concurrency, as the variable is, and
returns `false` without changing anything once the pool exists: after the first
`mtl::parallel::pool()` call or the first kernel that runs threaded.
`pool().size()` is the size in use.

## Running your own work on the pool

The pool is public API, `mtl::parallel::thread_pool`, reached through
`mtl::parallel::pool()` (#630). Code that runs its parallel work there shares one
set of workers with MTL5's kernels instead of oversubscribing the cores. The
older spelling, `mtl::detail::thread_pool::instance()` from
`<mtl/detail/thread_pool.hpp>`, still works and names the same pool.

`run(count, task)` is the primitive, and its contract is:

- `task(tid)` runs once for each `tid` in `[0, count)`; `count` is clamped to
  `size()`.
- The **calling thread runs tid 0**; the pool's workers run `1..count-1`.
- `run` **blocks** until every tid has completed, and every task's writes
  **happen-before** the caller's next action after it returns.
- A region started while another is in progress, nested inside a task or
  concurrently from another thread, **runs serially** on its caller, every tid
  in order. It never deadlocks or oversubscribes.
- The first exception from any tid is rethrown after all tids join.
- `run` **does not allocate**. It holds the task as a pointer to your callable
  plus a function pointer, so this holds whatever the callable captures.

`parallel_for` does not allocate either. `parallel_reduce` and
`parallel_accumulate` do: a threaded call keeps its per-chunk partials in a
`std::vector`.

Each of the three loops also takes a **team cap**, `max_threads`, before the
body: `parallel_for(n, grain, max_threads, body)`, and likewise for
`parallel_reduce<T>` and `parallel_accumulate<Acc>`. `0` means the whole pool.
Use it to run at a thread count of your own below the pool's. The grain still
applies under the cap. A capped reduction groups its sum by the capped team, so
it gives the same bits as an uncapped pool of that size.

## One pool across shared libraries

The pool is a static inside an inline function in a header-only library. A
program that includes MTL5 from several binaries can end up with **one pool per
binary**:

- every shared object built with `-fvisibility=hidden` keeps its own copy, and
  Python extension modules built with nanobind or pybind11 are built that way;
- every Windows DLL that includes the header has its own copy;
- on macOS, a single copy can't be relied on.

Each copy is a correct pool, so nothing deadlocks. But each one is sized from
`MTL5_NUM_THREADS`, so two such modules start twice the threads asked for, and
`configure(n)` sizes only the copy in the binary that calls it. On Linux with
default visibility, and in a statically linked program, there is one pool
already, and none of this is needed.

To give every binary the same pool (#639), let one shared library hold it and
have each module that uses MTL5 link that library:

```bash
cmake -B build -DMTL5_SHARED_POOL_LIBRARY=ON   # builds and installs mtl5_pool
```

```cmake
find_package(MTL5 REQUIRED)
target_link_libraries(my_extension PRIVATE MTL5::shared_pool)
```

Linking `MTL5::shared_pool` defines `MTL5_SHARED_POOL` for the module. Its
`pool()` and `configure()` then refer to the pool in `mtl5_pool`. The shared
library is loaded once per process, by every module that depends on it, so
they share the same pool and the same configuration. That holds even for
Python extensions loaded with `RTLD_LOCAL`. Each module looks the pool up once,
so `run()` stays inline, with no extra call per region.

Without MTL5's CMake, compile one translation unit of one shared library with
`MTL5_SHARED_POOL` and `MTL5_SHARED_POOL_BUILD` defined, and include
`<mtl/parallel/shared_pool_definition.hpp>` from it. Compile every other module
with `MTL5_SHARED_POOL` defined, and link that library.

Rules:

- **Define `MTL5_SHARED_POOL` for a whole module, or not at all.** Two
  translation units of one binary that disagree about it violate the
  one-definition rule.
- **Every module must be built against the same MTL5 pool header.** The exported
  functions carry a layout version (`shared_pool_v1`). It changes when the
  pool's layout does, so a module built against an incompatible header fails to
  link rather than misbehaving at run time.
- The default, header-only build is unchanged. A module that does not link
  `MTL5::shared_pool` keeps its own pool.

## Vendor libraries inside a region

With `MTL5_WITH_BLAS`/`MTL5_WITH_LAPACK`, `dense2D<float/double>` operations
call the vendor library, and with the SuiteSparse options CHOLMOD and SPQR are
called. These libraries run thread teams of their own. A call made from a tid of
an MTL5 region would start one team per tid: a region over 8 workers, each
calling an 8-thread vendor GEMM, puts 64 runnable threads on 8 cores. MTL5's own
kernels avoid this by running a nested region serially. Vendor calls follow the
same rule (#640):

- **Inside a region**, a vendor call runs on one thread where the vendor
  offers a control scoped to the calling thread. MTL5 sets it for the duration
  of the call and then restores the previous value.
- **Outside a region**, nothing changes. A direct `mult(A, B, C)` on BLAS types
  never enters the pool, so the vendor's team is the only parallelism there,
  and capping it would make the call serial.

"Inside a region" is `mtl::parallel::detail::in_region()`, a thread-local flag.
It is true on every tid of a region that the pool runs on more than one thread,
and on every tid of a region that runs serially because another region is in
progress. The serial fast path (a pool of one, or a region of one tid) leaves
the flag unchanged: such a region has no MTL5 parallelism for the vendor to
compete with. Under `MTL5_SHARED_POOL` the flag is exported with the pool, so a
task started in one module sees it in another.

| Backend | Control | Scope | Inside a region |
|---|---|---|---|
| MKL | `mkl_set_num_threads_local` | the calling thread | one thread¹ |
| BLIS ≥ 0.9 | `bli_thread_set_num_threads` | the calling thread; a new thread starts from `BLIS_NUM_THREADS` | one thread¹ |
| OpenBLAS | `openblas_set_num_threads`, `OPENBLAS_NUM_THREADS` | the whole process | **unchanged**; see below |
| CHOLMOD, SPQR | `cholmod_common::nthreads_max` (their own OpenMP), and the BLAS they call | that call | one thread, and the BLAS cap above |

¹ The vendor is set to one thread at the first vendor call of a region and
stays at one for the rest of the region. Later calls then cost a thread-local
check rather than two writes, and with BLIS a write costs more than a small
`ddot`. The region's calling thread (tid 0) gets its previous setting back when
the region ends, including when the region ends with an exception. A pool
worker runs nothing but region tids, so its setting stays at one. A vendor
setting that a task changes during a region is not capped again.

CMake works out the vendor from the symbol that vendor's thread control exports
in `BLAS_LIBRARIES`/`LAPACK_LIBRARIES`, and reports it as
`BLAS/LAPACK vendor: mkl|blis|openblas`. A generic `libblas.so.3` that forwards
to a vendor without exporting its control is reported as not identified. The
Debian/Ubuntu BLIS alternative is one such library. In that case vendor calls are
not capped; naming the vendor (`-DBLA_VENDOR=FLAME` for BLIS,
`-DBLA_VENDOR=Intel10_64lp` for threaded MKL) links the library that has the
control. `mtl::interface::blas_vendor()` and `blas_capped_in_region()` report the
result to code.

Measured on the i7-12700K: 64 dgemms in one 8-tid region, each vendor set to 8
threads ([Benchmark systems](../benchmarks/systems.md#a-vendor-blas-inside-an-mtl5-region-640)).
Uncapped, BLIS ran at 0.1–3 GFLOP/s; capped it ran at 405–507. MKL went from
328–381 GFLOP/s to 468–513. OpenBLAS cannot be capped per thread: uncapped it
varied between rounds from 102 to 407 GFLOP/s, and set to one thread for the
whole process it held 430–486.

### OpenBLAS: the application decides which runtime owns the cores

OpenBLAS's only control is process-wide, so MTL5 never changes it: doing so
would also make the application's direct calls serial. Choose one of two
configurations:

- **A threaded MTL5, a serial vendor.** Set `OPENBLAS_NUM_THREADS=1`, or call
  `openblas_set_num_threads(1)` once at startup. Vendor calls made inside
  regions then run serially, as they do under MKL and BLIS. Vendor calls made
  outside regions also run serially.
- **A threaded vendor, a serial MTL5.** Leave `MTL5_NUM_THREADS` unset (1), and
  give the cores to OpenBLAS's team.

The same choice applies to BLIS through its process-wide settings and to any
vendor that MTL5 cannot identify. The variables are `OPENBLAS_NUM_THREADS`,
`BLIS_NUM_THREADS`, `MKL_NUM_THREADS` and `OMP_NUM_THREADS`, the last for
OpenMP builds of any of them. MTL5 has no switch that sets these for you: each is
one line of the vendor's own API, and the application is the one that knows
which configuration it wants.

## What is parallelized

| Kernel | Where | Partition | Bit-identical to serial? |
|---|---|---|---|
| **GEMM** (`mult`, native blocked) | `detail/gemm_blocked.hpp` | ic-blocks of C | **Yes** |
| **GEMV** (row- and col-major) | `operation/mult.hpp` | output rows | **Yes** |
| **axpy**, **scal** | `operation/{axpy,scale}.hpp` | element range | **Yes** |
| **dot**, **two_norm** | `operation/{dot,norms}.hpp` | chunked reduction | No¹ |
| **dot**, **dot_real**, **two_norm**, **frobenius_norm** with an `Accumulator` | `operation/{dot,norms}.hpp`, `detail/accumulate.hpp` | chunked reduction, partials combined by `accumulator_traits::merge` | **Yes** for an exact accumulator, otherwise No¹ ² |
| **SpMV** (`compressed2D * vector`) | `mat/operators.hpp` | output rows | **Yes** |
| **GEMM**, **GEMV**, **SpMV** with an `Accumulator` | `operation/mult.hpp` | output rows, one accumulator per output entry | **Yes** |
| **LU** (native right-looking trailing update; accumulator path: column k and row k) | `operation/lu.hpp` | rows of the trailing matrix / entries of the column and row | **Yes** |
| **Cholesky** (native and accumulator paths) | `operation/cholesky.hpp` | rows below the diagonal | **Yes** |
| **Householder application** (QR, tridiagonalization; void path) | `operation/householder.hpp` | columns (left) / rows (right) | **Yes** |
| **Bunch–Kaufman LDLᵀ**, accumulator path | `operation/ldlt_bk.hpp` | rows of the on-demand Schur column | **Yes** |
| **Sparse triangular solves** (level-scheduled, #535) | `sparse/factorization/level_schedule.hpp` | rows within a level | **Yes** |

¹ The reduction kernels sum per-chunk partials, so the result is **deterministic
for a fixed thread count** but not bit-identical to the serial summation (the
grouping/associativity differs). The library's tolerances (`WithinRel`) account
for this. Every other kernel produces each output element from exactly one chunk,
so its result is bit-identical regardless of `MTL5_NUM_THREADS`.

² The accumulator-typed reductions (#582) split only when the operands are plain
dense storage (`ThreadableDenseVector`/`ThreadableDenseMatrix`) and the accumulator
satisfies `mergeable_accumulator`. An accumulator without a `merge` runs serially,
as before. With an exact accumulator, such as a Universal quire, every partial and
the merge are exact, so the result is the same bits for any thread count: on 2²⁰
posit<32,2> elements, `dot<quire<P>, P>` took 830 ms on 1 thread, 437 ms on 2,
228 ms on 4 and 125 ms on 8, with identical results. A rounding accumulator (a plain
wider type, `fma_accumulator`) regroups like the other reductions: a `double`
accumulator over the same data gave a different last few bits at each of 1, 2, 4 and
8 threads.

Because the iterative solvers (`mtl::itl` — CG, GMRES, BiCGSTAB, MINRES, …) and
the eigensolvers are written against `A * x`, `dot`, and `axpy`, they inherit the
SpMV/L1 threading with no solver-code changes.

## How it works

- **`parallel::thread_pool`** — a persistent pool of worker threads. `run(count,
  task)` runs `task(tid)` for `tid ∈ [0,count)` (the caller runs tid 0), blocking
  until all finish. One condition-variable handoff per parallel region — no
  thread is spawned per call.
- **`parallel_for(n, grain, body)`** — contiguous chunking of `[0,n)` for the
  element-wise/per-row kernels.
- **`parallel_reduce<T>(n, grain, map)`** — chunked reduction combined with
  `operator+` in chunk order, for `dot`/`nrm2`.
- **`parallel_accumulate<Acc>(n, grain, init, map, combine)`** — the same chunking,
  but each chunk accumulates into its own partial and the partials are folded in
  chunk order by the caller's `combine`, for accumulators that have no `+`. One
  thread, or too little work to split, is exactly `acc = init(); map(acc, 0, n)`.
  `detail::accumulate` drives it with `accumulator_traits::merge`.
- A **grain threshold** keeps small problems serial so the handoff never costs
  more than the work; nested/concurrent regions fall back to serial (no
  oversubscription or deadlock).

## Measurement methodology (pin to *distinct physical cores*)

Getting an honest scaling curve requires confining the run to distinct
**physical cores** — not to SMT siblings and not to slower efficiency cores.
`taskset` sets the *process* affinity mask (it does not hard-pin individual
threads); but when the mask is exactly N logical CPUs, one per physical core, the
OS keeps the N worker threads one-per-core in steady state. This matters
especially on hybrid CPUs. On the i7-12700K used below, logical CPUs pair up as
SMT siblings on 8 P-cores and the 4 E-cores sit at the top:

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

> **Indicative only.** Single-run on a shared **Intel i7-12700K**, fp64, the
> process affinity-masked to N distinct physical P-cores (`0,2,4,6,8,10,12,14`,
> one logical CPU each — no SMT siblings, no E-cores). Directional (±10–15%), not
> an authoritative benchmark.

fp64 throughput (GFLOP/s) and speedup vs 1 thread:

| Kernel | 1T | 2T | 4T | 8T | speedup @4T | @8T |
|---|---|---|---|---|---|---|
| **GEMM** (N=2048) | 57.4 | 110.3 | 205.1 | 362.5 | 3.57× | **6.32×** |
| **GEMM** (N=4096) | 57.4 | — | — | 393.7 | — | **6.85×** |
| **GEMV** (N=8192) | 9.4 | 13.2 | 17.1 | 20.6 | 1.82× | 2.19× |
| **dot** (N=8M) | 4.6 | 7.6 | 8.6 | 11.1 | 1.89× | 2.43× |
| **axpy** (N=8M) | 3.4 | 5.3 | 6.4 | 6.9 | 1.87× | 2.02× |
| **scal** (N=8M) | 2.6 | 4.2 | 5.6 | 6.6 | 2.12× | 2.48× |

**SpMV** (490k-row 2D Laplacian) is measured in **ms/matvec** (lower is better),
not GFLOP/s:

| threads | 1T | 2T | 4T | 8T | speedup @8T |
|---|---|---|---|---|---|
| ms/matvec | 2.18 | 1.64 | 1.21 | 1.13 | **1.92×** |

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
> [Multi-Core Scaling of MTL5's Threaded Kernels](../performance/multicore-scaling-investigation.md).

Every row added above is owner-computes: each output entry is produced by exactly
one thread, so the accumulator paths are bit-identical to serial for every
accumulation configuration. Only the four reductions in the `merge` row split a
single sum across threads. [The Accuracy
Architecture](../architecture/accuracy-architecture.md#4-parallel-reductions-threads-and-mpi)
explains the two kinds, and [Accumulation
Configurations](accumulation-configurations.md#determinism) gives the measured
determinism results.

## Not yet threaded

- **Transposed SpMV** (scatters into `y`, which would race across rows). Its
  accumulator path keeps one accumulator per output entry and stays serial, since
  privatizing them per thread would change the summation grouping.
- **Sparse numeric factorizations** (LU, Cholesky, LDLᵀ, supernodal, QR), `spgemm`,
  the preconditioners and the smoothers run serially, with or without an
  accumulator. Their solves use the level-scheduled triangular solves above.
- **Dense factorizations** are threaded within each step (above) but not blocked
  over the L3 building blocks, and `ldlt_factor` is serial.

## Determinism guidance

For reproducible-to-the-bit results across runs, keep `MTL5_NUM_THREADS` fixed
(or `=1`): every kernel except the reductions (`dot`/`nrm2`, and the
accumulator-typed ones with a rounding accumulator) is bit-identical across thread
counts, and even those are deterministic for a fixed thread count. If you need
serial-exact reductions, run those with `MTL5_NUM_THREADS=1`, or use an exact
accumulator, whose reductions give the same bits at any thread count.
