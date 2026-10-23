# One thread pool per process, or one per component?

**Status:** design analysis. It follows the public thread-pool API
([#630](https://github.com/stillwater-sc/mtl5/issues/630), PR
[#637](https://github.com/stillwater-sc/mtl5/pull/637)) and the real-time pool
follow-up ([#636](https://github.com/stillwater-sc/mtl5/issues/636)).

MTL5 now has public concurrency API: `mtl::parallel::thread_pool`, reached
through `mtl::parallel::pool()`. The stack around it has several components that
want threads:

- MTL5's own kernels;
- synapse's graph runtime;
- the Python bindings, which share a process with NumPy and its BLAS;
- vendor BLAS/LAPACK and SuiteSparse, when MTL5 dispatches to them;
- reflex's real-time control loop.

Each component could run its own pool, or all of them could share one. This
document sets out the options and what each costs, and reviews what the
PLASMA/DPLASMA line of dense linear algebra runtimes learned about the same
question. It ends with a recommendation.

---

## 1. Who runs threads today

| Component | Threads it can start | How it is sized | Shares MTL5's pool? |
|---|---|---|---|
| MTL5 kernels (GEMM, GEMV, SpMV, LU, Cholesky, Householder, level-scheduled sparse solves, reductions) | `mtl::parallel::thread_pool`, persistent | `MTL5_NUM_THREADS` or `configure(n)`, once per process; default serial | it *is* the pool |
| synapse (`graph/session.hpp`) | none of its own | a per-session thread count, applied with `run(team, ...)` | **yes**, deliberately (#630's motivation) |
| mtl5-python | none of its own | `mtl5.set_num_threads(n)`, which sets `MTL5_NUM_THREADS` before the pool exists | yes |
| NumPy/SciPy in the same Python process | OpenBLAS, MKL or Accelerate threads, often an OpenMP runtime | `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `threadpoolctl` | **no** |
| Vendor BLAS/LAPACK behind `MTL5_WITH_BLAS`/`MTL5_WITH_LAPACK` | the vendor's own pool | the vendor's variables; MTL5 does not set them | **no** |
| CHOLMOD, SPQR, SuperLU behind the SuiteSparse/SuperLU options | OpenMP, or TBB for SPQR, depending on how they were built | `OMP_NUM_THREADS`, library-specific controls | **no** |
| Highway SIMD | none | n/a | n/a |
| reflex (control runtime) | its own loop threads | the application | not today; #636 is for it |

Two consequences follow from the table:

1. **The shared-pool model already covers everything MTL5 controls.** The
   oversubscription risk sits at the boundary with foreign runtimes: vendor
   BLAS, OpenMP inside SuiteSparse, and NumPy's BLAS.
2. **Process-wide is weaker than it looks.** `instance()` is a function-local
   static in an inline function in a header-only library. How many copies of it
   a process holds depends on how the binaries are linked (§4.3).

---

## 2. The options

**A. A pool per component.** MTL5, synapse, the Python layer and reflex each own
a private pool sized for their own work.

**B. One shared pool per process (the current design).** One set of workers is
sized once by the application. Every component dispatches its regions to it.
Nested or concurrent regions run serially on their caller.

**C. A shared pool with injectable executors.** The same workers as B, but a
component receives its executor from the application rather than finding it
through a global. Teams can be partitioned per caller: a cap (done in #637), or
reserved slices of the pool.

**D. A task-DAG runtime that owns the cores.** Kernels become tasks with declared
data dependencies, and a scheduler (QUARK, OpenMP tasks, PaRSEC, StarPU) runs
them. Kernels inside a task run serially. This is the PLASMA model (§5).

**E. Delegate to OpenMP or TBB.** MTL5 calls a standard runtime instead of
owning threads.

E was rejected when the pool was built (#221). OpenMP is optional and differs
between compilers: MSVC's support is incomplete, and Apple Clang ships no
`libomp`. TBB is a dependency. Both clash with the "header-only, no required
dependencies" rule. E appears below only to show what it would give up.

---

## 3. The criteria

1. **Oversubscription.** More runnable threads than cores.
2. **Straggler cost.** A fork-join region ends when its last tid finishes, so
   the slowest thread decides the time.
3. **Determinism.** MTL5 promises bit-identical results across thread counts for
   element-wise kernels, and fixed bits for a fixed team for reductions
   ([parallelization patterns](parallelization-patterns-and-pitfalls.md), §1).
4. **Composition.** What happens when parallel code calls parallel code.
5. **Isolation.** Whether one component's load can delay another's, and whether
   one component can have a different latency class (real time).
6. **Configuration.** How many settings a user must get right.
7. **Memory and locality.** Thread stacks, caches and NUMA placement.
8. **Portability.** Linux, macOS and Windows; static and shared linking.

---

## 4. Analysis

### 4.1 Oversubscription and the straggler effect

With `P` components each running a pool of `N` threads on `N` cores, up to `P·N`
threads are runnable. For throughput work that is not just a fair share of a
slower machine. Every MTL5 region is a fork-join with a barrier at the end, so
**a region takes as long as its most-delayed tid**. When the OS preempts one
worker to run another pool's thread, the whole region waits for that worker's
next time slice. On Linux the scheduler's slice is a few milliseconds. MTL5's
regions are sized by a grain of about 64K flops per chunk, which is tens of
microseconds. One preemption can therefore make a region ten to a hundred times
slower than its work. The other threads in the region either idle or spin while
they wait.

Spin-waiting runtimes make this worse. OpenBLAS's threads spin between calls,
and so does an OpenMP runtime under `OMP_WAIT_POLICY=active`. GNU OpenMP also
spins for a while before blocking by default, controlled by `GOMP_SPINCOUNT`. A
spinning thread from a foreign pool is runnable for its whole spin. It takes the
core the MTL5 worker needs, so the barrier waits on a thread that is doing no
useful work. This is the well-known NumPy + OpenBLAS + PyTorch oversubscription
problem, and the reason `threadpoolctl` exists.

**Per-component pools (A)** create this problem inside a process that is
otherwise well behaved: each pool is correctly sized on its own, and the sum is
too many. **A shared pool (B, C)** removes it for every component that shares
the pool. It leaves the foreign runtimes, which only configuration can fix
(§6, item 3).

### 4.2 Determinism

MTL5's chunking is a pure function of `(n, grain, team)`. The team is computed
from the pool size and the caller's cap, never from how many workers happen to
be free. When a region falls back to serial because the pool is busy, it still
runs every tid of the same team, in order, on the caller. **So contention never
changes the bits.** A `dot` gives the same result whether it ran alone or
behind another region.

All of the options keep this, provided each component's team is fixed by
configuration and not by load:

- A pool per component: the team is that pool's size. Deterministic.
- A shared pool: the team is the pool size or the caller's cap. Deterministic.
- An adaptive scheme that sizes teams from the workers currently idle (work
  stealing, TBB-style elastic teams) would break bit-identity for reductions.
  **MTL5 should not adopt it for reductions.** Element-wise kernels would stay
  bit-identical, because every output is written by one chunk whatever the
  split.

### 4.3 How many "process-wide" pools a process has

`thread_pool::instance()` holds `static thread_pool pool(...)` inside an inline
member function. The C++ language gives that one instance per program. Shared
libraries are outside the language, and the platforms differ:

- **Linux/ELF, default visibility.** GCC emits the static as a `STB_GNU_UNIQUE`
  symbol and the dynamic linker merges the copies, so one pool. Clang emits a
  weak symbol, which the dynamic linker also resolves to one copy.
- **Linux/ELF, `-fvisibility=hidden`.** Each shared object keeps **its own
  copy**. nanobind and pybind11 build extension modules with hidden visibility.
  So if mtl5-python's `_core` module and another extension module both include
  MTL5, the process has two pools. Each is sized from `MTL5_NUM_THREADS`, so the
  process starts twice the threads it was asked for.
- **Windows DLLs.** Each DLL that includes the header has its own copy, always.
- **macOS.** Weak definitions are merged only in some cases (flat namespace, or
  weak-definition binding into the main image). Do not rely on a single copy.

So **B is "one pool per binary that includes MTL5"**, and that equals one per
process only for static linking or default-visibility ELF. Nothing breaks when
there are several copies. Each copy is a correct pool, and none can deadlock
another. But on those platforms option B behaves like option A without anyone
having chosen A. The cure is C's injection point: a non-inline pool provider
that the application, or one designated shared library, defines (§6, item 2).

### 4.4 Composition: parallel code calling parallel code

MTL5's rule is that **a nested region runs serially on its caller**. With one
shared pool, a synapse region whose tasks call `mult` runs each `mult` serially
inside its tid, so the cores are used once. This is the same rule PLASMA arrived
at for the BLAS called inside its tile kernels (§5.2).

With per-component pools the rule does not cross components. A synapse worker
that calls into an MTL5 pool of its own starts a second team. `N` synapse tids
times `N` MTL5 tids gives `N²` runnable threads. OpenMP's
`OMP_MAX_ACTIVE_LEVELS` and TBB's arenas exist to contain exactly this, and both
work only because there is one runtime that can see the nesting.

### 4.5 Isolation: the case for a separate pool

The shared pool has one real cost: **concurrent regions from different threads
serialize.** Suppose the application calls a synapse session on thread A while
thread B runs a CG solve. Whichever starts second runs its whole region on its
own thread. That is correct and deadlock-free, but the second caller loses all
its parallelism for as long as the first region runs. A pool per component
avoids this, since both run in parallel. The cost is oversubscription whenever
their combined teams exceed the cores.

Isolation matters most when the components have **different latency classes**.
Throughput work wants to block when idle, saving power and leaving cores to
others. A hard real-time loop wants:

- workers that spin, or are pinned to reserved cores;
- no mutex on dispatch;
- a bounded worst case (#636).

These two policies conflict. If both share one pool, the real-time caller can
find the pool busy with a long GEMM and fall back to serial, or wait on a mutex
held by a preempted low-priority worker (priority inversion). **That is the one
case where a separate pool is the right design.** The real-time pool should own
a disjoint set of cores, and the throughput pool should be sized to the rest.
Two pools on separate cores do not oversubscribe each other.

### 4.6 Configuration

A pool per component means a thread-count setting per component, plus the
foreign ones (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, ...). The user has to
add them up correctly. A shared pool means one number, chosen by the one party
that knows the machine and the workload: the application. `configure(n)` (#637)
lets the application set it in code, and it returns `false` once the pool
exists. A library that tries to size the pool after an application has done so
fails visibly rather than quietly resizing it.

### 4.7 Memory and locality

Each worker reserves a stack: 8 MB of virtual address space on Linux by default,
and 1 MB on Windows. Physical use is small, so for `P·N` threads this matters
mainly on 32-bit or memory-capped targets. **Cache and NUMA locality matter more.**
A private pool per component evicts the other pools' working sets from shared L2
and L3 caches each time it runs. A shared pool with a stable tid-to-core mapping
keeps a component's chunk `t` on the same core from region to region. That
helps GEMM's packed panels, and the BLIS-style blocking in
`detail/gemm_blocked.hpp` depends on it. On multi-socket machines, one pool per
NUMA domain, each pinned to its socket, is the established exception (PaRSEC
and StarPU both bind workers per socket). It is a partition of one runtime's
cores, not a pool per component.

### 4.8 Summary

| Criterion | A: pool per component | B: one shared pool | C: shared pool + injection/partitions | D: task-DAG runtime |
|---|---|---|---|---|
| Oversubscription | **bad**: the sum of `P` correctly sized pools | good inside MTL5's reach | good, and extends across shared-library boundaries | good: the runtime owns the cores |
| Straggler cost | high under contention | low | low | low; dynamic scheduling also hides imbalance |
| Determinism | yes, team fixed per pool | yes, team fixed by size and cap | yes | yes for tile algorithms (§5.3), if reductions keep a fixed tree |
| Nested parallelism | `N²` threads across components | serial nesting, cores used once | same as B | tasks run serial kernels by construction |
| Concurrent callers | run in parallel | **second caller serializes** | can reserve slices so both run | interleaved at task level |
| Real-time isolation | possible, but not by default | **no**: shares mutex and queue | a separate RT pool on reserved cores | no: a general scheduler has no hard bound |
| Configuration | one setting per pool | one number | one number plus optional partitions | one number, plus a tile size |
| Locality | pools evict each other's caches | stable tid-to-core | stable, NUMA partitions possible | depends on the scheduler; locality-aware ones exist |
| Portability, header-only | yes | yes, but copies multiply across DLLs and hidden-visibility `.so` files (§4.3) | needs one non-inline definition somewhere | heavy: a runtime dependency or a large subsystem |
| Cost to build | low | done | small | large |

---

## 5. What PLASMA and DPLASMA learned

The PLASMA project (Parallel Linear Algebra Software for Multicore
Architectures, Dongarra et al., ICL/UTK) set out to answer "how should dense
linear algebra use many cores". It answered it three times: with a static
scheduler, a dynamic one (QUARK), and then OpenMP tasks. PLASMA itself is
shared-memory. Its distributed-memory line, which runs over MPI, is **DPLASMA on
the PaRSEC runtime**. SLATE, ScaLAPACK's successor, uses MPI with OpenMP tasks
on each node. The tile-kernel side of PLASMA is summarized in kpu-sim's
`docs/plans/plasma-tile-algorithms.md`. This section covers only the runtime
side.

### 5.1 Tile algorithms need a runtime that owns the cores

LAPACK gets its parallelism from a threaded BLAS: each step of a blocked
factorization calls a parallel GEMM, a fork-join region. The panel factorization
between those calls is sequential and memory-bound, so most cores idle during
it. As core counts grew, that idle time dominated (Buttari, Langou, Kurzak and
Dongarra, *A class of parallel tiled linear algebra algorithms for multicore
architectures*, Parallel Computing 35(1), 2009).

PLASMA's fix was to reorganize the factorizations into **DAGs of tile kernels**
and let a scheduler run any task whose inputs are ready. The panel of step
`k+1` can then overlap the trailing update of step `k` (lookahead). That only
works if **one scheduler sees all the work**. Two independent pools, one running
panels and one running updates, cannot overlap them, because neither knows the
other's dependencies.

### 5.2 Inside a task, BLAS runs on one thread

PLASMA's documentation and build require the vendor BLAS to run on **one
thread**. The tile kernels call sequential `dgemm` and `dtrsm`, and PLASMA's
scheduler provides all the parallelism. A multithreaded vendor BLAS under PLASMA
oversubscribes the machine and slows it down, which is §4.4's `N²` problem.

This is the strongest published evidence on the question this document asks:
**two runtimes that both think they own the cores lose to one that does.** It is
also what MTL5's nested-serial rule does automatically for MTL5 kernels called
from a region on the shared pool. It does not happen for a vendor BLAS behind
`MTL5_WITH_BLAS`, which keeps its own threads (§6, item 3).

### 5.3 Static, then dynamic, then standard

- **Static scheduling (PLASMA 2.0).** Each core stepped through a precomputed
  share of the DAG, synchronizing on progress tables. Overhead was lowest and
  the schedule was deterministic. But it was rigid: it could not absorb load
  imbalance or interference from other work on the machine, and every algorithm
  needed a hand-written schedule.
- **Dynamic scheduling with QUARK (PLASMA 2.4 onward).** QUARK (QUeuing And
  Runtime for Kernels; YarKhan, Kurzak and Dongarra, ICL technical report
  ICL-UT-11-02, 2011) inferred dependencies from each task's declared
  `INPUT`/`OUTPUT`/`INOUT` tiles, in the order the tasks were inserted. It
  hid imbalance and allowed lookahead without hand scheduling, at a per-task
  overhead of a few microseconds. That overhead sets a floor on the tile size:
  tiles of a few hundred on a side kept the overhead small relative to the
  `O(b³)` kernel work.
- **OpenMP tasks (PLASMA 17).** PLASMA was ported to OpenMP 4 `task depend`
  clauses (YarKhan, Kurzak, Luszczek and Dongarra, *Porting the PLASMA numerical
  library to the OpenMP standard*, International Journal of Parallel
  Programming, 2017). The paper reports performance close to QUARK's. The
  argument for the port was that **a standard runtime composes with the
  application's other OpenMP code; a private one does not.** The project gave up
  its own scheduler to avoid being a second runtime in the process.

**On determinism:** with `inout` dependencies resolved in insertion order, every
update to a tile is applied in the same sequence on every run. Only the
interleaving between tiles changes, so the result is bit-reproducible whatever
the schedule. That matches MTL5's §1 property. A dynamic DAG runtime does not
force MTL5 to give up bit-identity. Only reductions over a variable tree would,
and tile algorithms fix their trees (flat or binary) in the algorithm.

### 5.4 Distributed memory: DPLASMA on PaRSEC

PaRSEC (formerly DAGuE; Bosilca, Bouteiller, Danalis, Herault, Lemarinier and
Dongarra, *DAGuE: A generic distributed DAG engine for High Performance
Computing*, Parallel Computing 38(1–2), 2012) runs a **parameterized task
graph**. It is a compact symbolic description of the DAG that each MPI process
evaluates locally, so no process needs the whole graph. DPLASMA (Bosilca et al.,
*Flexible development of dense linear algebra algorithms on massively parallel
architectures with DPLASMA*, IPDPS Workshops, 2011) built ScaLAPACK-class
factorizations on it. The lessons that bear on this document:

- **One runtime per process, owning every core.** Each MPI process runs
  PaRSEC's compute threads, bound to cores, plus **one communication thread**
  that drives MPI. There is no second pool. Computation and communication
  overlap because the one scheduler knows which tasks wait on remote data.
- **Communication gets a dedicated thread, not a share of the compute pool.**
  MPI progress needs a thread that is never stuck in a long kernel. Putting
  communication on its own thread is the same separation as §4.5's real-time
  pool: work with a different latency class gets its own execution resource,
  not a slot in the throughput pool.
- **Bind per NUMA domain.** PaRSEC and StarPU bind workers to cores and keep a
  tile's tasks on its socket's workers where possible. This partitions one
  runtime by locality. It does not add runtimes.
- **The MPI threading level is a design input.** A runtime with one
  communication thread can use `MPI_THREAD_FUNNELED` or `SERIALIZED`, which MPI
  implementations support better than `MPI_THREAD_MULTIPLE`. When every
  component in a process may call MPI from its own threads, the process needs
  `MULTIPLE`. Per-component runtimes push toward the weakest-supported mode.

SLATE (Gates, Kurzak, Charara, YarKhan and Dongarra, *SLATE: design of a modern
distributed and accelerated linear algebra library*, SC 2019) takes the same
structure to GPUs. On each node, OpenMP tasks schedule tiles and call batched
BLAS on GPUs. MPI moves tiles between nodes. The node runtime is still the
single owner of that node's cores.

### 5.5 Composing runtimes that already exist

Separate runtimes do end up in one process: a NumPy BLAS next to MTL5, or
CHOLMOD's OpenMP next to synapse. For that case the research gives a narrower
answer. **Lithe** (Pan, Hindman and Asanović, *Composing parallel software
efficiently with Lithe*, PLDI 2010) showed that runtimes compose well when a
parent lends hardware threads ("harts") to a child explicitly and the child
returns them when done. It also showed that each runtime grabbing its own OS
threads is what goes wrong. TBB arenas, OpenMP's nesting controls and
`threadpoolctl` are partial, runtime-specific forms of the same idea: one party
decides how many cores each runtime gets. BLIS goes furthest within one library:
a per-call runtime object (`rntm_t`) carries the thread count and the loop
partition. Each call can use a different share of the cores without a second
pool. `max_threads` in #637 is MTL5's equivalent.

---

## 6. Recommendation

**Keep one shared pool per process as the default (option B). Close its gaps
with option C's two mechanisms. Allow exactly one kind of separate pool: one with
a different latency class, on reserved cores.** In order of value:

1. **Done in #637.** Public API (`mtl::parallel::pool()`) with a documented
   contract. `run()` does not allocate. `configure(n)` lets the application set
   the size. A per-call cap (`max_threads`) lets a component take a share of the
   pool without a pool of its own. synapse and mtl5-python should switch from
   `mtl::detail::thread_pool` to the public spelling. Their behavior does not
   change.

2. **A single-definition pool provider, so shared libraries share the pool
   (§4.3).** Suggested shape: `mtl::parallel::pool()` keeps working as it does,
   but an application or a host library can install the pool once, through a
   non-inline hook in one translation unit, or through
   `MTL5_POOL_PROVIDER`-style linkage that a static or shared "runtime" library
   provides. Python extension modules and Windows DLLs would then share one set
   of workers. The hot path must not take a virtual call per region (the same
   constraint #636 sets): the provider resolves once to a concrete pool, and
   `run()` stays inline. Tracked in
   [#639](https://github.com/stillwater-sc/mtl5/issues/639).

3. **Make foreign runtimes single-threaded under a threaded MTL5 pool (§5.2).**
   When `MTL5_WITH_BLAS`/`MTL5_WITH_LAPACK` dispatch is active and
   `pool().size() > 1`, the vendor BLAS should run on one thread. Otherwise MTL5
   regions and vendor regions compete for the same cores. At minimum this goes in
   `on-node-threading.md` and the build-options page. Better, MTL5 calls the
   vendor's set-threads entry point (`openblas_set_num_threads`,
   `mkl_set_num_threads_local`, or `BLIS_NUM_THREADS` semantics) on a dispatch
   path, when that symbol is linked. The same applies to CHOLMOD and SPQR built
   with OpenMP/TBB. mtl5-python can recommend `threadpoolctl` to cap NumPy's
   BLAS when `mtl5.set_num_threads(n > 1)`. Tracked in
   [#640](https://github.com/stillwater-sc/mtl5/issues/640).

4. **A real-time pool on its own cores (#636).** The one legitimate second pool
   (§4.5, §5.4's communication thread). It has the same interface and contract,
   mutex-free dispatch, spin-then-futex waits, and a documented worst case. The
   application gives it reserved cores (affinity is the application's call, not
   MTL5's) and sizes the throughput pool to the remaining cores. The two pools
   then never compete for a core. Kernels called from a real-time tid must run
   serially or on the real-time pool. They must not take the throughput pool's
   mutex.

5. **Partitions for concurrent callers (optional, measure first).** If two
   application threads routinely run large regions at the same time (a synapse
   session alongside a solver), add reserved sub-teams of the one pool, in the
   spirit of TBB arenas. Each caller gets a fixed slice and its own busy flag.
   Teams stay fixed by configuration, so determinism holds (§4.2). Until a
   workload shows the serialization in §4.5, the cap from item 1 is enough.

6. **A DAG runtime, only when the tile algorithms need one (option D).** If MTL5
   gains PLASMA-style tile factorizations, which would let lookahead hide panel
   time (§5.1), build the scheduler **on the same workers**, as one runtime that
   owns the cores. Do not add it as a second pool. Kernels inside a task run
   serially, which MTL5's nested-serial rule already guarantees. Insertion-order
   dependency resolution keeps the bits fixed (§5.3). For distributed memory, the
   PaRSEC structure applies: compute workers plus one communication thread, per
   process (`MPI_THREAD_FUNNELED`). The element serialization point proposed in
   the [Elemental assessment](https://github.com/stillwater-sc/mtl5/blob/main/docs/other/mtl5-elemental-assessment.md) lets MTL5
   keep MPI out of its core.

**What not to do.** Do not give each component its own throughput pool. It
re-creates, inside one process, the oversubscription that PLASMA had to tell
users to switch off in their vendor BLAS. Do not size teams from idle workers
for reductions, because that gives up bit-identity. And do not let a library size
the pool behind the application's back: `configure` fails once the pool exists,
and that should stay true.

---

## References

- Buttari, Langou, Kurzak, Dongarra. *A class of parallel tiled linear algebra
  algorithms for multicore architectures.* Parallel Computing 35(1), 2009.
- YarKhan, Kurzak, Dongarra. *QUARK Users' Guide: QUeueing And Runtime for
  Kernels.* ICL technical report ICL-UT-11-02, University of Tennessee, 2011.
- YarKhan, Kurzak, Luszczek, Dongarra. *Porting the PLASMA numerical library to
  the OpenMP standard.* International Journal of Parallel Programming 45, 2017.
- Bosilca, Bouteiller, Danalis, Herault, Lemarinier, Dongarra. *DAGuE: A generic
  distributed DAG engine for High Performance Computing.* Parallel Computing
  38(1–2), 2012.
- Bosilca et al. *Flexible development of dense linear algebra algorithms on
  massively parallel architectures with DPLASMA.* IEEE IPDPS Workshops, 2011.
- Gates, Kurzak, Charara, YarKhan, Dongarra. *SLATE: design of a modern
  distributed and accelerated linear algebra library.* SC 2019.
- Pan, Hindman, Asanović. *Composing parallel software efficiently with Lithe.*
  PLDI 2010.
- Van Zee, van de Geijn. *BLIS: A framework for rapidly instantiating BLAS
  functionality.* ACM TOMS 41(3), 2015. Threading: Smith et al., *Anatomy of
  high-performance many-threaded matrix multiplication*, IPDPS 2014.
- MTL5: [on-node threading](../algorithms/on-node-threading.md),
  [parallelization patterns and pitfalls](parallelization-patterns-and-pitfalls.md),
  [multicore scaling investigation](../performance/multicore-scaling-investigation.md),
  [Elemental assessment](https://github.com/stillwater-sc/mtl5/blob/main/docs/other/mtl5-elemental-assessment.md); issues
  [#221](https://github.com/stillwater-sc/mtl5/issues/221),
  [#630](https://github.com/stillwater-sc/mtl5/issues/630),
  [#636](https://github.com/stillwater-sc/mtl5/issues/636),
  [#639](https://github.com/stillwater-sc/mtl5/issues/639),
  [#640](https://github.com/stillwater-sc/mtl5/issues/640).
- kpu-sim: `docs/plans/plasma-tile-algorithms.md` (the PLASMA tile-kernel
  catalog and its mapping to the KPU).
