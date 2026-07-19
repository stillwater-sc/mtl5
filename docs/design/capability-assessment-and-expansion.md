# MTL5 Capability Assessment & Expansion Analysis

**Date:** 2026-07-19
**Scope:** Functionality, performance, distributed-memory expansion, hardware-accelerator expansion
**Basis:** Source survey of `include/mtl/`, `tests/`, `benchmarks/`, `CMakeLists.txt`

MTL5 is a C++20, header-only linear algebra library — a concepts-based redesign of
MTL4 aimed at mixed-precision algorithm design with custom number types. This
document assesses where the library stands on four axes and what each expansion
direction would require.

## At-a-glance scorecard

| Axis | Maturity | One-line summary |
|---|---|---|
| **Functionality** | ● ● ● ● ○ Mature | Broad dense + sparse + iterative + eigen coverage; a few advanced gaps |
| **Performance — single-thread** | ● ● ● ● ○ Strong | Highway SIMD + BLIS-style blocked GEMM at ~80–84% of OpenBLAS |
| **Performance — multi-thread** | ● ● ○ ○ ○ Nascent | Only one kernel (GEMM) threads; everything else is serial |
| **Distributed memory** | ● ○ ○ ○ ○ Absent | No MPI/partitioning; but the solver contract is nearly drop-in ready |
| **Hardware accelerators** | ● ○ ○ ○ ○ Absent | No GPU code; CPU SIMD only; a clean backend-dispatch hook exists |

Legend: ○ absent · nascent · developing · strong · mature.

The two "Absent" axes share a single missing prerequisite — a **memory-space
abstraction in the storage layer** — which is the first thing either expansion
must add.

---

## 1. Functionality

### Strengths — broad, layered coverage

- **Data types.** Dense (`mat::dense2D`, configurable orientation, fixed or
  dynamic size), sparse CRS (`compressed2D`), COO (`coordinate2D`), ELLPACK
  (`ell_matrix`), structured dense (identity, block-diagonal, permutation), plus
  N-D `array::ndarray` (NumPy-style) and a compile-time `tensor::tensor`. Rich
  view and expression-template layers (`mat/view/`, `mat/expr/`, `vec/expr/`).
- **Dense operations** (`operation/`). LU, QR/LQ, Cholesky, LDLᵀ (incl.
  Bunch–Kaufman for indefinite), Hessenberg; **eigenvalues + eigenvectors** for
  general (Francis double-shift QR) and symmetric matrices; SVD (Jacobi / QR);
  triangular solves, inverse; the full norm family; matvec/matmul; a complete
  BLAS-1 and transcendental/element-wise set; Kronecker product; mixed-precision
  `convert`.
- **Sparse direct solvers** (`sparse/`). Orderings (RCM, AMD, COLAMD, minimum
  degree, Dulmage–Mendelsohn); symbolic analysis (elimination/column etree,
  postorder, supernode detection); factorizations (sparse LU, Cholesky, LDLᵀ,
  **supernodal LU/LDLᵀ**, QR, native KLU/Gilbert–Peierls); mixed-precision
  iterative refinement; sparse shift-invert eigensolver.
- **Iterative solvers** (`itl/`). Ten Krylov methods (CG, CGS, BiCG, BiCGSTAB,
  BiCGSTAB(ℓ), GMRES, MINRES, QMR, TFQMR, IDR(s)); preconditioners (Jacobi,
  block-Jacobi, IC(0), ILU(0), ILUT, ILDL, SSOR); smoothers; geometric
  multigrid; matrix-free Krylov eigensolvers (power iteration, Lanczos, Arnoldi).
- **Mixed precision as a first-class concern.** `math/accumulator_traits` exposes
  three independent precisions (element / accumulator / result) threaded through
  dot, norms, GEMM, and LU columns — the library's headline differentiator versus
  fixed-precision external BLAS.
- **Number-type genericity.** Structural support for foreign field types (posit,
  cfloat, LNS) via ADL `abs()`/`sqrt()`, with no hard dependency on any external
  number library.
- **Tooling.** Matrix Market + element I/O; ~21 test-matrix generators; a mature
  benchmark harness; heavy test coverage across all subsystems.

### Gaps (functional)

- **Sparse formats:** CRS is the only first-class sparse *matrix* (CSC exists only
  as a conversion utility); no BSR/block-sparse, DIA, or hybrid formats.
- **Eigen:** no generalized problem `A x = λ B x`; no public Schur decomposition;
  symmetric path is QR-iteration only (no divide-and-conquer / MRRR); the Krylov
  eigensolvers have **no implicit restart** (no IRAM/IRLM/Krylov–Schur).
- **Matrix functions:** no `expm`/`logm`/`sqrtm`, polar decomposition, or
  pseudo-inverse; no condition-number estimator API.
- **Preconditioners:** no algebraic multigrid (AMG), no sparse approximate inverse
  (SPAI); the multigrid is geometric-style.

**Verdict:** functionally this is a *mature single-node* library — it covers the
common dense/sparse/iterative workload well. The gaps are advanced/specialized
rather than foundational.

---

## 2. Performance

### What exists and works well

- **SIMD abstraction** (`simd/batch.hpp`). A compile-time-lane `batch<T>` over
  Google Highway (x86 SSE…AVX-512, NEON) with a dependency-free scalar fallback.
  Vectorized L1 kernels (`simd/algorithm.hpp`: dot, sum-of-squares, axpy, scal,
  each with 4 accumulators to hide FMA latency) and an orientation-aware SIMD
  GEMV (`detail/gemv.hpp`). Includes a **widening load** (`load_widen`) for
  mixed-precision.
- **Blocked GEMM** (`detail/gemm_blocked.hpp` + `gemm_microkernel.hpp` +
  `gemm_pack.hpp`). A textbook BLIS/GotoBLAS 5-loop cache-blocking nest with
  packing and a register micro-kernel, templated `<TC accumulator, TAB operand>`
  so **narrow operands widen into wide accumulators** at SIMD speed. Gated by
  `MTL5_NATIVE_FAST_GEMM`.
- **Compile-time backend dispatch** (`interface/dispatch_traits.hpp`). Each
  operation uses an `if constexpr` ladder **BLAS → native-fast → generic**,
  keyed on `BlasDenseMatrix`/`BlasDenseVector` concepts (float/double + contiguous
  `data()`) and an accumulator-policy predicate. External BLAS/LAPACK and
  SuiteSparse/SuperLU plug in through uniform `MTL5_WITH_*` toggles.
- **Measured single-thread result** (i7-12700K, 1 P-core, fp64; `benchmarks/`):
  native-fast GEMM ≈ **80–84 % of OpenBLAS** for N ≥ 256 (~76–78 % of FMA peak);
  GEMV ~100–116 %; dot ~88–110 %; nrm2 ~120–278 %. A respectable from-scratch
  kernel stack.
- **Benchmark maturity.** Four-backend comparison (native / native-fast /
  OpenBLAS / MKL), one executable per backend, size sweeps, P-core pinning,
  pass/fail gate scripts, and GFLOP/s-vs-N plots.

### Performance gaps / bottlenecks

1. **Threading is essentially absent.** The *only* multithreaded kernel is the
   native blocked GEMM, which partitions one loop across a raw `std::thread` team
   (`MTL5_NUM_THREADS`, **default 1**). No OpenMP, TBB, `std::execution`, or thread
   pool anywhere. GEMV, all L1 ops, **all dense factorizations, all sparse
   factorizations, and all iterative/Krylov/multigrid solvers are serial.**
2. **GEMM threading is immature:** threads are created/joined per outer iteration
   (no persistent pool), only one loop is parallelized, and parallel efficiency
   falls from ~92 % (4 cores) to ~73 % (8 cores) — native-fast reaches **5.8×** on
   8 P-cores vs OpenBLAS 7.15× / MKL 7.30×.
3. **Blocking parameters are hardcoded to a Haswell-class core** (`simd/blocking.hpp`);
   no runtime cache detection, so they are suboptimal on other microarchitectures.
4. **No SIMD on scalable-vector ISAs** — ARM SVE and RISC-V V fall back to scalar
   (Highway's scalable path has no constexpr lane count).
5. **Mixed-precision SIMD covers only float→double;** every other accumulator
   (Kahan, bf16, quire, …) runs the scalar generic kernel.
6. **Element-wise/transcendental ops and expression-template materialization are
   not vectorized** through `batch<T>` (plain scalar loops; `operation/lazy.hpp`).
7. **Native dense factorizations are slow at large N** — real large-scale
   performance still requires an external BLAS/LAPACK.
8. The **block-recursive `recursion/` infrastructure is unwired** for performance
   (the production GEMM uses flat cache blocking).

**Verdict:** strong *single-threaded* node performance with a genuine
from-scratch kernel; **on-node parallel scaling is the biggest near-term gap.**

---

## 3. Distributed-memory expansion

### Current state: none

A repo-wide search finds **no MPI, no communicators, no partitioned/distributed
matrix or vector, no halo/ghost exchange, no domain decomposition.** (The only
"decomposition"/"partition" hits are algorithmic — Dulmage–Mendelsohn/BTF and
cache-blocking comments.) MTL5 is single-address-space today.

### Readiness — genuinely favorable hooks

The iterative-solver layer is **nearly drop-in** for distribution because it is
written against a minimal, layout-agnostic contract:

- `concepts/linear_operator.hpp` requires only `{ a * x }`.
- The Krylov solvers (`itl/krylov/*.hpp`) touch nothing but `A * x`, `dot`, and
  `axpy`-style updates — e.g. CG uses `auto Ap = A * p;` and `mtl::dot(p, q)` and
  no element indexing.
- There is a **deliberately-placed seam**: `operation/resource.hpp` — *"Named
  boundary for future distributed-vector extension"* — so solvers size workspaces
  through an indirection rather than calling `.size()` directly.

A distributed matrix that overloads `operator*` (with internal halo exchange) and
a distributed `dot`/`nrm2` (with an internal `MPI_Allreduce`) would satisfy the
existing Krylov solvers **unchanged**.

### What an expansion must build

1. **Memory/partition model** — a distributed vector/matrix type with a row (or
   2-D block) partition and rank ownership. (Shared prerequisite with §4: the
   storage layer, `detail/contiguous_memory_block.hpp` and `compressed2D`, is
   hard-wired to host `std::vector`/`new[]` with no allocator/space parameter.)
2. **Communication primitives** — an MPI (or abstract transport) wrapper: reducing
   `dot`/`nrm2`, halo/ghost exchange for the distributed SpMV, and gather/scatter.
3. **Distributed preconditioners** — the current preconditioners (`itl/pc/`) are
   local; block-Jacobi/additive-Schwarz variants would be needed.
4. **Distributed sparse direct solvers** — a much larger effort (the serial
   factorizations do not decompose trivially); realistically deferred to wrapping
   an external distributed solver (e.g. MUMPS/SuperLU_DIST/PT-Scotch ordering).

### Effort estimate

- **Krylov + SpMV + reductions over MPI:** *moderate* — the abstractions are
  ready; the work is the distributed type, halo exchange, and collective `dot`.
  This alone unlocks distributed iterative solves.
- **Distributed direct/multigrid:** *large* — new algorithms or external-library
  wrapping.

**Verdict:** distributed *iterative* solving is the highest-leverage, lowest-risk
expansion — the solver contract was clearly designed with it in mind.

---

## 4. Hardware-accelerator expansion

### Current state: none (beyond CPU SIMD)

No CUDA/HIP/SYCL/OpenCL/oneAPI code exists (`.cu`, `cudaMalloc`, `__global__`,
cuBLAS/cuSOLVER all absent). "GPU" appears only as prose (a "how to add a CUDA
backend later" note in `benchmarks/README.md`) and MKL/oneAPI are referenced
purely as a **CPU** BLAS vendor. The `ell_matrix` (ELLPACK) format is a GPU-shaped
*hint* but has no device code behind it. The sole acceleration layer is CPU SIMD
via Highway, compiled **static single-ISA** — explicitly *not* the runtime-dispatch
mechanism you would repurpose for offload.

### Readiness — one strong hook, one hard tension

**The strong hook:** the `interface/` pattern is an ideal template for a vendor
GPU backend. Each external library is one macro-guarded header
(`blas.hpp`/`lapack.hpp`/`umfpack.hpp`/…) exposing thin
`mtl::interface::<lib>` wrappers, selected at compile time by `dispatch_traits`.
A `cublas.hpp` / `cusolver.hpp` with `MTL5_WITH_CUBLAS` → `MTL5_HAS_CUBLAS` would
be **mechanically identical** to add.

**The hard part — there is no notion of *where data lives*.** Current dispatch
keys only on scalar type and a host `data()` pointer. A GPU path additionally
needs:

1. A **memory-space abstraction** (device buffers, allocator/space parameter on
   `contiguous_memory_block`/`compressed2D`) — the same missing primitive as §3.
2. **Explicit H2D/D2H transfer management** — the existing external backends never
   need this because they share the host buffer; a GPU backend does.
3. A **device-aware dispatch key** so the `if constexpr` ladder can route
   host-vs-device operands correctly.

**The number-type tension.** MTL5's core value — arbitrary `Value` types — does
*not* come along for free onto a GPU:

- BLAS/LAPACK dispatch is already gated to float/double
  (`is_blas_scalar_v`), and any non-default accumulator forces the generic path.
  A cuBLAS/cuSOLVER backend inherits exactly this **float/double-only** restriction.
- Custom types (posit/LNS) would each need POD/`__device__`-compatible
  representations and bespoke device kernels. The realistic story splits cleanly:
  **float/double → vendor GPU BLAS via the `interface/` pattern; custom types →
  generic kernels that need individual device ports.**

**Tiling substrate.** The block-recursive `recursion/matrix_recursator.hpp`
(quad-tree subdivision) is conceptually the right tool to generate
GPU-kernel-sized tiles, but today it is a host-only traversal with no link to
memory spaces or backends.

### Effort estimate

- **float/double GPU BLAS/solver offload via `interface/`:** *moderate–large* —
  the dispatch pattern exists, but the memory-space abstraction + transfer
  management are net-new and cross-cutting.
- **Custom-type device kernels:** *large and open-ended* — outside what vendor
  libraries provide.

**Verdict:** a float/double GPU acceleration backend is architecturally feasible
and follows an existing pattern; the blocker is the shared memory-space
abstraction, and the custom-type mission does not transfer to GPUs cheaply.

---

## 5. The common prerequisite: a memory-space abstraction

Both expansion directions are gated by the same missing primitive. Today:

- `detail/contiguous_memory_block.hpp` hard-wires host allocation (`new[]` via
  `aligned_allocator`) with a 3-state ownership enum (`own`/`external`/`view`) —
  but "external/view" only ever wraps another **host** pointer. There is no
  allocator or memory-space template parameter.
- `compressed2D` is three raw `std::vector`s with no space abstraction at all.

Introducing a **memory-space / allocator policy** on these storage primitives (and
a matching tag the dispatch traits can key on) is the enabling refactor for
distributed partitions (host memory owned per rank) *and* device buffers (GPU
memory). It is the single highest-leverage architectural investment for either
roadmap.

---

## 6. Prioritized recommendations

Ordered by leverage-to-effort:

1. **On-node threading (highest immediate value).** Introduce a thread-pool
   abstraction and parallelize beyond GEMM: a persistent pool for the blocked
   GEMM (removes per-iteration spawn), plus threaded GEMV, L1 ops, and — where the
   dependency structure allows — sparse triangular solves and Krylov SpMV. This
   directly addresses the biggest measured gap and benefits every subsystem.
2. **Runtime cache/ISA detection** for the GEMM blocking parameters and a scalable
   SVE/RVV SIMD path — unlocks the existing kernels on non-Haswell and ARM/RISC-V
   hardware.
3. **Memory-space abstraction** in the storage layer — the shared enabler for
   both distributed and GPU work; do this before either backend.
4. **Distributed iterative solving** — a distributed vector/matrix + MPI
   `dot`/halo SpMV; the Krylov layer is already contract-ready (`resource.hpp`
   seam). Highest-leverage distributed step.
5. **float/double GPU BLAS/solver backend** via a `cublas.hpp`/`cusolver.hpp`
   under the `interface/` pattern, once (3) lands.
6. **Functional depth** as demand dictates — implicit-restart Krylov eigensolvers
   (Krylov–Schur), generalized eigenproblem, AMG preconditioner, matrix functions.

### Strategic note

MTL5's differentiator is **mixed precision with arbitrary number types**, and that
mission is largely orthogonal to vendor GPU/BLAS acceleration (which is float/double
only). The most defensible expansion path keeps the generic, custom-type kernels as
the core, adds **on-node and distributed parallelism** (which *do* apply to custom
types), and treats float/double GPU offload as an opt-in `interface/` backend for
the subset of workloads that use standard precision.
