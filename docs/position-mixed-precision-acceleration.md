# MTL5 + Universal: A Platform for Accelerated Mixed-Precision Linear Algebra

**Position Paper — Stillwater Supercomputing, Inc.**

## Executive Summary

The convergence of autonomous systems, edge AI, and energy-constrained computing demands a fundamental rethinking of how linear algebra is computed. Today's numerical libraries are locked to IEEE 754 floating-point and general-purpose CPUs. SuiteSparse and SuperLU represent the state of the art for sparse direct methods — clean, well-packaged, and battle-tested — but they cannot target custom hardware accelerators, and they cannot exploit application-specific number systems that trade unnecessary precision for speed and energy.

MTL5 and Universal together form a platform that decouples *what* is computed (linear algebra) from *how* it is computed (arithmetic and hardware). MTL5 provides the algebraic structure — vectors, matrices, decompositions, solvers — parameterized over arbitrary value types. Universal provides the arithmetic — thousands of number systems, from posits and logarithmic numbers to custom floats, all defined at the bit level and ready for hardware synthesis. Together, they enable **mixed-precision linear algebra on custom hardware accelerators**, a capability that no existing library provides.

This document describes the vision, the current state of the platform, and the engineering roadmap to realize it.

---

## 1. The Problem: Why IEEE 754 on CPUs Is Not Enough

### 1.1 Autonomous Systems Need Constrained Linear Algebra

Autonomous vehicles, robots, and drones continuously solve linear systems to navigate their world. Kalman filters, model-predictive control, SLAM, and physics-based simulation all reduce to linear algebra at their core. These systems operate under hard constraints:

- **Latency**: a self-driving car processing LiDAR at 10 Hz has 100 ms per frame for perception, prediction, and planning. Every microsecond in the linear algebra kernel matters.
- **Power**: a delivery drone carries its power source. Every watt spent on computation is a watt not spent on flight. A 10x reduction in solver energy directly extends range.
- **Size**: edge devices cannot accommodate server-class GPUs. The compute must fit in embedded form factors with strict thermal budgets.

IEEE 754 double-precision (64-bit) provides far more precision than these applications need. A robot localizing to centimeter accuracy does not need 15 decimal digits. But the entire software stack — BLAS, LAPACK, SuiteSparse, SuperLU — is built around `float` and `double`. There is no clean path to exploit reduced or mixed precision, let alone custom number systems optimized for specific workloads.

### 1.2 SuiteSparse and SuperLU: Powerful but Inflexible

SuiteSparse (UMFPACK, CHOLMOD, KLU, SPQR) and SuperLU represent decades of algorithmic refinement in sparse direct methods. They are indispensable for CPU-based computation, and MTL5 provides clean interfaces to all of them (Phase 6 of the sparse direct solver design).

However, these libraries share fundamental limitations:

| Limitation | Consequence |
|-----------|-------------|
| Hardcoded to `float`/`double` | Cannot use posits, LNS, or custom types |
| C/Fortran implementations | Cannot be synthesized to hardware |
| CPU-only execution | No path to FPGA or ASIC acceleration |
| Fixed precision throughout | Cannot mix precisions across algorithm phases |
| IEEE 754 semantics | Non-reproducible in parallel execution |

These are not bugs — they are architectural constraints. SuiteSparse was designed to be the best possible CPU implementation, and it succeeds. But the next generation of computing requires something different.

### 1.3 The Mixed-Precision Opportunity

Mixed-precision computing exploits the observation that different parts of an algorithm have different precision requirements:

- **Sparse triangular solve**: often tolerates 16-bit arithmetic
- **Pivot selection**: requires full-precision comparison
- **Iterative refinement**: converges in a few steps from low-precision factorization to high-precision solution
- **Matrix-vector products**: dominate runtime and benefit most from reduced precision
- **Accumulation**: needs higher precision to avoid catastrophic cancellation

Research demonstrates that mixed-precision iterative refinement can achieve double-precision accuracy using single-precision (or lower) factorization, with 2-4x speedup on CPUs and 10x+ on accelerators. The potential grows with custom number systems:

- **Posits** provide tapered precision — high near 1.0 where most computation happens, trading range at the extremes. A 32-bit posit outperforms 32-bit IEEE float on many benchmarks.
- **Logarithmic Number Systems (LNS)** replace multiplication with addition, yielding simpler hardware and lower power for DSP-heavy workloads.
- **Custom floats** (e.g., 12-bit with 4-bit exponent) can be tuned per application domain — more exponent bits for dynamics simulation, more mantissa bits for geometry.

The obstacle is not the algorithms — it is the infrastructure. No existing linear algebra library makes it easy to explore, validate, and deploy mixed-precision algorithms across arbitrary number systems.

---

## 2. The Solution: MTL5 + Universal

### 2.1 Architecture

The platform has three layers:

```
┌──────────────────────────────────────────────────────────┐
│                    Application Layer                      │
│   Autonomous navigation, FEM, circuit simulation, ML      │
├──────────────────────────────────────────────────────────┤
│                  MTL5 — Algebra Layer                      │
│   Vectors, matrices, decompositions, solvers              │
│   C++20 concepts: Scalar, Field, Matrix, Vector           │
│   Expression templates, sparse/dense, iterative/direct    │
│   Parameterized over Value type — any arithmetic works    │
├──────────────────────────────────────────────────────────┤
│                Universal — Arithmetic Layer                │
│   posit<N,ES>  cfloat<N,E>  lns<N>  fixpnt<N,M>          │
│   Bit-exact, constexpr, hardware-synthesizable            │
│   std::numeric_limits specializations                     │
│   ADL-compatible abs(), sqrt(), math functions            │
├──────────────────────────────────────────────────────────┤
│                  Hardware Layer                            │
│   CPU (reference)  │  FPGA  │  ASIC  │  KPU              │
│   SuiteSparse/BLAS │  HLS   │  RTL   │  Custom ALU/SFU   │
│   (IEEE 754 only)  │  (any) │  (any) │  (any)            │
└──────────────────────────────────────────────────────────┘
```

**Key property**: the algebra layer (MTL5) and arithmetic layer (Universal) are connected only through C++20 concepts. Changing the number system changes a single template parameter, not the algorithm. Changing the hardware target changes the dispatch layer, not the mathematics.

### 2.2 How MTL5 Enables Type-Generic Linear Algebra

MTL5's C++20 concept hierarchy defines *what* a number must support, not *what* it is:

```cpp
concept Scalar = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { T{0} };
};

concept Field = Scalar<T> && requires(T a, T b) {
    { a / b } -> std::convertible_to<T>;
};

concept OrderedField = Field<T> && std::totally_ordered<T>;
```

Every MTL5 algorithm — from dense matrix-vector multiply to sparse Cholesky factorization — is parameterized over the value type and constrained only by these concepts. A `posit<32,2>` satisfies `OrderedField`. A `lns<16>` satisfies `Field`. They work in MTL5 without modification.

**Concrete example — same algorithm, different arithmetic:**

```cpp
#include <mtl/mtl.hpp>
#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

using posit32 = sw::universal::posit<32, 2>;
using float16 = sw::universal::cfloat<16, 5>;

// Sparse Cholesky with posit arithmetic
mtl::mat::compressed2D<posit32> A_posit(n, n);
// ... fill A ...
mtl::sparse::factorization::sparse_cholesky_solve(A_posit, x, b,
    mtl::sparse::ordering::amd{});

// Same algorithm with 16-bit custom float
mtl::mat::compressed2D<float16> A_fp16(n, n);
mtl::sparse::factorization::sparse_cholesky_solve(A_fp16, x16, b16,
    mtl::sparse::ordering::amd{});
```

One line changes. The algorithm, data structures, ordering, and solver infrastructure are identical.

### 2.3 How Universal Enables Hardware-Ready Arithmetic

Universal number types are designed for hardware synthesis:

- **Static bit-width**: `posit<32,2>` is always exactly 32 bits, with known encoding
- **Bit-exact semantics**: operations produce identical results on any platform
- **No special cases**: posits have no NaN, no denormals, no ±0 ambiguity — simpler hardware
- **Constexpr constructors**: `T{0}` and `T{1}` work at compile time
- **Hardware verification**: the `universal-hw` repository provides QA suites for ALU and SFU designs, validating hardware against software reference

The path from software to hardware:

1. **Explore** in software: use Universal + MTL5 to test algorithms across number systems
2. **Profile** precision requirements: identify which phases need which precision
3. **Validate** hardware designs: test FPGA/ASIC implementations against Universal's bit-exact reference
4. **Deploy**: MTL5 dispatch layer routes operations to CPU or accelerator based on type

### 2.4 The Dispatch Architecture

MTL5 already implements a compile-time dispatch pattern for BLAS/LAPACK:

```cpp
template <Matrix M, Vector V>
auto mult(const M& A, const V& x) {
    if constexpr (interface::BlasDenseMatrix<M> && interface::BlasDenseVector<V>) {
        // BLAS dgemv for float/double
        return blas::gemv(A, x);
    } else {
        // Generic implementation for any type
        return detail::mult_generic(A, x);
    }
}
```

This pattern extends naturally to hardware accelerators:

```cpp
if constexpr (is_blas_scalar_v<Value>) {
    // CPU BLAS path (float/double)
} else if constexpr (is_acceleratable_v<Value>) {
    // Custom accelerator path (posit, lns, cfloat)
    accelerator::dispatch(A, x, result);
} else {
    // Software reference path (any type)
    detail::mult_generic(A, x);
}
```

The type system determines the hardware target at compile time. No runtime overhead. No type erasure. No virtual dispatch.

---

## 3. The Role of SuiteSparse in the MTL5 Ecosystem

SuiteSparse is not being replaced — it is being complemented. MTL5's relationship to SuiteSparse is:

### 3.1 SuiteSparse as Production CPU Backend

For `float`/`double` on CPUs, SuiteSparse remains the optimal choice. MTL5 provides RAII interfaces to:

| Library | MTL5 Interface | Purpose |
|---------|---------------|---------|
| UMFPACK | `umfpack_solver` | Multifrontal unsymmetric LU |
| CHOLMOD | `cholmod_solver` | Supernodal Cholesky |
| KLU | `klu_solver` | Circuit simulation LU |
| SPQR | `spqr_solver` | Multifrontal sparse QR |
| SuperLU | `superlu_solver` | Supernodal left-looking LU |

These are the right tools when the problem is: "solve a large sparse system as fast as possible on a CPU in double precision."

### 3.2 Native Solvers as Algorithm Templates

MTL5's native sparse solvers (Phases 2-5) serve a different purpose: they are **type-generic algorithm templates** that work with *any* number system. They implement the same algorithms as SuiteSparse (Gilbert-Peierls, elimination trees, fill-reducing orderings) but in C++20 template form:

| Native Solver | Algorithm | SuiteSparse Equivalent |
|--------------|-----------|----------------------|
| `sparse_cholesky` | Up-looking LL^T | CHOLMOD |
| `sparse_lu` | Left-looking PA=LU | UMFPACK/KLU |
| `sparse_qr` | Householder QR | SPQR |
| `amd` | Approximate min degree | AMD |
| `colamd` | Column AMD | COLAMD |

For `double` on CPUs, SuiteSparse will be faster (supernodal BLAS, decades of tuning). But for `posit<32,2>` on an FPGA, the native solvers are the only option — and they work out of the box.

### 3.3 Unified Dispatch (Phase 7)

The planned unified dispatch layer (design document Phase 7) will automatically select the best backend:

```
sparse_cholesky_factor(A)
  ├── double + MTL5_HAS_CHOLMOD  → CHOLMOD (production CPU)
  ├── double                      → native sparse Cholesky (portable CPU)
  ├── posit<32,2>                 → native sparse Cholesky (software reference)
  ├── posit<32,2> + HAS_KPU      → KPU accelerated Cholesky (future)
  └── lns<16> + HAS_FPGA         → FPGA accelerated Cholesky (future)
```

The user writes one line of code. The compiler selects the optimal implementation.

---

## 4. Mixed-Precision Solver Strategies

### 4.1 Iterative Refinement

The most practical mixed-precision strategy for sparse direct solvers:

1. **Factor** in low precision: `sparse_lu<posit<16,1>>(A_low)`
2. **Solve** in low precision: `L_low \ (U_low \ (P * b))`
3. **Compute residual** in high precision: `r = b - A_high * x`
4. **Correct**: `x += L_low \ (U_low \ (P * r))`
5. **Repeat** until `||r|| < tol`

Typically converges in 2-3 iterations, achieving high-precision accuracy with low-precision factorization cost. The factorization dominates runtime (O(n^{1.5} to n^2) for sparse), while refinement is O(nnz) per iteration.

### 4.2 Precision-Layered Factorization

Different phases of factorization have different precision needs:

| Phase | Precision Need | Opportunity |
|-------|---------------|-------------|
| Fill-reducing ordering | Integer only | Already optimal |
| Symbolic analysis | Integer only | Already optimal |
| Numeric factorization | Medium | 16-32 bit custom type |
| Pivot selection | High (comparisons) | Full precision for stability |
| Triangular solve | Low-medium | 16-bit sufficient |
| Residual computation | High | Extended accumulation |

MTL5's symbolic/numeric phase separation already enables different precisions per phase. The `cholesky_symbolic` result is reusable across precisions — compute it once, factorize in multiple precisions for comparison.

### 4.3 Preconditioner Cascades

Use a low-precision direct factorization as a preconditioner for a high-precision iterative solver:

```cpp
// Low-precision incomplete Cholesky as preconditioner
auto L_low = sparse_cholesky_numeric<posit<16,1>>(A_low, sym);

// High-precision CG with low-precision preconditioner
itl::cg(A_high, x, b, direct_preconditioner(L_low), iter);
```

This combines the robustness of direct methods with the precision control of iterative methods, while exploiting reduced-precision hardware for the expensive preconditioner.

---

## 5. Roadmap: From Vision to Deployment

### 5.1 Current State (Completed)

| Component | Status |
|-----------|--------|
| MTL5 core: dense/sparse types, concepts, expression templates | Done |
| Sparse direct solvers: Cholesky, LU, QR | Done |
| Fill-reducing orderings: RCM, AMD, COLAMD | Done |
| SuiteSparse/SuperLU interfaces | Done |
| Universal: posit, cfloat, lns, fixpnt, integer types | Done |
| Universal: hardware verification suites (universal-hw) | Done |

### 5.2 Near-Term Engineering (Next Steps)

**Mixed-Precision Infrastructure**

- [ ] Add `mixed_precision_refine()` — iterative refinement with configurable low/high precision types
- [ ] Add precision-cast utilities: `mat::cast<TargetType>(source_matrix)` for lossless and lossy conversions between number systems
- [ ] Add `precision_profile()` diagnostic: given a matrix and solver, report per-phase precision requirements (condition estimates, growth factors)
- [ ] Integration tests: Universal posit/cfloat/lns types through full Cholesky/LU/QR solve pipelines
- [ ] Benchmark: compare `posit<32,2>` vs `float` vs `double` on standard sparse test matrices (UF Sparse Matrix Collection)

**Unified Dispatch (Phase 7)**

- [ ] Extend `if constexpr` dispatch in `operation/lu.hpp`, `qr.hpp`, `cholesky.hpp` to auto-select native vs SuiteSparse based on value type and library availability
- [ ] Add `is_hardware_acceleratable_v<T>` trait for future accelerator dispatch
- [ ] Define `AcceleratorBackend` concept for pluggable hardware targets

**Solver Robustness**

- [ ] Implement sparse LU with Gilbert-Peierls reach (Phase 3 improvement — currently left-looking with dense column scan)
- [ ] Add Dulmage-Mendelsohn / BTF decomposition for KLU-style block solvers
- [ ] Add incomplete factorization preconditioners (ILU/IC) that work with custom types

### 5.3 Medium-Term Research

**Hardware Acceleration Path**

- [ ] Define MTL5 accelerator dispatch API: `accelerator::cholesky_factor<posit<32,2>>(A)` with FPGA/ASIC backends
- [ ] Implement FPGA proof-of-concept: posit-based sparse matrix-vector multiply on Xilinx/Intel FPGA via HLS
- [ ] Validate against Universal software reference using bit-exact comparison
- [ ] Measure energy/performance vs CPU BLAS on representative workloads

**Mixed-Precision Algorithm Research**

- [ ] Implement GMRES-IR (GMRES-based iterative refinement) with three-precision scheme: factorize in low, solve GMRES in working, accumulate in high
- [ ] Characterize precision requirements of sparse direct solvers on autonomous systems workloads (Kalman filters, MPC, SLAM)
- [ ] Develop auto-precision selection: given condition number and accuracy target, automatically choose the minimum-precision number system
- [ ] Publish comparative study: posit vs IEEE vs LNS for sparse linear systems arising in robotics and control

### 5.4 Long-Term Vision

- [ ] KPU integration: MTL5 dispatches sparse factorizations to Stillwater Knowledge Processing Unit with application-tailored arithmetic
- [ ] Multi-device heterogeneous solve: symbolic analysis on CPU, numeric factorization on FPGA, iterative refinement on CPU
- [ ] Reproducible parallel sparse solvers: exploit posit/valid determinism for lock-step parallel factorization without numerical drift
- [ ] Standard proposal: contribute mixed-precision dispatch patterns to C++ standardization (P1673 BLAS interface, future sparse extensions)

---

## 6. Why This Matters Now

Three trends make this work urgent:

1. **Autonomous systems are scaling**: millions of vehicles, drones, and robots will need real-time linear algebra at the edge. Power and latency budgets rule out general-purpose GPU solutions.

2. **Custom silicon is democratizing**: FPGA costs have dropped 10x in a decade. RISC-V enables custom ISA extensions. Chiplet architectures allow mixing general-purpose and specialized cores. The hardware exists — what's missing is the software stack.

3. **IEEE 754 is showing its age**: non-reproducible parallel results, wasted bits on NaN/denormal encoding, no tapered precision, no quire accumulation. Posits and other modern formats fix these problems, but the numerical library ecosystem hasn't caught up.

MTL5 + Universal is the bridge. It preserves everything that works (SuiteSparse algorithms, BLAS performance, decades of numerical analysis) while opening the door to everything that's next (custom arithmetic, hardware acceleration, mixed-precision optimization, reproducible computation).

The sparse direct solver infrastructure implemented in Phases 1-6 — native Cholesky, LU, QR with AMD/COLAMD orderings plus SuiteSparse interfaces — provides the concrete foundation. Every algorithm works with any number system today, in software. Tomorrow, the same algorithms run on custom hardware. The template parameter changes. The mathematics doesn't.

---

## References

1. Davis, T.A. *Direct Methods for Sparse Linear Systems*. SIAM, 2006.
2. Greenbaum, A. *Iterative Methods for Solving Linear Systems*. SIAM, 1997.
3. Gustafson, J.L. *The End of Error: Unum Computing*. CRC Press, 2015.
4. Omtzigt, E.T.L., Gottschling, P., Seligman, M., Zorn, W. "Universal Numbers Library: Design and Implementation of a High-Performance Reproducible Number Systems Library." *arXiv:2012.11011*, 2020.
5. Higham, N.J., Pranesh, S. "Exploiting Lower Precision Arithmetic in Solving Symmetric Positive Definite Linear Systems and Least Squares Problems." *SIAM J. Sci. Comput.*, 43(1), 2021.
6. Carson, E., Higham, N.J. "Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions." *SIAM J. Sci. Comput.*, 40(2), 2018.
7. Davis, T.A., Natarajan, E.P. "Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems." *ACM TOMS*, 37(3), 2010.
8. Gustafson, J.L., Yonemoto, I.T. "Beating Floating Point at its Own Game: Posit Arithmetic." *Supercomputing Frontiers and Innovations*, 4(2), 2017.
9. Omtzigt, E.T.L., Kliuchnikov, V., Stillwater Supercomputing. "Universal: Reliable, Reproducible, and Energy-Efficient Numerics." *LNCS 13348*, Springer, 2022.
10. Demmel, J.W. et al. "A Supernodal Approach to Sparse Partial Pivoting." *SIAM J. Matrix Anal. Appl.*, 20(3), 1999.
