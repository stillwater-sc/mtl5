# MTL5 — Matrix Template Library 5

C++20 header-only linear algebra library for mixed-precision algorithm design and optimization.

MTL5 is a modernized successor to [MTL4](https://github.com/stillwater-sc/mtl4), preserving the proven architecture for custom number types (posits, LNS, etc.) while leveraging C++20 features for cleaner, safer, and more maintainable code. Its distinguishing feature is a **mixed-precision accumulator model** — store narrow, accumulate wide, round out once — carried through the dense kernels, iterative solvers, and native sparse factorizations.

## Highlights

- **Header-only** — no build step; drop `include/` on your path and `#include <mtl/mtl.hpp>`
- **Zero Boost dependency** — pure C++20 (concepts, `constexpr`, `std::span`, ranges)
- **Mixed precision throughout** — a shared `accumulator_traits` policy expresses the three precisions of a kernel (element, accumulate, result); e.g. `mult<float>(A_bf16, B_bf16, C_bf16)` accumulates in fp32 and stores bf16 once
- **Custom arithmetic types** — designed for posits, LNS, and other Universal number types, with quire-based exact accumulation
- **Dense & sparse** — CSR/CSC, COO, ELLPACK, block-diagonal, plus dense row/column-major
- **Complete BLAS surface** — L1/L2/L3, generic over any type, auto-dispatching to external BLAS/LAPACK for dense float/double
- **Sparse direct solvers** — native Cholesky/LDLᵀ/LU/QR, supernodal LU/LDLᵀ, KLU, fill-reducing orderings, plus wrappers for SuiteSparse/SuperLU
- **Iterative solvers** — a full Krylov suite, preconditioners, smoothers, and multigrid
- **Eigen & SVD** — dense (symmetric + general with eigenvectors), matrix-free iterative (Lanczos/Arnoldi/power), and sparse shift-invert
- **On-node threading** — a dependency-free thread pool; kernels are bit-identical across thread counts
- **Optional SIMD** — Google Highway-backed batch kernels with narrow→wide widening on load

## Requirements

- CMake 3.22+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+, Apple Clang)
- Catch2 v3 (fetched automatically for tests)

Portable across Linux (x64/ARM64), macOS (ARM64), and Windows (MSVC / Clang-CL).

## Quick Start

```bash
# Clone
git clone https://github.com/stillwater-sc/mtl5.git
cd mtl5

# Build (uses dev preset: Debug, tests ON)
cmake --preset dev
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build

# Run an example
./build/examples/phase01_core_types/...
```

Because MTL5 is header-only, using it in your own project needs no build of MTL5 itself:

```cpp
#include <mtl/mtl.hpp>

int main() {
    mtl::mat::dense2D<double> A(3, 3);
    mtl::vec::dense_vector<double> x(3), b(3);
    // ... fill A, b ...
    mtl::lu_apply(A, x, b);   // factor A and solve A*x = b (A modified in place)
}
```

## Feature Overview

### Core types

- **Dense** — `dense2D` (row- or column-major), `dense_vector`, `strided_vector_ref`, `unit_vector`
- **Sparse matrices** — `compressed2D` (CSR/CSC), `coordinate2D` (COO), `ell_matrix` (ELLPACK), `block_diagonal2D`, `identity2D`, `permutation_matrix`, plus `sparse_vector`
- **Expression templates & views** — lazy element-wise expressions (CRTP), transposed / sub-matrix / row / column views
- **Mathematical tensors** — `tensor<T, Rank, Dim>` (stack-allocated, compile-time rank/dimension), symmetric/antisymmetric tensors, metric and index helpers
- **NumPy-style arrays** — `array::ndarray<T, N, Order>`: static rank, runtime shape/strides, owning or view, C- or F-order, slicing, broadcasting, and interop

### Operations (`mtl::operation`)

- **BLAS Level 1** — `dot`, `dot_real`, `axpy`, `scal`, and the norm family (`two_norm`/`nrm2`, `one_norm`, `inf_norm`, `frobenius_norm`)
- **BLAS Level 2** — `gemv`, `ger` (rank-1 update), `symv`, `trmv`, `trsv`
- **BLAS Level 3** — `gemm`/`mult`, `trmm`, `trsm`, `symm`, `syrk`, `syr2k`
- **Factorizations** — LU, QR, LQ, Cholesky, LDLᵀ (and Bunch–Kaufman `ldlt_bk`), SVD, Hessenberg reduction, Householder & Givens
- **Eigenvalues** — symmetric solver and general (Francis implicit double-shift QR); `eigen` returns eigenvalues **and** eigenvectors
- **Solvers & inverses** — triangular solves, `inv`, `sparse_solve`
- **Structure & utility** — `trace`, `diagonal`, `kron` (Kronecker product), `projection`, `reorder`, `trans`, `conj`, `real`/`imag`, `product`, `sum`, `min`/`max`, `fill`, `random`
- **Transcendentals** — the full element-wise set: trig, inverse trig, hyperbolic, `exp`/`log`/`pow`/`sqrt`/`cbrt`, `erf`/`erfc`, rounding
- **Property predicates** — `is_symmetric`/`is_hermitian`, `is_spd`/`is_positive_definite`, `is_singular`/`is_invertible`, `determinant`, `condition_number`/`rcond`, `numerical_rank`/`nullity`, `spectral_radius`, `inertia`, `is_orthogonal`/`is_unitary`/`is_normal`, and many structural/vector checks

### Mixed precision

- **`math::accumulator_traits<Acc, Value>`** — a cross-cutting policy expressing element (storage), accumulator (compute), and result (serialize) precisions; the accumulate→output conversion is fused into the final store
- **`math::quire_accumulator`** — exact posit accumulation via the quire
- **`convert`** — standalone element-wise re-quantization (distinct from the fused epilogue)
- Applied across `dot`, `gemm`/`mult`, `gemv`, and the sum-of-squares norms; default (`Accumulator = void`) is byte-identical to the plain path

### SIMD (`mtl::simd`, optional)

Google Highway-backed `batch`, `algorithm`, and `blocking` layers with **widening-on-load** (`load_widen`): the micro-kernels promote narrow operands into wide accumulator registers, giving large speedups for mixed-precision GEMM and dot. Enabled with `-DMTL5_WITH_HIGHWAY=ON`.

### On-node threading

A persistent `detail::thread_pool` built on the C++ standard concurrency runtime (no OpenMP/TBB) with `parallel_for` and `parallel_reduce`. Threaded kernels — blocked GEMM, GEMV, `axpy`/`scal`, `dot`/`nrm2`, and sparse SpMV — are **bit-identical across thread counts** (reductions are deterministic per thread count). Threading is **off by default**; `MTL5_NUM_THREADS` sizes the pool. Iterative and eigen solvers inherit the SpMV/L1 threading with no solver-code changes.

### Iterative solvers (`mtl::itl`)

- **Krylov** — `cg`, `bicg`, `bicgstab`, `bicgstab_ell`, `cgs`, `gmres`, `idr_s`, `minres`, `qmr`, `tfqmr`
- **Preconditioners** (`itl::pc`) — `identity`, `diagonal`, `block_diagonal`, `ic_0`, `ildl`, `ilu_0`, `ilut`, `ssor`
- **Smoothers** (`itl::smoother`) — `jacobi`, `gauss_seidel`, `sor`
- **Multigrid** (`itl::mg`) — geometric multigrid with prolongation/restriction
- **Iterative eigensolvers** (`itl::eigen`) — `power_iteration`, `lanczos`, `arnoldi`, matrix-free through the `LinearOperator` concept

### Sparse direct solvers (`mtl::sparse`)

- **Fill-reducing orderings** — RCM, AMD, COLAMD, minimum-degree, Dulmage–Mendelsohn
- **Symbolic analysis** — elimination tree (O(nnz)), column elimination tree, postorder, supernode partitioning
- **Numeric factorization** — `sparse_cholesky` (LLᵀ), `sparse_ldlt`, `sparse_lu` (threshold partial pivoting), `sparse_qr` (least squares), native **supernodal LU/LDLᵀ**, and native **KLU**; all generic over the mixed-precision accumulator
- **Refactorization** — reuse a prior symbolic structure + pivot sequence to recompute same-pattern matrices ~2–3× faster (the SPICE-transient path)
- **Iterative refinement** — Universal-free, templated residual precision, scaled variant for narrow-exponent low-precision factors
- **Sparse eigen** — largest-magnitude Arnoldi and shift-invert for eigenpairs near a target

### External library interfaces (`mtl::interface`, optional)

Auto-dispatching bindings that engage when the type qualifies (dense column-major float/double) and the library is present, otherwise falling back to the in-house path: **BLAS** (L1/L2/L3), **LAPACK** (factorizations, `syev`/`geev` eigensolvers), **UMFPACK**, **SuperLU**, **KLU**, **CHOLMOD**, **SPQR**. Any non-default accumulator forces the native kernel, since external BLAS cannot honor a custom accumulator.

### I/O (`mtl::io`)

- **Matrix Market** reader/writer, with transparent gzip (`.mtx.gz`) when built with zlib
- **`.el` edge-list** read/write
- **From-first-principles PNG writer** and **`spy`** sparsity-pattern visualization

### Test matrix generators (`mtl::generators`)

A catalog of classic and random test matrices: `hilbert`, `frank`, `wilkinson`, `clement`, `companion`, `forsythe`, `kahan`, `lehmer`, `lotkin`, `minij`, `moler`, `pascal`, `rosser`, `vandermonde`, `laplacian`, `poisson`, `ones`, and randomized `randorth`, `randspd`, `randsym`, `randsvd`.

## Build Options

| Option | Default | Description |
|---|---|---|
| `MTL5_BUILD_TESTS` | ON | Build the Catch2 test suite |
| `MTL5_BUILD_EXAMPLES` | ON | Build example programs |
| `MTL5_BUILD_BENCHMARKS` | OFF | Build the benchmark suite |
| `MTL5_BUILD_REGRESSION_TESTS` | OFF | Build large-scale regression tests (slow) |
| `MTL5_WITH_BLAS` | OFF | Link BLAS for dense L1/L2/L3 acceleration |
| `MTL5_WITH_LAPACK` | OFF | Link LAPACK for factorizations & eigensolvers |
| `MTL5_WITH_HIGHWAY` | OFF | Use Google Highway for SIMD-accelerated kernels |
| `MTL5_NATIVE_ARCH` | OFF | Tune in-tree builds for the host CPU (`-march=native`) |
| `MTL5_NATIVE_FAST_GEMM` | OFF | Route `mtl::mult` through the native blocked GEMM / SIMD GEMV path |
| `MTL5_WITH_ZLIB` | OFF | Link zlib for transparent gzip (`.mtx.gz`) Matrix Market reading |
| `MTL5_WITH_UMFPACK` | OFF | Link UMFPACK (SuiteSparse) |
| `MTL5_WITH_SUPERLU` | OFF | Link SuperLU |
| `MTL5_WITH_SUITESPARSE_KLU` | OFF | Link KLU (SuiteSparse) |
| `MTL5_WITH_SUITESPARSE_CHOLMOD` | OFF | Link CHOLMOD (SuiteSparse) |
| `MTL5_WITH_SUITESPARSE_SPQR` | OFF | Link SuiteSparseQR (SuiteSparse) |

Threading is a runtime setting, not a build option: set the `MTL5_NUM_THREADS` environment variable (unset or `1` runs the serial paths).

## Project Structure

```
include/mtl/
├── concepts/       # C++20 concepts (Scalar, Matrix, Vector, LinearOperator, ...)
├── tag/            # Compile-time tags (orientation, sparsity, shape, storage)
├── traits/         # Type traits and metafunctions
├── math/           # Algebraic identities + mixed-precision accumulator policy
├── detail/         # Internal: memory blocks, GEMM kernels/packing, thread pool
├── simd/           # Highway-backed batch/blocking SIMD layer
├── mat/            # Matrix types, expressions, views
├── vec/            # Vector types and expressions
├── tensor/         # Mathematical tensors (compile-time rank/dimension)
├── array/          # NumPy-style ndarray (slicing, broadcasting, interop)
├── operation/      # Free-function operations (BLAS, decompositions, eigen, predicates)
├── functor/        # Scalar and typed functors
├── recursion/      # Block-recursive infrastructure
├── generators/     # Test matrix generators
├── io/             # Matrix Market, edge-list, PNG, spy
├── itl/            # Iterative solvers, preconditioners, smoothers, multigrid, eigen
├── sparse/         # Sparse direct solvers: orderings, analysis, factorization
└── interface/      # Optional BLAS/LAPACK/UMFPACK/SuperLU/KLU/CHOLMOD/SPQR bindings
```

The `examples/` directory contains a phased tour (`phase01`–`phase15`) from core types through iterative solvers, sparse assembly, decompositions, eigen/SVD, expression templates, I/O, and sparse direct solvers, plus applied demos (e.g. an unscented Kalman filter).

## Key Boost-to-C++20 Replacements

| MTL4 (Boost) | MTL5 (C++20) |
|---|---|
| `boost::enable_if<is_matrix<T>>` | `requires Matrix<T>` |
| `boost::mpl::if_<cond, A, B>::type` | `std::conditional_t<cond, A, B>` |
| `boost::is_same<A,B>` | `std::is_same_v<A,B>` |
| `BOOST_STATIC_ASSERT` | `static_assert` |
| `boost::shared_ptr<T>` | `std::shared_ptr<T>` |

## Installation

```bash
cmake --preset release
cmake --build build-release
cmake --install build-release --prefix /usr/local
```

Then in your project:
```cmake
find_package(MTL5 REQUIRED)
target_link_libraries(myapp PRIVATE MTL5::mtl5)
```

## Documentation

Full documentation — architecture, algorithm write-ups (mixed-precision kernels, on-node threading, eigenvalues, measuring solver accuracy), and a Doxygen C++ API reference — is published from the `docs/` tree to the project's GitHub Pages site.

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

MTL5 builds on the foundational work of MTL4 by Peter Gottschling and the Simunova team.
