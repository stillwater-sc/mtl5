# Phase 14 Summary: External Library Bindings (BLAS, LAPACK, UMFPACK)

## Overview

Phase 14 completes the MTL4-to-MTL5 modernization by wiring the pre-existing BLAS/LAPACK
interface stubs (from Phase 9) into the operation layer, and adding UMFPACK sparse direct
solver support. All dispatch is compile-time gated via `#ifdef` / `if constexpr`, ensuring
zero overhead when external libraries are not enabled.

## What Was Done

### 14.1 Dispatch Infrastructure

- **`interface/dispatch_traits.hpp`** (new) -- Compile-time traits for dispatch decisions:
  - `is_blas_scalar_v<T>` -- true for `float`/`double`
  - `BlasDenseMatrix<M>` concept -- contiguous `data()` + BLAS scalar
  - `BlasDenseVector<V>` concept -- contiguous `data()` + BLAS scalar
  - `is_row_major_v<M>` -- orientation detection for transpose trick

### 14.2 BLAS Dispatch

- **`operation/mult.hpp`** -- `mult(A, x, y)` and `mult(A, B, C)` dispatch to
  `gemv`/`gemm` when all operands are `dense2D<float/double>` or `dense_vector<float/double>`.
  Row-major matrices use the transpose trick (`C = A*B <=> C_col = B_col * A_col`).
- **`operation/norms.hpp`** -- `two_norm(v)` dispatches to `nrm2` for BLAS-eligible vectors.

### 14.3 LAPACK Dispatch

Each operation file gained a `#ifdef MTL5_HAS_LAPACK` + `if constexpr` block that dispatches
to LAPACK for column-major `dense2D<float/double>`, falling back to the C++ implementation
for row-major or non-BLAS scalar types:

| File | LAPACK routine | Notes |
|------|----------------|-------|
| `operation/lu.hpp` | `getrf` | 1-based pivot conversion |
| `operation/qr.hpp` | `geqrf` | Workspace query pattern |
| `operation/cholesky.hpp` | `potrf` | Lower triangle, SPD check |
| `operation/svd.hpp` | `gesdd` | Divide-and-conquer, workspace query |
| `operation/eigenvalue_symmetric.hpp` | `syev` | Eigenvalues only, workspace query |

### 14.4 Additional LAPACK Wrappers

Added to `interface/lapack.hpp`:
- `getrs` -- solve after LU factorization
- `potrs` -- solve after Cholesky factorization
- `orgqr` -- generate Q from QR factorization
- `trtrs` -- triangular solve

### 14.5 UMFPACK Interface

- **CMake** -- `MTL5_ENABLE_UMFPACK` option with `find_path`/`find_library` detection
- **`interface/umfpack.hpp`** (new):
  - `crs_to_ccs()` -- CSR to CSC format conversion (UMFPACK requires CCS)
  - `umfpack_solver` -- RAII class managing symbolic/numeric factorization handles
  - `umfpack_solve()` -- convenience free function for one-shot sparse solves

### 14.6 Tests

Three new test files in `tests/unit/interface/`:
- `test_blas_dispatch.cpp` -- GEMV, GEMM, norms, dispatch trait static checks
- `test_lapack_dispatch.cpp` -- LU, QR, Cholesky, SVD, eigenvalue correctness
- `test_umfpack.cpp` -- Sparse direct solve (conditional on `MTL5_HAS_UMFPACK`)

All 76 tests pass (73 existing + 3 new).

## Files Changed

| File | Action |
|------|--------|
| `include/mtl/interface/dispatch_traits.hpp` | Created |
| `include/mtl/interface/blas.hpp` | Unchanged |
| `include/mtl/interface/lapack.hpp` | Extended with getrs, potrs, orgqr, trtrs |
| `include/mtl/interface/umfpack.hpp` | Created |
| `include/mtl/operation/mult.hpp` | BLAS dispatch added |
| `include/mtl/operation/norms.hpp` | BLAS dispatch added (two_norm) |
| `include/mtl/operation/lu.hpp` | LAPACK dispatch added |
| `include/mtl/operation/qr.hpp` | LAPACK dispatch added |
| `include/mtl/operation/cholesky.hpp` | LAPACK dispatch added |
| `include/mtl/operation/svd.hpp` | LAPACK dispatch added |
| `include/mtl/operation/eigenvalue_symmetric.hpp` | LAPACK dispatch added |
| `include/mtl/mtl.hpp` | Added dispatch_traits.hpp and umfpack.hpp includes |
| `CMakeLists.txt` | Added MTL5_ENABLE_UMFPACK option and detection |
| `tools/cmake/summary.cmake` | Added UMFPACK to config summary |
| `CLAUDE.md` | Updated interface description and CMake options |
| `tests/unit/CMakeLists.txt` | Added interface test glob |
| `tests/unit/interface/test_blas_dispatch.cpp` | Created |
| `tests/unit/interface/test_lapack_dispatch.cpp` | Created |
| `tests/unit/interface/test_umfpack.cpp` | Created |
| `docs/modernization/phase14-plan.md` | Created (implementation plan) |
| `docs/modernization/phase14-summary.md` | Created (this file) |

## Design Decisions

1. **Fortran BLAS over CBLAS** -- Maximum portability; CBLAS availability varies.
   Row-major handled via transpose trick.
2. **Column-major only for LAPACK** -- Row-major matrices fall back to C++ rather than
   introducing error-prone transpose conversions for in-place factorizations.
3. **`if constexpr` + concepts** -- Zero-overhead dispatch; no runtime checks.
4. **UMFPACK via RAII** -- `umfpack_solver` manages factorization handles with proper cleanup.
5. **Complex deferred** -- `std::complex<float/double>` support left for future work.
