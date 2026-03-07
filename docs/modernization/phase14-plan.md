# Phase 14: External Library Bindings (BLAS, LAPACK, UMFPACK)

## Current State

### Already done (Phase 9)

- `include/mtl/interface/blas.hpp` -- Fortran extern "C" declarations + C++ wrappers in `mtl::interface::blas` (BLAS L1/L2/L3: dot, axpy, copy, scal, nrm2, gemv, trsv, gemm)
- `include/mtl/interface/lapack.hpp` -- Fortran extern "C" declarations + C++ wrappers in `mtl::interface::lapack` (potrf, getrf, geqrf, gesdd, syev)
- CMake options `MTL5_ENABLE_BLAS` / `MTL5_ENABLE_LAPACK` with `find_package`, compile definitions (`MTL5_HAS_BLAS`, `MTL5_HAS_LAPACK`), and link targets
- `mtl.hpp` already includes both interface headers (self-guarded with `#ifdef`)
- Pedagogical example `examples/phase9c_blas_dispatch.cpp`

### Not done -- the work of Phase 14

1. No dispatch logic in any operation file
2. No UMFPACK support at all
3. No tests for external library paths

---

## Phase 14.1: BLAS Dispatch in Operations

**Goal:** When `MTL5_HAS_BLAS` is defined and the operands are `dense2D<float/double>` or `dense_vector<float/double>`, dispatch to BLAS; otherwise fall back to the existing C++ implementation.

### 14.1a: Dispatch traits header

Create `include/mtl/interface/dispatch_traits.hpp`:

```cpp
#pragma once
#include <type_traits>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::interface {

template <typename T>
inline constexpr bool is_blas_scalar_v =
    std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename M>
concept BlasDenseMatrix = requires {
    requires is_blas_scalar_v<typename M::value_type>;
} && requires(const M& m) { m.data(); };

template <typename V>
concept BlasDenseVector = requires {
    requires is_blas_scalar_v<typename V::value_type>;
} && requires(const V& v) { v.data(); };

} // namespace mtl::interface
```

### 14.1b: `operation/mult.hpp` -- GEMM and GEMV dispatch

Strategy: Use `if constexpr` to detect at compile time whether the types qualify for BLAS dispatch.

- In `mult(A, x, y)`: if `BlasDenseMatrix<M> && BlasDenseVector<VIn> && BlasDenseVector<VOut>`, call `interface::blas::gemv`
- In `mult(A, B, C)`: if all three satisfy `BlasDenseMatrix`, call `interface::blas::gemm`
- **Row-major handling:** BLAS expects column-major (Fortran order). For row-major `dense2D`, use the identity `C = A*B <=> C^T = B^T * A^T` -- swap arguments and transpose flags. The existing example `phase9c_blas_dispatch.cpp` already demonstrates this pattern.

### 14.1c: `operation/norms.hpp` -- BLAS L1 dispatch (optional)

- Dispatch `one_norm` / `two_norm` / `infinity_norm` on dense vectors to `sasum_`/`dasum_`, `snrm2_`/`dnrm2_`, `isamax_`/`idamax_` when applicable.

---

## Phase 14.2: LAPACK Dispatch in Operations

**Goal:** When `MTL5_HAS_LAPACK` is defined and the matrix is `dense2D<float/double>`, dispatch factorizations to LAPACK.

Each operation file gets a `#ifdef MTL5_HAS_LAPACK` block with `if constexpr` selection:

| Operation file | LAPACK routine | Notes |
|---|---|---|
| `operation/lu.hpp` | `getrf` | Returns pivot vector; adapt to MTL5's LU return format |
| `operation/qr.hpp` | `geqrf` | Workspace query pattern (call with lwork=-1 first) |
| `operation/cholesky.hpp` | `potrf` | Upper/lower triangle selection via `uplo` param |
| `operation/svd.hpp` | `gesdd` | Divide-and-conquer SVD; workspace query needed |
| `operation/eigenvalue_symmetric.hpp` | `syev` | Workspace query needed |

### Key considerations

- **In-place modification:** LAPACK routines modify the input matrix in-place. Verify each operation's API contract matches.
- **Workspace queries:** Call the routine with `lwork = -1` to get optimal workspace size, then allocate with `std::vector<T>` and call again.
- **Column-major requirement:** Use the transpose trick for row-major matrices, or fall back to C++ for row-major and only dispatch column-major to LAPACK.
- **Error handling:** LAPACK returns `info` codes. Throw `std::runtime_error` on `info < 0` (illegal argument) and provide diagnostic on `info > 0` (singular matrix, etc.).

### Additional LAPACK wrappers to add to `lapack.hpp`

- `getrs` -- solve after LU factorization
- `potrs` -- solve after Cholesky factorization
- `orgqr`/`ungqr` -- generate Q from QR factorization
- `trtrs` -- triangular solve

---

## Phase 14.3: UMFPACK Interface (Optional)

**Goal:** Provide a sparse direct solver for `compressed2D` matrices using SuiteSparse/UMFPACK.

### 14.3a: CMake detection

Add to `CMakeLists.txt`:

```cmake
option(MTL5_ENABLE_UMFPACK "Enable UMFPACK sparse solver" OFF)

if(MTL5_ENABLE_UMFPACK)
    find_package(SuiteSparse COMPONENTS UMFPACK)
    # or manual find via find_path/find_library
    target_compile_definitions(mtl5 INTERFACE MTL5_HAS_UMFPACK)
    target_link_libraries(mtl5 INTERFACE ${UMFPACK_LIBRARIES})
    target_include_directories(mtl5 INTERFACE ${UMFPACK_INCLUDE_DIRS})
endif()
```

### 14.3b: Interface header

Create `include/mtl/interface/umfpack.hpp`:

- `extern "C"` declarations for UMFPACK routines (`umfpack_di_symbolic`, `umfpack_di_numeric`, `umfpack_di_solve`, `umfpack_di_free_symbolic`, `umfpack_di_free_numeric`)
- RAII wrapper class `umfpack_solver<T>` that manages symbolic/numeric factorization handles
- Simple free function: `umfpack_solve(const compressed2D<double>& A, dense_vector<double>& x, const dense_vector<double>& b)`

### 14.3c: Integration with `compressed2D`

- Verify `compressed2D` exposes CSC-compatible data (column pointers, row indices, values) -- UMFPACK expects compressed-column format
- May need a CSR-to-CSC conversion utility if `compressed2D` is CSR-only

---

## Phase 14.4: Tests

Create test files under `tests/unit/interface/`:

| Test file | What it tests |
|---|---|
| `test_blas_dispatch.cpp` | GEMM/GEMV dispatch produces same results as C++ fallback |
| `test_lapack_dispatch.cpp` | LU/QR/Cholesky/SVD/eigenvalue dispatch matches C++ fallback |
| `test_umfpack.cpp` | Sparse direct solve on small test matrices |

**Test strategy:**

- Each test computes a result with the native C++ path, then (if the library is available) computes via BLAS/LAPACK and compares within tolerance
- Tests are conditionally compiled: `#ifdef MTL5_HAS_BLAS` etc.
- CMake registers them only when the corresponding option is enabled

Update `tests/unit/CMakeLists.txt` to conditionally add the interface test subdirectory.

---

## Phase 14.5: Documentation & Examples

- Update or create `examples/phase14_external_libs.cpp` demonstrating actual dispatch
- Update `CLAUDE.md` Architecture section to document the dispatch mechanism
- Add `docs/modernization/phase14-summary.md` after completion

---

## Implementation Order

| Step | Description | Files | Est. Lines |
|---|---|---|---|
| 1 | Dispatch traits header | `interface/dispatch_traits.hpp` | ~40 |
| 2 | BLAS dispatch in `mult.hpp` | `operation/mult.hpp` | ~60 |
| 3 | LAPACK dispatch in LU | `operation/lu.hpp` | ~40 |
| 4 | LAPACK dispatch in QR | `operation/qr.hpp` | ~50 |
| 5 | LAPACK dispatch in Cholesky | `operation/cholesky.hpp` | ~30 |
| 6 | LAPACK dispatch in SVD | `operation/svd.hpp` | ~50 |
| 7 | LAPACK dispatch in eigenvalue | `operation/eigenvalue_symmetric.hpp` | ~40 |
| 8 | Additional LAPACK wrappers | `interface/lapack.hpp` | ~60 |
| 9 | BLAS norm dispatch (optional) | `operation/norms.hpp` | ~30 |
| 10 | UMFPACK CMake + interface | `CMakeLists.txt`, `interface/umfpack.hpp` | ~150 |
| 11 | Tests | `tests/unit/interface/*.cpp` | ~200 |
| 12 | Documentation | docs, examples | ~100 |
| **Total** | | | **~850** |

---

## Design Decisions

### 1. Row-major dispatch

**Decision:** Implement the transpose trick (`C = A*B <=> C^T = B^T * A^T`).

Rationale: Well-understood pattern, maximizes BLAS utilization for both orientations, already demonstrated in `phase9c_blas_dispatch.cpp`.

### 2. UMFPACK priority

**Decision:** Implement last; truly optional with graceful degradation.

Rationale: Requires SuiteSparse, a heavier dependency. Lower priority than BLAS/LAPACK which benefit all dense workloads.

### 3. Complex number support

**Decision:** Defer to a follow-up.

Rationale: Current wrappers only handle `float`/`double`. Adding `cblas_zgemm`/`cblas_cgemm` for `std::complex<float/double>` is straightforward but not needed for the initial phase.

### 4. CBLAS vs Fortran BLAS

**Decision:** Keep Fortran BLAS (trailing underscore convention).

Rationale: Maximum portability -- CBLAS header availability varies across platforms. Use the transpose trick for row-major handling instead of relying on CBLAS row-major support.
