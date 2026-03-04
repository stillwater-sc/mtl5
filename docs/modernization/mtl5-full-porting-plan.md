# MTL5 Full Porting Plan — Unported MTL4 Features

**Goal:** Port the ~200 MTL4 headers that have no MTL5 equivalent at all.

**Design principle:** Not everything should be ported 1:1. C++20 eliminates the need for many MTL4 utility headers. This plan categorizes features into **Port**, **Redesign**, **Skip**, or **Defer**.

---

## Triage Summary

| Category | MTL4 Files | Decision |
|----------|-----------|----------|
| Expression templates & CRTP | ~30 | **Redesign** — use C++20 concepts, not CRTP |
| Vector expression types | ~20 | **Skip** — eager ops sufficient; optional Phase 8 |
| Matrix expression types | ~7 | **Skip** — same as above |
| Triangular/special views | ~6 | **Port** (Phase 10) |
| Matrix setup utilities | ~4 | **Port** (Phase 12) |
| Specialized matrix types | ~6 | **Selective port** (Phase 10) |
| Vector utilities | ~8 | **Selective port** (Phase 11) |
| Insertion & reordering | ~10 | **Selective port** (Phase 11) |
| Iterator infrastructure | ~5 | **Skip** — use `std::ranges` instead |
| Utility metaprogramming | ~45 | **Mostly skip** — C++20 concepts replace this |
| Index evaluation & lazy | ~6 | **Skip** — tied to expression templates |
| Advanced ITL | ~15 | **Selective port** (Phase 13) |
| I/O & file formats | ~8 | **Port** (Phase 12) |
| External interfaces | ~3 | **Port** (Phase 14) |
| **Total** | **~200** | **~60 to port, ~140 eliminated by C++20** |

---

## Phase 10: Matrix Views & Specialized Types

**What to port:** The most commonly used matrix views and formats that MTL4 users rely on.

### 10.1 Triangular Views
Create `mat/view/` headers:

| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `mat/view/upper_view.hpp` | `matrix/upper.hpp` | Read-only upper triangular view |
| `mat/view/lower_view.hpp` | `matrix/lower.hpp` | Read-only lower triangular view |
| `mat/view/strict_upper_view.hpp` | `matrix/strict_upper.hpp` | Strict upper (zero diagonal) |
| `mat/view/strict_lower_view.hpp` | `matrix/strict_lower.hpp` | Strict lower (zero diagonal) |

**Design:**
- Lightweight non-owning views like `transposed_view`
- `operator()(r,c)` returns zero outside the triangle
- Satisfy `Matrix` concept
- Category: inherit from underlying matrix

### 10.2 Indirect (Submatrix) View
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `mat/view/indirect_view.hpp` | `matrix/indirect.hpp` | Reindexed submatrix |

**Design:**
- `indirect_view(A, row_indices, col_indices)`
- `operator()(r,c)` maps to `A(row_indices[r], col_indices[c])`
- Enables block extraction and submatrix operations

### 10.3 Permutation Matrix
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `mat/permutation.hpp` | `matrix/permutation.hpp` | Implicit permutation matrix |

**Design:**
- Store permutation vector only (no matrix storage)
- `operator()(r,c)` returns 1 if `perm[r] == c`, else 0
- Efficient `perm * x` and `perm * A` via index remapping
- Used by LU pivoting, reordering algorithms

### 10.4 Block Diagonal Matrix
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `mat/block_diagonal2D.hpp` | `matrix/block_diagonal2D.hpp` | Block-diagonal storage |

**Design:**
- Store vector of dense2D blocks
- Efficient matvec: apply each block independently
- Used in block preconditioners

**Estimated effort:** ~400 lines + ~200 lines of tests

---

## Phase 11: Vector & Insertion Utilities

### 11.1 Unit Vector
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `vec/unit_vector.hpp` | `vector/unit_vector.hpp` | Standard basis vector e_i |

**Design:**
- `unit_vector<T>(n, i)` — returns dense_vector with 1 at position i
- Or implicit version: `operator()(j)` returns `j == i ? 1 : 0`

### 11.2 Strided Vector Reference
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `vec/strided_ref.hpp` | `vector/strided_vector_ref.hpp` | Non-owning strided view |

**Design:**
- `strided_ref(data_ptr, size, stride)` — access every stride-th element
- Needed for column extraction from row-major matrices
- Could use `std::span` with custom accessor in C++23; manual in C++20

### 11.3 Mapped Inserters
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `mat/shifted_inserter.hpp` | `matrix/shifted_inserter.hpp` | Offset insertion for FEM assembly |

**Design:**
- Wraps an inserter, adds offset to row/col indices
- `shifted_inserter(base_inserter, row_offset, col_offset)`
- Essential for finite element local-to-global assembly

### 11.4 Reordering
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `operation/reorder.hpp` | `matrix/reorder.hpp` | Permutation-based reordering |

**Design:**
- `reorder_rows(A, perm)`, `reorder_cols(A, perm)`, `reorder(A, perm)`
- Returns new compressed2D with reordered entries
- Used for bandwidth reduction (Cuthill-McKee, etc.)

**Estimated effort:** ~300 lines + ~150 lines of tests

---

## Phase 12: I/O, Setup Utilities & Convenience

### 12.1 Matrix Setup Helpers
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `operation/laplacian_setup.hpp` | `matrix/laplacian_setup.hpp` | 1D/2D Laplacian matrix |
| `operation/poisson2D.hpp` | `matrix/poisson2D_dirichlet.hpp` | 2D Poisson with BCs |
| `operation/diagonal_setup.hpp` | `matrix/diagonal_setup.hpp` | Diagonal matrix from vector |

**Design:**
- Free functions that return compressed2D
- `laplacian_1d(n)`, `laplacian_2d(nx, ny)`, `poisson2d_dirichlet(nx, ny)`
- `diag(vec)` → compressed2D with vector on diagonal

### 12.2 Extended I/O
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `io/read_el.hpp` | Various | Element-by-element file reader |
| `io/write_el.hpp` | Various | Element-by-element file writer |

**Design:**
- Simple CSV/whitespace format for quick prototyping
- `read_dense(filename)`, `write_dense(filename, A)`
- `read_sparse(filename)`, `write_sparse(filename, A)`

### 12.3 Pretty Printing Enhancements
- Enhance existing `operation/print.hpp` with:
  - Configurable precision
  - Sparse matrix printing (show only non-zeros)
  - MATLAB-compatible output format

**Estimated effort:** ~400 lines + ~200 lines of tests

---

## Phase 13: Advanced ITL Components

### 13.1 Additional Krylov Variants
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `itl/krylov/cgs.hpp` | `krylov/cgs.hpp` | Conjugate Gradient Squared |
| `itl/krylov/bicgstab_ell.hpp` | `krylov/bicgstab_ell.hpp` | BiCGSTAB(ell) |
| `itl/krylov/minres.hpp` | `krylov/minres.hpp` | MINRES for symmetric indefinite |

### 13.2 Advanced Preconditioners
| New File | MTL4 Source | Description |
|----------|-------------|-------------|
| `itl/pc/ilut.hpp` | `pc/ilut.hpp` | ILU with threshold dropping |
| `itl/pc/ildl.hpp` | — | Incomplete LDL^T for symmetric |
| `itl/pc/block_diagonal.hpp` | — | Block-diagonal PC |
| `itl/pc/ssor.hpp` | — | Symmetric SOR preconditioner |

### 13.3 Multigrid Framework
| New File | Description |
|----------|-------------|
| `itl/mg/multigrid.hpp` | V-cycle/W-cycle multigrid |
| `itl/mg/restriction.hpp` | Restriction operator (fine → coarse) |
| `itl/mg/prolongation.hpp` | Prolongation operator (coarse → fine) |

**Design:**
- `multigrid(levels, smoother, coarse_solver, restrictor, prolongator)`
- Template on smoother type (use existing Gauss-Seidel, Jacobi, SOR)
- Geometric and algebraic variants

**Estimated effort:** ~1000 lines + ~400 lines of tests

---

## Phase 14: External Library Bindings

### 14.1 BLAS Interface
- Detect BLAS via CMake `find_package(BLAS)`
- Dispatch dense matvec/matmul to BLAS when available
- Fallback to native MTL5 implementations
- Key routines: `dgemv`, `dgemm`, `dtrsv`, `dtrsm`

### 14.2 LAPACK Interface
- Detect LAPACK via CMake `find_package(LAPACK)`
- Dispatch LU, QR, Cholesky, SVD, eigenvalue to LAPACK when available
- Key routines: `dgetrf`, `dgeqrf`, `dpotrf`, `dgesvd`, `dsyev`

### 14.3 UMFPACK Interface (Optional)
- Sparse direct solver for compressed2D
- `umfpack_solve(A, x, b)` — direct sparse LU

**Estimated effort:** ~600 lines + ~200 lines of tests

---

## What NOT to Port (~140 MTL4 files)

### Eliminated by C++20 (~100 files)
These MTL4 headers exist solely because C++03/11 lacked language features that C++20 provides:

| MTL4 Category | Why Not Needed |
|---------------|----------------|
| `utility/is_what.hpp` family (~15 files) | Replaced by C++20 concepts |
| `utility/tag.hpp`, `glas_tag.hpp` | Replaced by `mtl::tag::` namespace |
| `utility/ashape.hpp`, `algebraic_category.hpp` | Replaced by `mtl::traits::` |
| `utility/type_parameter*.hpp` (~5 files) | Replaced by `if constexpr` + concepts |
| `utility/enable_if*.hpp` (~3 files) | Replaced by `requires` clauses |
| `utility/property_map.hpp` | Replaced by direct trait queries |
| `utility/wrapped_object.hpp`, `sometimes_data.hpp` | Replaced by `std::optional`, `if constexpr` |
| `utility/static_assert.hpp` | Built-in `static_assert` |
| `utility/exception.hpp` | Use `<stdexcept>` directly |
| `utility/common_include.hpp` | Not needed with modules/pragma once |
| `utility/omp_size_type.hpp`, `papi.hpp` | Niche; defer indefinitely |
| `utility/complexity.hpp` | Academic; not needed for correctness |
| CRTP bases (~10 files) | Replaced by C++20 concepts |
| Boost.MPL dispatch (~8 files) | Replaced by `if constexpr` |

### Expression Template Types (~40 files)
All `vec_vec_*_expr.hpp`, `vec_scal_*_expr.hpp`, `mat_mat_*_expr.hpp` files. These implement lazy evaluation through expression templates. MTL5 uses eager operators. If expression templates are ever added (Phase 8), they'll be redesigned from scratch using C++20 features, not 1:1 ported.

### Iterator Infrastructure (~5 files)
`iterator_adaptor_1D.hpp`, `dense_el_cursor.hpp`, etc. MTL5 uses range-based for loops and `begin()`/`end()` directly. C++20 ranges can replace custom iterators if needed.

---

## Implementation Priority Order

| Priority | Phase | Files | Impact |
|----------|-------|-------|--------|
| **Critical** | 5 | 12 | Triangular solvers, LU, QR, Cholesky, ILU(0), IC(0) |
| **High** | 6 | 7 | Eigenvalues, SVD, remaining Krylov solvers |
| **High** | 7 | 7 | Sparse formats, Matrix Market I/O |
| **Medium** | 10 | 7 | Triangular views, permutation, block diagonal |
| **Medium** | 11 | 4 | Vector utilities, reordering |
| **Medium** | 12 | 6 | Setup helpers, extended I/O |
| **Medium** | 13 | 7+ | Advanced ITL (ILUT, multigrid) |
| **Low** | 14 | 3 | BLAS/LAPACK bindings |
| **Optional** | 8 | 4 | Expression templates |
| **Optional** | 9 | 5 | Recursion, sparse vector |

---

## Total Estimated New Code

| Phase | Implementation Lines | Test Lines |
|-------|---------------------|------------|
| 5 (stubs) | 800 | 400 |
| 6 (stubs) | 1200 | 500 |
| 7 (stubs) | 600 | 400 |
| 8 (stubs, optional) | 1500 | 300 |
| 9 (stubs) | 500 | 200 |
| 10 (new) | 400 | 200 |
| 11 (new) | 300 | 150 |
| 12 (new) | 400 | 200 |
| 13 (new) | 1000 | 400 |
| 14 (new) | 600 | 200 |
| **Total** | **~7300** | **~2950** |

**Bottom line:** Of MTL4's ~367 files, roughly **60 need porting as new code**, **35 are stubs to fill**, and **~140 are eliminated by C++20 modernization**. The remaining ~130 are already implemented. The library will be functionally complete at ~95 stub + new files, totaling approximately 10,000 lines of new code including tests.
