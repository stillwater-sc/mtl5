# MTL5 Modernization Status Assessment

**Date:** 2026-03-03
**Scope:** Full comparison of MTL4 (367 headers) vs MTL5 (123 headers)

---

## Executive Summary

MTL5 is approximately **32% complete** relative to MTL4's full scope. The foundation is solid: core data types, eager arithmetic, iterative solvers, and the C++20 concept framework are all working. The remaining work falls into two categories: **35 stub files** that need implementation, and **~200 MTL4 headers** that have no MTL5 equivalent at all.

---

## What's Working (88 implemented files)

### Core Types
| Component | File | Status |
|-----------|------|--------|
| Dense matrix | `mat/dense2D.hpp` | Full implementation (row/col-major, fixed/dynamic) |
| Sparse matrix (CRS) | `mat/compressed2D.hpp` | Full implementation (three-array CRS) |
| Sparse inserter | `mat/inserter.hpp` | Full implementation (slot + overflow, RAII) |
| Dense vector | `vec/dense_vector.hpp` | Full implementation (fixed/dynamic) |
| Transposed view | `mat/view/transposed_view.hpp` | Full implementation with `base()` accessor |

### C++20 Concepts (7 files)
`scalar`, `magnitude`, `collection`, `matrix` (with `DenseMatrix`/`SparseMatrix`), `vector`, `linear_operator`, `preconditioner`

### Tags & Traits (9 files)
`orientation`, `sparsity`, `storage`, `shape`, `traversal`, `category`, `ashape`, `transposed_orientation`, `flatcat`

### Operations (24 implemented)
- **Reductions:** `dot`, `sum`, `product`, `norms` (one/two/infinity/Frobenius)
- **Element-wise:** `abs`, `conj`, `negate`, `sqrt`, `max`, `min`
- **Matrix ops:** `trans`, `mult`, `diagonal`, `scale`, `givens`
- **Utilities:** `set_to_zero`, `fill`, `print`, `size`, `num_rows`, `num_cols`, `resource`
- **Operators:** vector operators, matrix operators (including sparse matvec)

### Functor Library (12 files)
Scalar functors: `plus`, `minus`, `times`, `divide`, `assign`, `negate`, `abs`, `conj`, `sqrt`
Typed functors: `scale`, `rscale`, `divide_by`

### ITL — Iterative Template Library (12 implemented)
- **Iteration controllers:** `basic_iteration`, `cyclic_iteration`, `noisy_iteration`
- **Krylov solvers:** CG, BiCG, BiCGSTAB, GMRES (with restart)
- **Preconditioners:** `identity`, `diagonal` (Jacobi)
- **Smoothers:** `gauss_seidel`, `jacobi`, `sor` (all with compressed2D specializations)

### Infrastructure (4 files)
`config.hpp`, `mtl.hpp`, `mtl_fwd.hpp`, `math/identity.hpp`, `math/operations.hpp`, `detail/contiguous_memory_block.hpp`, `detail/index.hpp`

---

## Stub Files (35 files — planned but not implemented)

### Decompositions & Solvers (16 stubs)

| File | What It Should Do |
|------|-------------------|
| `operation/lu.hpp` | LU factorization with partial pivoting |
| `operation/qr.hpp` | QR factorization (Householder) |
| `operation/lq.hpp` | LQ factorization |
| `operation/cholesky.hpp` | Cholesky decomposition (SPD matrices) |
| `operation/svd.hpp` | Singular Value Decomposition |
| `operation/eigenvalue.hpp` | General eigenvalue solver |
| `operation/eigenvalue_symmetric.hpp` | Symmetric eigenvalue solver |
| `operation/householder.hpp` | Householder reflections |
| `operation/inv.hpp` | Matrix inverse |
| `operation/lower_trisolve.hpp` | Forward substitution (lower triangular) |
| `operation/upper_trisolve.hpp` | Back substitution (upper triangular) |
| `operation/trace.hpp` | Matrix trace (sum of diagonal) |
| `operation/kron.hpp` | Kronecker product |
| `operation/random.hpp` | Random matrix/vector generation |
| `operation/fuse.hpp` | Fused operations |
| `operation/lazy.hpp` | Lazy evaluation entry point |

### Sparse Matrix Formats (3 stubs)

| File | What It Should Do |
|------|-------------------|
| `mat/coordinate2D.hpp` | COO (coordinate) sparse format |
| `mat/ell_matrix.hpp` | ELLPACK sparse format |
| `mat/identity2D.hpp` | Identity matrix (implicit, O(1) storage) |

### Matrix Views (3 stubs)

| File | What It Should Do |
|------|-------------------|
| `mat/view/banded_view.hpp` | Banded matrix view |
| `mat/view/hermitian_view.hpp` | Hermitian (conjugate transpose) view |
| `mat/view/map_view.hpp` | Remapped index view |

### Expression Templates (4 stubs)

| File | What It Should Do |
|------|-------------------|
| `mat/expr/dmat_expr.hpp` | Dense matrix expression |
| `mat/expr/smat_expr.hpp` | Sparse matrix expression |
| `mat/expr/mat_expr.hpp` | Generic matrix expression base |
| `mat/expr/mat_mat_times_expr.hpp` | Lazy matrix-matrix multiply |

### Vector (2 stubs)

| File | What It Should Do |
|------|-------------------|
| `vec/sparse_vector.hpp` | Sparse vector type |
| `vec/inserter.hpp` | Vector inserter for sparse assembly |

### ITL (5 stubs)

| File | What It Should Do |
|------|-------------------|
| `itl/krylov/tfqmr.hpp` | Transpose-Free QMR solver |
| `itl/krylov/qmr.hpp` | Quasi-Minimal Residual solver |
| `itl/krylov/idr_s.hpp` | IDR(s) solver |
| `itl/pc/ilu_0.hpp` | Incomplete LU(0) preconditioner |
| `itl/pc/ic_0.hpp` | Incomplete Cholesky(0) preconditioner |

### Recursion (3 stubs)

| File | What It Should Do |
|------|-------------------|
| `recursion/base_case_test.hpp` | Base case test for recursive algorithms |
| `recursion/matrix_recursator.hpp` | Recursive matrix subdivision |
| `recursion/predefined_masks.hpp` | Predefined recursion masks |

### External Interfaces (2 stubs)

| File | What It Should Do |
|------|-------------------|
| `interface/blas.hpp` | BLAS bindings |
| `interface/lapack.hpp` | LAPACK bindings |

### I/O (1 stub)

| File | What It Should Do |
|------|-------------------|
| `io/matrix_market.hpp` | Matrix Market format reader/writer |

---

## MTL4 Features With No MTL5 Equivalent (~200 files)

### Expression Template Infrastructure (~30 files)
The entire lazy evaluation framework: CRTP bases, vector-vector/vector-scalar/matrix-matrix expression types, index evaluators, lazy reduction framework. MTL5 uses eager operators; expression templates would be a Phase 7+ effort.

### Matrix Views & Specialized Formats (~15 files)
- `upper.hpp`, `lower.hpp`, `strict_upper.hpp`, `strict_lower.hpp` — Triangular views
- `indirect.hpp` — Reindexed matrix view
- `block_diagonal2D.hpp` — Block diagonal format
- `morton_dense.hpp` — Morton Z-order layout
- `sparse_banded.hpp` — Sparse banded format
- `permutation.hpp` — Permutation matrix
- `multi_vector.hpp` — Column-block matrix

### Matrix Setup Utilities (~4 files)
`diagonal_setup.hpp`, `hessian_setup.hpp`, `laplacian_setup.hpp`, `poisson2D_dirichlet.hpp`

### Vector Utilities (~8 files)
`unit_vector.hpp`, `extracter.hpp`, `strided_vector_ref.hpp`, expression types for vector-vector and vector-scalar operations

### Insertion & Reordering (~10 files)
`shifted_inserter.hpp`, `mapped_inserter.hpp`, `reorder.hpp`, `reorder_matrix_rows.hpp`, element structures

### Iterator & Adaptor Infrastructure (~5 files)
`iterator_adaptor_1D.hpp`, `dense_el_cursor.hpp`, composition framework

### Utility & Metaprogramming (~45 files)
Type generators, type queries (`is_what.hpp`, `is_row_major.hpp`, `is_lazy.hpp`), property maps, complexity tags, range/set utilities, exception hierarchy, PAPI support

### Advanced ITL Components (~15 files)
ARMS, ILUT, IMF preconditioners; CGS, BiCGSTAB(ell) solver variants; FSM framework

---

## Test Coverage

| Test Suite | Tests | Status |
|------------|-------|--------|
| test_concepts | 1 | Pass |
| test_zero_one | 1 | Pass |
| test_contiguous_memory_block | 1 | Pass |
| test_dense2D | 1 | Pass |
| test_compressed2D | 1 | Pass |
| test_dense_vector | 1 | Pass |
| test_dot | 1 | Pass |
| test_norms | 1 | Pass |
| test_vector_ops | 1 | Pass |
| test_matrix_ops | 1 | Pass |
| test_scalar_functors | 1 | Pass |
| test_cg | 1 | Pass |
| test_bicgstab | 1 | Pass |
| test_bicg | 1 | Pass |
| test_gmres | 1 | Pass |
| test_smoothers | 1 | Pass |
| **Total** | **16** | **All pass** |

---

## Summary by Category

| Category | MTL4 | MTL5 Implemented | MTL5 Stubs | Not Started |
|----------|------|-------------------|------------|-------------|
| Matrix types | 11 | 2 | 3 | 6 |
| Vector types | 2 | 1 | 1 | 0 |
| Operations | 143 | 24 | 16 | ~103 |
| Views | 8 | 1 | 3 | 4 |
| ITL solvers | 14 | 4 | 3 | 7 |
| ITL preconditioners | 16 | 2 | 2 | 12 |
| ITL smoothers | 6 | 3 | 0 | 3 |
| Utilities | ~50 | 9 (concepts/traits) | 0 | ~41 |
| Expression templates | ~30 | 0 | 4 | ~26 |
| Interfaces | 5 | 0 | 2 | 3 |
| I/O | 9 | 0 | 1 | 8 |
| **Total** | **~367** | **88** | **35** | **~244** |
