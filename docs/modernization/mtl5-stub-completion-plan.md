# MTL5 Stub Completion Plan

**Goal:** Implement all 35 stub files in MTL5, organized into dependency-ordered phases.

---

## Phase 5: Triangular Solvers & Core Decompositions

**Rationale:** Triangular solvers are the foundation for LU, Cholesky, ILU(0), and IC(0). These are the most impactful stubs to fill.

### Step 1: `operation/trace.hpp`
- Simplest stub — `sum(A(i,i))` for any Matrix
- No dependencies beyond existing concepts
- ~10 lines of real code

### Step 2: `operation/random.hpp`
- Random fill for vectors and matrices using `<random>`
- Useful for tests of all subsequent phases
- Functions: `random_vector(n)`, `random_matrix(m,n)`, `fill_random(collection)`

### Step 3: `operation/lower_trisolve.hpp` and `operation/upper_trisolve.hpp`
- Forward substitution: solve `L*x = b` for lower triangular L
- Back substitution: solve `U*x = b` for upper triangular U
- Generic version using `A(i,j)` for any Matrix
- compressed2D specialization using raw CRS access for O(nnz)
- Unit-diagonal variant (for LU without diagonal storage)

### Step 4: `operation/lu.hpp`
- LU factorization with partial pivoting
- In-place: overwrites A with L\U, returns permutation vector
- `lu_factor(A, pivot)` — factor in place
- `lu_solve(LU, pivot, b)` — solve using trisolvers from Step 3
- Dense-only for Phase 5 (sparse LU deferred to UMFPACK interface)

### Step 5: `operation/householder.hpp`
- Householder reflection computation: `v, beta = house(x)`
- Apply Householder to matrix column/row
- Needed by QR, eigenvalue solvers, SVD

### Step 6: `operation/qr.hpp`
- QR factorization via Householder reflections
- `qr_factor(A)` — overwrites A with R, returns Q implicitly as Householder vectors
- `qr_solve(QR, tau, b)` — solve least-squares via Q^T*b then back-substitution
- Dense-only

### Step 7: `operation/lq.hpp`
- LQ factorization (QR on transposed, then transpose back)
- Can delegate to QR machinery from Step 6

### Step 8: `operation/cholesky.hpp`
- Cholesky decomposition for SPD matrices: A = L*L^T
- In-place: overwrites lower triangle of A with L
- `cholesky_factor(A)` — factor
- `cholesky_solve(L, b)` — forward/back substitution

### Step 9: `operation/inv.hpp`
- Matrix inverse via LU factorization from Step 4
- `inv(A)` returns dense2D
- Solve A*X = I column by column

### Step 10: `itl/pc/ilu_0.hpp`
- Incomplete LU(0) factorization for compressed2D
- Preserves sparsity pattern of A; no fill-in
- Constructor does factorization; `solve(x, b)` does L/U trisolves
- Requires triangular solvers from Step 3

### Step 11: `itl/pc/ic_0.hpp`
- Incomplete Cholesky(0) for SPD compressed2D
- Preserves sparsity pattern; no fill-in
- Constructor does factorization; `solve(x, b)` does L/L^T trisolves

**Tests to add:**
- `test_trisolve.cpp` — lower/upper triangular solve (dense + sparse)
- `test_lu.cpp` — LU factorization + solve
- `test_qr.cpp` — QR factorization + least-squares
- `test_cholesky.cpp` — Cholesky factorization + solve
- `test_ilu_ic.cpp` — ILU(0)/IC(0) preconditioned CG/BiCGSTAB

**Estimated effort:** ~800 lines of implementation, ~400 lines of tests

---

## Phase 6: Eigenvalue Solvers, SVD & Remaining Krylov

**Rationale:** Eigenvalue solvers and SVD are the most mathematically complex stubs. The remaining Krylov solvers are straightforward ports.

### Step 1: `operation/eigenvalue_symmetric.hpp`
- Symmetric eigenvalue solver via QR iteration with shifts
- Tridiagonalize via Householder, then implicit QR
- Returns eigenvalues as dense_vector, optionally eigenvectors as dense2D
- Uses Givens rotations (already implemented) for implicit QR step

### Step 2: `operation/eigenvalue.hpp`
- General eigenvalue solver via QR algorithm
- Reduce to upper Hessenberg via Householder, then implicit double-shift QR
- Returns eigenvalues (potentially complex) and eigenvectors

### Step 3: `operation/svd.hpp`
- Singular Value Decomposition: A = U*S*V^T
- Bidiagonalize via Householder, then implicit QR on bidiagonal
- Returns U, S (as vector), V

### Step 4: `itl/krylov/tfqmr.hpp`
- Transpose-Free QMR solver
- Same API pattern as BiCGSTAB: `tfqmr(A, x, b, M, iter)`
- No trans(A) needed (transpose-free)
- ~80 lines

### Step 5: `itl/krylov/qmr.hpp`
- Quasi-Minimal Residual solver
- Uses Lanczos biorthogonalization
- Requires trans(A) like BiCG
- ~100 lines

### Step 6: `itl/krylov/idr_s.hpp`
- IDR(s) — Induced Dimension Reduction with shadow space dimension s
- Modern solver, good for non-symmetric systems
- Default s=4
- ~120 lines

### Step 7: `operation/kron.hpp`
- Kronecker product: `kron(A, B)` returns dense2D of size (m1*m2, n1*n2)
- Straightforward nested loop implementation
- ~20 lines

**Tests to add:**
- `test_eigenvalue.cpp` — symmetric + general eigenvalue
- `test_svd.cpp` — SVD factorization + reconstruction
- `test_tfqmr.cpp`, `test_qmr.cpp`, `test_idr_s.cpp` — Krylov convergence tests
- `test_kron.cpp` — Kronecker product

**Estimated effort:** ~1200 lines of implementation, ~500 lines of tests

---

## Phase 7: Sparse Formats, Views & I/O

**Rationale:** Additional sparse formats and views expand the matrix ecosystem. Matrix Market I/O is essential for benchmarking.

### Step 1: `mat/identity2D.hpp`
- Implicit identity matrix: no storage, `operator()(r,c)` returns 1 if r==c else 0
- Satisfies Matrix concept
- ~30 lines

### Step 2: `mat/coordinate2D.hpp`
- COO (coordinate) sparse format: three arrays (row, col, value)
- `sort()` to order entries, `compress()` to convert to compressed2D
- Inserter pattern for easy assembly
- Useful for finite element assembly where entries arrive in arbitrary order

### Step 3: `mat/ell_matrix.hpp`
- ELLPACK format: fixed-width per-row storage
- Good for GPU kernels and matrices with uniform row lengths
- Constructor from compressed2D (auto-detect max row width)
- Matvec operator

### Step 4: `mat/view/banded_view.hpp`
- Read-only view that restricts access to a band [lower, upper] around diagonal
- `banded_view(A, lower, upper)` — only returns non-zero within band

### Step 5: `mat/view/hermitian_view.hpp`
- Read-only view: `operator()(r,c)` returns `conj(A(c,r))` when r > c
- Only stores upper (or lower) triangle

### Step 6: `mat/view/map_view.hpp`
- Read-only view with index remapping
- `map_view(A, row_map, col_map)` — reindexed access

### Step 7: `io/matrix_market.hpp`
- Read/write Matrix Market (.mtx) format
- Supports: real/complex/integer/pattern, general/symmetric/hermitian/skew
- Coordinate format parsing → compressed2D or dense2D
- `mm_read(filename)` → matrix, `mm_write(filename, A)`

**Tests to add:**
- `test_coordinate2D.cpp` — COO construction, sorting, compression to CRS
- `test_ell_matrix.cpp` — ELLPACK construction and matvec
- `test_matrix_views.cpp` — banded, hermitian, map views
- `test_matrix_market.cpp` — read/write round-trip with sample .mtx files

**Estimated effort:** ~600 lines of implementation, ~400 lines of tests

---

## Phase 8: Expression Templates (Optional/Advanced)

**Rationale:** Expression templates eliminate temporaries in chained operations. This is the most complex remaining work and may not be necessary if eager operations are fast enough.

### Step 1: `mat/expr/mat_expr.hpp`
- Base expression template class using CRTP
- Lazy evaluation: stores references, evaluates on assignment
- Integrates with existing operator overloads

### Step 2: `mat/expr/dmat_expr.hpp` and `mat/expr/smat_expr.hpp`
- Dense and sparse matrix expression specializations
- Capture operands by reference, evaluate element-by-element

### Step 3: `mat/expr/mat_mat_times_expr.hpp`
- Lazy matrix-matrix multiply expression
- Evaluates to dense2D on assignment, can fuse with other operations

### Step 4: `operation/lazy.hpp` and `operation/fuse.hpp`
- Lazy evaluation entry points
- Operation fusion framework for combining expressions

### Step 5: Vector expression templates
- `vec/expr/vec_expr.hpp` — already exists as potential stub
- Vector-vector and vector-scalar expression types

**Note:** This phase is explicitly optional. MTL5's eager operations are correct and performant for most use cases. Expression templates add complexity and should only be pursued if benchmarking shows significant performance gaps.

**Estimated effort:** ~1500 lines of implementation, ~300 lines of tests

---

## Phase 9: Remaining Stubs

### Recursion (3 stubs)
- `recursion/base_case_test.hpp` — threshold test for recursive subdivision
- `recursion/matrix_recursator.hpp` — recursive matrix quad-tree subdivision
- `recursion/predefined_masks.hpp` — masks for selecting quadrants
- Only needed if block-recursive algorithms are desired

### Sparse vector (2 stubs)
- `vec/sparse_vector.hpp` — sparse vector with sorted index/value arrays
- `vec/inserter.hpp` — inserter pattern for sparse vector construction

### External interfaces (2 stubs)
- `interface/blas.hpp` — optional BLAS dispatch for dense operations
- `interface/lapack.hpp` — optional LAPACK dispatch for decompositions
- Detect via CMake `find_package(BLAS)` / `find_package(LAPACK)`
- Fallback to native MTL5 implementations when not available

**Estimated effort:** ~500 lines of implementation

---

## Summary: Stub Completion Roadmap

| Phase | Stubs Filled | Key Deliverables | Dependencies |
|-------|-------------|-----------------|--------------|
| **5** | 12 | Triangular solvers, LU, QR, Cholesky, ILU(0), IC(0) | None (Phase 4 complete) |
| **6** | 7 | Eigenvalues, SVD, TFQMR, QMR, IDR(s), Kronecker | Phase 5 (Householder) |
| **7** | 7 | COO, ELLPACK, views, Matrix Market I/O | None |
| **8** | 4 | Expression templates (optional) | Phases 5-7 |
| **9** | 5 | Recursion, sparse vector, BLAS/LAPACK bindings | Phases 5-7 |
| **Total** | **35** | All stubs filled | |

**Total estimated implementation:** ~4600 lines + ~1600 lines of tests
