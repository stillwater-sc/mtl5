# Sparse Direct Solvers for MTL5 — Design Document

## Background and Motivation

MTL5's iterative solver library (ITL) draws from Greenbaum's *Iterative Methods for Solving Linear Systems* (SIAM Frontiers in Applied Mathematics). The complementary world of **sparse direct methods** — rooted in Timothy Davis's *Direct Methods for Sparse Linear Systems* (SIAM Fundamentals of Algorithms) and his CSparse/SuiteSparse ecosystem — is currently absent from MTL5 except for the external UMFPACK binding.

This document architects a **native sparse direct solver framework** for MTL5, covering three factorizations (LU, QR, Cholesky), the critical infrastructure they depend on (fill-reducing orderings, elimination trees, symbolic analysis), and optional external solver interfaces (SuperLU, KLU, CHOLMOD, SPQR).

---

## 1. The CSparse World: What We're Drawing From

### Core Insight: Symbolic/Numeric Phase Separation

Every sparse direct solver in Davis's framework separates into two phases:

1. **Symbolic analysis** — Computes fill-reducing ordering, elimination tree, and nonzero structure of the factors. Uses only the sparsity pattern (no numerical values). Can be reused across matrices with the same pattern.

2. **Numeric factorization** — Fills in the numerical values using the pre-computed structure. This is where the BLAS-level arithmetic happens.

This separation is the single most important architectural decision. It enables:
- Solving multiple systems with the same sparsity pattern (common in time-stepping PDE solvers)
- Accurate memory pre-allocation (no reallocs during factorization)
- Compile-time analysis opportunities (the Sympiler approach)

### The Solver Family

| Solver | Method | Best For | Key Feature |
|--------|--------|----------|-------------|
| **CSparse** | Left-looking LU/QR/Cholesky | Education, small-medium problems | Gilbert-Peierls algorithm, concise implementation |
| **UMFPACK** | Multifrontal LU | General unsymmetric systems | Rectangular frontal matrices, Level-3 BLAS |
| **SuperLU** | Supernodal left-looking LU | Large unsymmetric systems | Dense supernodal BLAS on column groups |
| **KLU** | BTF + left-looking LU per block | Circuit simulation matrices | Dulmage-Mendelsohn decomposition to BTF |
| **CHOLMOD** | Supernodal/up-looking Cholesky | SPD systems | Highest-performance sparse Cholesky |
| **SPQR** | Multifrontal QR | Rank-deficient/least-squares | Multithreaded rank-revealing QR |

### Fill-Reducing Orderings

All solvers depend on orderings to minimize fill-in:
- **AMD** (Approximate Minimum Degree) — for symmetric patterns (Cholesky, symmetric LU)
- **COLAMD** (Column AMD) — for unsymmetric patterns (LU with partial pivoting, QR on A^T A)
- **Nested dissection** — for 2D/3D mesh problems with separator structure
- **BTF** (Block Triangular Form via Dulmage-Mendelsohn) — for circuit-like matrices

### The Elimination Tree

The **elimination tree** is the foundational data structure for all sparse direct methods:
- Encodes column dependencies during factorization
- Parent of node j = first nonzero below the diagonal in column j of L
- Determines optimal traversal order, memory allocation, parallelism opportunities
- Computable in O(nnz) time

---

## 2. Proposed Source Layout

```
include/mtl/
├── sparse/                          # Sparse direct solver infrastructure
│   ├── ordering/
│   │   ├── amd.hpp                  # Approximate Minimum Degree
│   │   ├── colamd.hpp               # Column Approximate Minimum Degree
│   │   ├── rcm.hpp                  # Reverse Cuthill-McKee (bandwidth reduction)
│   │   └── ordering_concepts.hpp    # FillReducingOrdering concept
│   ├── analysis/
│   │   ├── elimination_tree.hpp     # Elimination tree construction
│   │   ├── symbolic_factorization.hpp  # Symbolic analysis phase
│   │   ├── column_counts.hpp        # Nonzero count prediction for L/U
│   │   └── postorder.hpp            # Tree postordering
│   ├── factorization/
│   │   ├── sparse_lu.hpp            # Sparse LU (Gilbert-Peierls left-looking)
│   │   ├── sparse_qr.hpp           # Sparse QR (Householder, left-looking)
│   │   ├── sparse_cholesky.hpp     # Sparse Cholesky (up-looking or left-looking)
│   │   ├── factor_result.hpp        # Factorization result types (symbolic + numeric)
│   │   └── triangular_solve.hpp     # Sparse triangular solve (reach + solve)
│   ├── util/
│   │   ├── csc.hpp                  # CSC view/adapter for compressed2D
│   │   ├── permutation.hpp          # Permutation vector operations
│   │   ├── dulmage_mendelsohn.hpp   # BTF decomposition
│   │   └── scatter.hpp              # Sparse accumulator (scatter/gather)
│   └── sparse.hpp                   # Umbrella include for sparse direct
├── interface/
│   ├── umfpack.hpp                  # (existing)
│   ├── superlu.hpp                  # SuperLU interface
│   ├── klu.hpp                      # KLU interface
│   ├── cholmod.hpp                  # CHOLMOD interface
│   └── spqr.hpp                     # SPQR interface
├── operation/
│   ├── lu.hpp                       # (existing — add sparse dispatch)
│   ├── qr.hpp                       # (existing — add sparse dispatch)
│   └── cholesky.hpp                 # (existing — add sparse dispatch)
```

### Namespace Mapping

```cpp
namespace mtl::sparse {              // Top-level sparse direct namespace
namespace mtl::sparse::ordering {     // Fill-reducing orderings
namespace mtl::sparse::analysis {     // Symbolic analysis infrastructure
namespace mtl::sparse::factorization{ // Numeric factorization algorithms
namespace mtl::sparse::util {         // Permutations, format conversion, etc.
namespace mtl::interface {            // External solver wrappers (existing)
```

---

## 3. Core Concepts and Type Design

### New Concepts

```cpp
// include/mtl/sparse/ordering/ordering_concepts.hpp

namespace mtl::sparse {

/// A fill-reducing ordering algorithm
template <typename O>
concept FillReducingOrdering = requires(const O& ord,
                                         const mat::compressed2D<double>& A) {
    { ord(A) } -> std::convertible_to<std::vector<std::size_t>>;
    // Returns permutation vector p where p[new] = old
};

/// Result of symbolic analysis — reusable across matrices with same pattern
template <typename S>
concept SymbolicAnalysis = requires(const S& sym) {
    { sym.nnz_L() } -> std::convertible_to<std::size_t>;  // predicted fill
    { sym.permutation() } -> std::ranges::range;           // ordering used
};

/// A sparse direct solver (factorization + solve)
template <typename F>
concept SparseDirectSolver = requires(const F& f,
                                       vec::dense_vector<double>& x,
                                       const vec::dense_vector<double>& b) {
    { f.solve(x, b) };           // Solve Ax = b
    { f.num_rows() } -> std::convertible_to<std::size_t>;
    { f.num_cols() } -> std::convertible_to<std::size_t>;
};

}  // namespace mtl::sparse
```

### Factorization Result Types

The key design decision: separate symbolic and numeric results into composable types.

```cpp
// include/mtl/sparse/factorization/factor_result.hpp

namespace mtl::sparse {

/// Symbolic factorization result — reusable for same-pattern matrices
template <typename Value>
struct symbolic_result {
    std::vector<std::size_t> parent;       // elimination tree
    std::vector<std::size_t> post;         // postordering
    std::vector<std::size_t> col_counts;   // nonzero counts per column of L
    std::vector<std::size_t> perm;         // fill-reducing permutation
    std::vector<std::size_t> pinv;         // inverse permutation
    std::size_t nnz_L;                     // predicted nnz in L
    std::size_t nnz_U;                     // predicted nnz in U (for LU)
};

/// Numeric LU factorization result
template <typename Value>
struct lu_numeric {
    mat::compressed2D<Value> L;            // lower triangular factor
    mat::compressed2D<Value> U;            // upper triangular factor
    std::vector<std::size_t> pivot;        // row pivoting permutation
    symbolic_result<Value> symbolic;       // symbolic analysis used

    void solve(auto& x, const auto& b) const;
    void solve_transpose(auto& x, const auto& b) const;
};

/// Numeric Cholesky factorization result
template <typename Value>
struct cholesky_numeric {
    mat::compressed2D<Value> L;            // lower triangular factor
    symbolic_result<Value> symbolic;

    void solve(auto& x, const auto& b) const;   // L L^T x = b
};

/// Numeric QR factorization result
template <typename Value>
struct qr_numeric {
    mat::compressed2D<Value> R;            // upper triangular factor
    mat::compressed2D<Value> V;            // Householder vectors
    std::vector<Value> beta;               // Householder coefficients
    symbolic_result<Value> symbolic;

    void solve(auto& x, const auto& b) const;   // least-squares
};

}  // namespace mtl::sparse
```

---

## 4. Implementation Phases

### Phase 1: Infrastructure (Foundation)

Build the reusable infrastructure that all three factorizations depend on.

| Component | File | Description |
|-----------|------|-------------|
| CSC adapter | `sparse/util/csc.hpp` | Zero-cost CSC view over `compressed2D` with col-major parameters; CRS-to-CSC conversion |
| Permutation | `sparse/util/permutation.hpp` | Apply/invert/compose permutation vectors; permuted matrix views |
| Scatter | `sparse/util/scatter.hpp` | Sparse accumulator for column assembly (the workhorse of sparse arithmetic) |
| Elimination tree | `sparse/analysis/elimination_tree.hpp` | O(nnz) etree construction from CSC |
| Postorder | `sparse/analysis/postorder.hpp` | DFS-based tree postordering |
| Column counts | `sparse/analysis/column_counts.hpp` | Predict nnz per column of L using etree |
| Sparse triangular solve | `sparse/factorization/triangular_solve.hpp` | Reach (topological sort via DFS) + sparse forward/back substitution |

The **sparse triangular solve** (reach + solve) is the single most critical routine — it's used inside every factorization and every solve phase. This is Gilbert-Peierls' key contribution: solving Lx = b in time proportional to nnz(x), not nnz(L).

### Phase 2: Sparse Cholesky

No pivoting needed (SPD matrices are unconditionally stable), simplest factorization, clearest demonstration of the symbolic/numeric framework.

```cpp
// Usage:
mat::compressed2D<double> A(n, n);  // SPD matrix
// ... fill A ...

// Two-phase approach:
auto sym  = sparse::cholesky_symbolic(A, sparse::ordering::amd{});
auto num  = sparse::cholesky_numeric(A, sym);
num.solve(x, b);

// One-shot convenience:
sparse::cholesky_solve(A, x, b);  // ordering chosen automatically
```

**Algorithm:** Up-looking Cholesky (Liu's algorithm)
- For each column j: solve L(1:j-1, 1:j-1) * x = A(1:j-1, j), then L(j,j) = sqrt(A(j,j) - dot)
- Uses etree to determine which columns of L affect column j
- O(nnz(L)) arithmetic operations

### Phase 3: Sparse LU

**Algorithm:** Gilbert-Peierls left-looking LU with threshold partial pivoting

```cpp
// Usage:
auto sym = sparse::lu_symbolic(A, sparse::ordering::colamd{});
auto num = sparse::lu_numeric(A, sym, threshold);  // threshold in (0,1], default 0.1
num.solve(x, b);

// One-shot:
sparse::lu_solve(A, x, b);
```

**Key decisions:**
- **Threshold partial pivoting** (tau = 0.1 default): accept pivot if |a_kk| >= tau * max|a_ik|
- **COLAMD** pre-ordering applied to columns; row pivoting during factorization
- Left-looking: for each column k, perform sparse triangular solve L * x = A(:,k), then select pivot from x

### Phase 4: Sparse QR

**Algorithm:** Left-looking Householder QR (CSparse-style)

```cpp
// Usage:
auto sym = sparse::qr_symbolic(A, sparse::ordering::colamd{});
auto num = sparse::qr_numeric(A, sym);
num.solve(x, b);  // least-squares solution

// Extract factors:
auto [Q, R] = sparse::qr_extract(num);
```

**Notes:**
- Works on rectangular matrices (m x n, m >= n)
- Symbolic analysis on A^T A pattern (without forming A^T A)
- Householder vectors stored in compact form

### Phase 5: AMD/COLAMD Orderings

Native implementations of the two essential orderings:

**AMD** — Approximate Minimum Degree:
- Quotient graph technique (O(nnz) space, doesn't grow during elimination)
- Approximate degree bounds (avoids expensive exact degree computation)
- Quality comparable to exact minimum degree; speed comparable to O(nnz)

**COLAMD** — Column AMD:
- Applies AMD to A^T A pattern without forming A^T A
- Used for unsymmetric LU and QR

**RCM** — Reverse Cuthill-McKee (simpler, for bandwidth reduction):
- BFS-based, useful as a baseline ordering

### Phase 6: External Solver Interfaces

Following the existing UMFPACK pattern, add optional interfaces:

| Interface | CMake Flag | Purpose |
|-----------|-----------|---------|
| `superlu.hpp` | `MTL5_HAS_SUPERLU` | Supernodal LU for large unsymmetric systems |
| `klu.hpp` | `MTL5_HAS_KLU` | Circuit simulation LU (BTF + left-looking) |
| `cholmod.hpp` | `MTL5_HAS_CHOLMOD` | High-performance supernodal Cholesky |
| `spqr.hpp` | `MTL5_HAS_SPQR` | Multifrontal rank-revealing QR |

Each follows the RAII pattern established by `umfpack_solver`:
```cpp
// Pattern for all external solver wrappers:
class klu_solver {
public:
    explicit klu_solver(const mat::compressed2D<double>& A);  // factorize
    ~klu_solver();                                             // free resources
    klu_solver(klu_solver&&) noexcept;                        // movable
    klu_solver& operator=(klu_solver&&) noexcept;
    klu_solver(const klu_solver&) = delete;                   // non-copyable

    void solve(auto& x, const auto& b) const;
    void solve_transpose(auto& x, const auto& b) const;

    std::size_t num_rows() const;
    std::size_t num_cols() const;
};
```

### Phase 7: Unified Dispatch

Extend the existing operations (`lu.hpp`, `qr.hpp`, `cholesky.hpp`) to auto-dispatch:

```
lu_factor(A, pivot)
  +-- if dense + MTL5_HAS_LAPACK  -> dgetrf_    (existing)
  +-- if dense                     -> dense LU   (existing)
  +-- if sparse + MTL5_HAS_UMFPACK -> UMFPACK   (existing)
  +-- if sparse + MTL5_HAS_SUPERLU -> SuperLU   (new)
  +-- if sparse                    -> sparse LU  (new, native)
```

This uses the existing `if constexpr` + `traits::category` pattern already in place for BLAS/LAPACK dispatch.

---

## 5. Integration with ITL

Sparse direct solvers naturally serve as **preconditioners** for iterative methods. The bridge is straightforward:

```cpp
namespace mtl::itl::pc {

/// Use any sparse direct solver as a preconditioner
template <SparseDirectSolver Solver>
class direct_preconditioner {
    Solver solver_;
public:
    template <Matrix M>
    explicit direct_preconditioner(const M& A) : solver_(A) {}

    void solve(auto& x, const auto& b) const { solver_.solve(x, b); }
    void adjoint_solve(auto& x, const auto& b) const {
        solver_.solve_transpose(x, b);
    }
};

}  // namespace mtl::itl::pc
```

This allows using a sparse Cholesky or LU as a direct preconditioner within GMRES, BiCGSTAB, etc. — particularly useful for block preconditioning or as a coarse-grid solver in multigrid.

---

## 6. Design Principles

### Consistent with MTL5 Patterns
- **Header-only** — all implementations in `.hpp`
- **C++20 concepts** — `SparseDirectSolver`, `FillReducingOrdering`, `SymbolicAnalysis`
- **`if constexpr` dispatch** — sparse vs. dense, native vs. external
- **RAII** — factorizations own their memory, movable but non-copyable
- **Template generality** — works with any `Scalar` value type, not just double

### CSparse-Inspired Algorithmic Choices
- **Left-looking** algorithms (not multifrontal) — simpler, more educational, good baseline performance
- **Gilbert-Peierls** sparse triangular solve — time proportional to output, the theoretically optimal approach
- **Symbolic/numeric separation** — the hallmark of all serious sparse direct solvers

### Pragmatic Layering
1. **Native implementations** for correctness, portability, and education
2. **External interfaces** for production performance (UMFPACK, SuperLU, KLU, CHOLMOD, SPQR)
3. **Automatic dispatch** so users get the best available solver transparently

---

## 7. Recommended Implementation Order

| Priority | Component | Rationale |
|----------|-----------|-----------|
| 1 | Permutation utilities | Everything else needs them |
| 2 | CSC adapter | Native format for column-based algorithms |
| 3 | Elimination tree + postorder | Foundation for all symbolic analysis |
| 4 | Sparse triangular solve (reach) | Core routine used everywhere |
| 5 | RCM ordering | Simple ordering to validate infrastructure |
| 6 | Sparse Cholesky (symbolic + numeric) | Simplest factorization, no pivoting |
| 7 | AMD ordering | Production-quality ordering for Cholesky |
| 8 | Sparse LU (symbolic + numeric) | Adds pivoting complexity |
| 9 | COLAMD ordering | Production ordering for LU/QR |
| 10 | Sparse QR (symbolic + numeric) | Adds Householder complexity |
| 11 | Dulmage-Mendelsohn / BTF | Advanced infrastructure for KLU-style |
| 12 | External interfaces (KLU, SuperLU, CHOLMOD, SPQR) | Production dispatching |
| 13 | Unified dispatch in `operation/` | Seamless user experience |

---

## 8. Testing Strategy

```
tests/unit/sparse/
├── test_permutation.cpp
├── test_csc.cpp
├── test_elimination_tree.cpp
├── test_amd.cpp
├── test_colamd.cpp
├── test_sparse_cholesky.cpp
├── test_sparse_lu.cpp
├── test_sparse_qr.cpp
├── test_dulmage_mendelsohn.cpp
└── test_sparse_dispatch.cpp       # verifies auto-dispatch logic
tests/unit/interface/
├── test_superlu.cpp               # conditional on MTL5_HAS_SUPERLU
├── test_klu.cpp                   # conditional on MTL5_HAS_KLU
├── test_cholmod.cpp               # conditional on MTL5_HAS_CHOLMOD
└── test_spqr.cpp                  # conditional on MTL5_HAS_SPQR
```

Test matrices should include:
- Small hand-verified examples (3x3 to 10x10)
- Diagonal, tridiagonal, arrow matrices (known structure)
- Matrices from Matrix Market for regression (e.g., `bcsstk01`, `west0067`)
- Randomly generated SPD matrices for Cholesky
- Rectangular matrices for QR

---

## References

- Davis, Timothy A. *Direct Methods for Sparse Linear Systems*. SIAM, 2006. (Fundamentals of Algorithms)
- Greenbaum, Anne. *Iterative Methods for Solving Linear Systems*. SIAM, 1997. (Frontiers in Applied Mathematics)
- Gilbert, John R. and Peierls, Tim. "Sparse Partial Pivoting in Time Proportional to Arithmetic Operations." *SIAM J. Sci. Stat. Comput.*, 9(5), 1988.
- Liu, Joseph W.H. "The Role of Elimination Trees in Sparse Factorization." *SIAM J. Matrix Anal. Appl.*, 11(1), 1990.
- Demmel, James W. et al. "A Supernodal Approach to Sparse Partial Pivoting." *SIAM J. Matrix Anal. Appl.*, 20(3), 1999. (SuperLU)
- Davis, Timothy A. and Palamadai Natarajan, Ekanathan. "Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems." *ACM Trans. Math. Softw.*, 37(3), 2010.
- Davis, Timothy A. "Algorithm 832: UMFPACK V4.3 — an unsymmetric-pattern multifrontal method." *ACM Trans. Math. Softw.*, 30(2), 2004.
- Chen, Yanqing, Davis, Timothy A., Hager, William W., and Rajamanickam, Sivasankaran. "Algorithm 887: CHOLMOD, Supernodal Sparse Cholesky Factorization and Update/Downdate." *ACM Trans. Math. Softw.*, 35(3), 2008.
- Davis, Timothy A. "Algorithm 915: SuiteSparseQR, a multifrontal multithreaded rank-revealing sparse QR factorization method." *ACM Trans. Math. Softw.*, 38(1), 2011.
