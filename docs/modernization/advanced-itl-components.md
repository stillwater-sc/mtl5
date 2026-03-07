# Phase 13: Advanced ITL Components

## Context

Phases 1-12 are complete (65 tests passing). MTL5 already has 7 Krylov solvers (CG, BiCG, BiCGSTAB, GMRES, TFQMR, QMR, IDR(s)), 5 preconditioners (identity, diagonal, solver, ILU(0), IC(0)), and 3 smoothers (Gauss-Seidel, Jacobi, SOR). Phase 13 adds the remaining ITL components: 3 additional Krylov solvers, 4 advanced preconditioners, and a multigrid framework.

Key findings from exploration:
- **CGS** exists in MTL4 — straightforward port (65 lines of algorithm)
- **BiCGSTAB(ell)** exists in MTL4 — complex, uses left+right preconditioners and MR orthogonalization
- **MINRES** does NOT exist in MTL4 — must be implemented from scratch (Lanczos-based)
- **ILUT** exists in MTL4 but marked "Not usable yet" — needs clean reimplementation
- **ILDL, block_diagonal, SSOR** do NOT exist in MTL4 — from scratch

## Canonical Patterns (from existing MTL5 code)

**Krylov solver** (`include/mtl/itl/krylov/bicgstab.hpp`):
```cpp
template <typename LinearOp, typename VecX, typename VecB, typename PC, typename Iter>
int solver(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();
    vec::dense_vector<value_type> r(n), ...;  // workspace
    // r = b - A*x (explicit loop to materialize)
    while (!iter.finished(r)) { ++iter; M.solve(z, r); auto Ap = A * p; ... }
    return iter;
}
```

**Preconditioner** (`include/mtl/itl/pc/ilu_0.hpp`):
```cpp
template <typename Value, typename Parameters = mat::parameters<>>
class pc_name {
    using matrix_type = mat::compressed2D<Value, Parameters>;
public:
    explicit pc_name(const matrix_type& A);  // factorize in constructor
    void solve(VecX& x, const VecB& b) const;  // forward+back substitution
    void adjoint_solve(VecX& x, const VecB& b) const;  // typically same as solve
};
```

**Smoother** (`include/mtl/itl/smoother/sor.hpp`):
- Generic version + compressed2D specialization
- `operator()(VecX& x, const VecB& b)` interface

## 13.1 Additional Krylov Variants (3 solvers)

### CGS — Conjugate Gradient Squared
- **Source**: Port from MTL4 `boost/numeric/itl/krylov/cgs.hpp`
- **Algorithm**: Shadow residual `rtilde = r`, uses `dot(rtilde, r)` for `rho`, breakdown on `rho == 0`
- **Key steps**: `p = u + beta*(q + beta*p)`, `vhat = A * M^{-1} p`, `q = u - alpha*vhat`, `u += q`, `uhat = M^{-1} u`, `x += alpha*uhat`, `r -= alpha*A*uhat`
- **File**: `include/mtl/itl/krylov/cgs.hpp`

### BiCGSTAB(ell) — BiCGSTAB with higher-order stabilization
- **Source**: Port from MTL4 `boost/numeric/itl/krylov/bicgstab_ell.hpp`
- **Algorithm**: Generalizes BiCGSTAB with ell > 1 for better convergence on stiff problems
- **Adaptation**: MTL4 uses left+right PCs. MTL5 will use **single PC** (consistent with all other MTL5 solvers). Apply as right PC: `M.solve(z, v); w = A * z`
- **Workspace**: `std::vector<dense_vector<value_type>>` for `r_hat[0..ell]`, `u_hat[0..ell]`, plus `tau` (dense2D), `sigma`, `gamma`, `gamma_a`, `gamma_aa` (dense_vectors of Scalar)
- **MR orthogonalization part**: Factorize residual polynomial minimization using modified Gram-Schmidt
- **File**: `include/mtl/itl/krylov/bicgstab_ell.hpp`

### MINRES — Minimum Residual for symmetric indefinite systems
- **Source**: From scratch (not in MTL4)
- **Algorithm**: Lanczos-based 3-term recurrence. Key property: monotonically decreasing residual norm for symmetric A (not necessarily SPD). Uses Givens rotations to maintain QR of tridiagonal Lanczos matrix.
- **Steps per iteration**:
  1. Lanczos step: `v_{k+1} = A*v_k - alpha_k*v_k - beta_k*v_{k-1}`
  2. Apply previous Givens rotations to new column
  3. Compute new Givens rotation to zero sub-diagonal
  4. Update solution: `x += d_k * tau_k`
- **Workspace**: 3 Lanczos vectors (`v`, `v_old`, `v_new`), 3 direction vectors (`d`, `d_old`, `d_older`), plus scalars for Givens rotation state (`cs`, `sn`, `beta`, `alpha`)
- **File**: `include/mtl/itl/krylov/minres.hpp`

## 13.2 Advanced Preconditioners (4 preconditioners)

### ILUT — ILU with Threshold and Fill Limiting
- **Source**: Clean reimplementation (MTL4's was marked "Not usable yet")
- **Algorithm**: Saad's ILUT — like ILU(0) but allows fill-in up to `p` entries per row and drops entries smaller than `tau * ||row||_2`
- **Constructor**: `ilut(const matrix_type& A, size_type fill = 10, value_type threshold = 1e-4)`
- **Factorization**: Row-by-row IKJ Gaussian elimination with:
  - Dense work row for accumulation
  - Drop entries below `tau * ||original_row||_2`
  - Keep at most `fill` largest entries in each of L and U parts
  - Sort by magnitude before truncation
- **Storage**: Same CRS L/U format as ILU(0)
- **File**: `include/mtl/itl/pc/ilut.hpp`

### ILDL — Incomplete LDL^T for Symmetric Systems
- **Source**: From scratch
- **Algorithm**: Incomplete factorization A ≈ L*D*L^T preserving sparsity pattern. D is diagonal (not necessarily positive), L is unit lower triangular. Works for symmetric indefinite systems.
- **Constructor**: `ildl(const matrix_type& A)` — A must be symmetric
- **Factorization**: Row-by-row: `d_i = a_ii - sum_{k<i} l_ik^2 * d_k`, then `l_ij = (a_ij - sum_{k<j} l_ik * d_k * l_jk) / d_j` at sparsity positions
- **Solve**: Forward L, diagonal D, backward L^T
- **File**: `include/mtl/itl/pc/ildl.hpp`

### Block Diagonal Preconditioner
- **Source**: From scratch
- **Algorithm**: Partition matrix into `nb` diagonal blocks of size `bs = n/nb`. Extract each block as a dense sub-matrix, compute its LU factorization, store. solve() applies each block's LU solve independently.
- **Constructor**: `block_diagonal(const Matrix& A, size_type block_size)` — extracts diagonal blocks
- **Storage**: Vector of dense blocks + their LU factorizations
- **File**: `include/mtl/itl/pc/block_diagonal.hpp`

### SSOR — Symmetric SOR Preconditioner
- **Source**: From scratch
- **Algorithm**: Forward SOR sweep followed by backward SOR sweep. Equivalent to `(D + omega*L) * D^{-1} * (D + omega*U)` preconditioning where L,U are strictly lower/upper parts.
- **Constructor**: `ssor(const matrix_type& A, value_type omega = 1.0)`
- **Solve**: Forward sweep (row 0 to n-1), then backward sweep (row n-1 to 0), each with SOR relaxation. Both generic and compressed2D specialization.
- **File**: `include/mtl/itl/pc/ssor.hpp`

## 13.3 Multigrid Framework

### Architecture
```
itl/mg/
  multigrid.hpp      — V-cycle / W-cycle driver
  restriction.hpp    — Fine → coarse transfer (full weighting)
  prolongation.hpp   — Coarse → fine transfer (linear interpolation)
```

### Multigrid Cycle (`multigrid.hpp`)
- **Template parameters**: `Smoother`, `CoarseSolver`, `Restrictor`, `Prolongator`
- **Class**: `multigrid<Smoother, CoarseSolver, Restrictor, Prolongator>`
- **Constructor**: Takes vector of level matrices `A[0..L-1]`, smoother factory, coarse solver, restrictor, prolongator, pre/post smoothing counts
- **V-cycle**: Recursive — pre-smooth, restrict residual, recurse, prolongate correction, post-smooth
- **W-cycle**: Like V-cycle but recurse twice at each level
- **Interface**: `void vcycle(VecX& x, const VecB& b, int level)` + `operator()(VecX& x, const VecB& b)` for use as a preconditioner
- **solve/adjoint_solve**: To match PC interface so multigrid can be used as a preconditioner for Krylov solvers

### Restriction (`restriction.hpp`)
- **Full weighting** for 1D: `[1/4, 1/2, 1/4]` stencil
- Stored as a sparse matrix (compressed2D)
- `restrict(const Vector& fine) -> Vector coarse`

### Prolongation (`prolongation.hpp`)
- **Linear interpolation** for 1D: coarse points copied, fine points averaged from neighbors
- Stored as a sparse matrix (compressed2D)
- `prolongate(const Vector& coarse) -> Vector fine`
- Transpose relationship: P = 2 * R^T (standard for geometric multigrid)

## Files to Create (10 new headers)

### Krylov solvers (3 in `include/mtl/itl/krylov/`)
- `cgs.hpp`
- `bicgstab_ell.hpp`
- `minres.hpp`

### Preconditioners (4 in `include/mtl/itl/pc/`)
- `ilut.hpp`
- `ildl.hpp`
- `block_diagonal.hpp`
- `ssor.hpp`

### Multigrid (3 in `include/mtl/itl/mg/`)
- `multigrid.hpp`
- `restriction.hpp`
- `prolongation.hpp`

## Files to Modify (1)

- `include/mtl/itl/itl.hpp` — Add includes for all 10 new headers

## Tests (4 new test files in `tests/unit/itl/`)

### `test_cgs.cpp`
- CGS on tridiagonal SPD system (should converge)
- CGS with diagonal PC converges faster than identity
- Verify Ax ≈ b at solution

### `test_bicgstab_ell.cpp`
- BiCGSTAB(2) on tridiagonal system
- BiCGSTAB(4) converges in fewer iterations than BiCGSTAB(2)
- Verify Ax ≈ b at solution

### `test_advanced_pc.cpp`
- ILUT: preconditioned BiCGSTAB converges, Ax ≈ b
- ILUT: fewer iterations than identity PC
- ILDL: preconditioned CG on SPD system converges, Ax ≈ b
- Block diagonal: preconditioned BiCGSTAB converges
- SSOR: preconditioned BiCGSTAB converges, fewer iterations than identity

### `test_multigrid.cpp`
- Restriction: size halving, correct stencil values
- Prolongation: size doubling, correct interpolation
- V-cycle on 1D Poisson with Gauss-Seidel smoother: converges
- Multigrid as preconditioner with CG: converges

## Examples (2 in `examples/`)

### `phase13a_krylov_comparison.cpp`
Solve a non-symmetric sparse system with CGS, BiCGSTAB, BiCGSTAB(2), and MINRES (on symmetric version). Compare iteration counts and demonstrate convergence behavior.

### `phase13b_multigrid_poisson.cpp`
Solve 1D Poisson equation using V-cycle multigrid. Show convergence rates, demonstrate multigrid as a stand-alone solver and as a preconditioner for CG.

## Implementation Order

1. **CGS solver** + test — simplest, direct port from MTL4
2. **SSOR preconditioner** — straightforward symmetric SOR sweep
3. **ILUT preconditioner** — extends ILU(0) pattern with threshold/fill
4. **ILDL preconditioner** — LDL^T variant of IC(0) pattern
5. **Block diagonal preconditioner** — uses dense LU on diagonal blocks
6. **Advanced PC test** (`test_advanced_pc.cpp`)
7. **BiCGSTAB(ell) solver** + test — most complex Krylov variant
8. **MINRES solver** + test — from-scratch Lanczos implementation
9. **Multigrid framework** (restriction, prolongation, multigrid) + test
10. Update `itl.hpp` umbrella
11. Examples 13a + 13b
12. Full build + full test suite

## Cross-Platform Notes

- All standard C++ only — no POSIX/GNU extensions
- `std::numbers::pi` for constants, never `M_PI`
- `std::size_t` for sizes, `std::filesystem` for paths
- No hardcoded `/tmp/` paths in tests

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake -B build && cmake --build build -j$(nproc) && ctest --test-dir build
```
