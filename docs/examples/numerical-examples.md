# Plan: 10 Pedagogical Examples for MTL5 (Phases 3-7)

## Context

Phases 3-7 of the MTL5 modernization are complete with 31/31 tests passing. The user wants 2 educational examples per phase (10 total) that demonstrate the numerical functionality with deep pedagogical value — showing not just "how" but "why" each algorithm exists and when to use it.

All examples will be standalone `.cpp` files in `examples/`, each printing results to stdout with commentary. They use `#include <mtl/mtl.hpp>` as the single umbrella header.

## Files Created (10 examples + CMake update)

| File | Phase | Title |
|------|-------|-------|
| `examples/phase3a_heat_equation_1d.cpp` | 3 | 1D Heat Equation — CG with preconditioning |
| `examples/phase3b_convection_diffusion.cpp` | 3 | Convection-Diffusion — CG vs BiCGSTAB |
| `examples/phase4a_laplacian_2d.cpp` | 4 | 2D Laplacian — Sparse assembly + GMRES |
| `examples/phase4b_smoother_convergence.cpp` | 4 | Stationary Iterative Methods — Jacobi/GS/SOR |
| `examples/phase5a_least_squares_qr.cpp` | 5 | Least Squares Curve Fitting — QR vs Normal Equations |
| `examples/phase5b_solve_three_ways.cpp` | 5 | Solving Ax=b — LU vs Cholesky vs QR |
| `examples/phase6a_vibrating_string.cpp` | 6 | Vibrating String — Eigenvalues + Kronecker |
| `examples/phase6b_pca_svd.cpp` | 6 | PCA via SVD — Data dimensionality reduction |
| `examples/phase7a_sparse_formats.cpp` | 7 | Sparse Format Shootout — COO vs CRS vs ELL |
| `examples/phase7b_structured_views.cpp` | 7 | Structured Matrix Views — Hermitian/Banded/Map |
| `examples/CMakeLists.txt` | — | Register all 11 executables |

## Implementation Details

### Example 3A: `phase3a_heat_equation_1d.cpp`
**Goal**: Solve steady-state 1D heat equation via finite differences → SPD tridiagonal system.

**Content**:
1. Build n×n tridiagonal dense matrix: A(i,i)=2, A(i,i±1)=-1 (1D Laplacian)
2. RHS b from known solution u(x)=sin(πx) → f(x)=π²sin(πx)
3. Solve with `itl::cg()` + `itl::pc::identity` (no PC), print iterations via `noisy_iteration`
4. Solve with `itl::cg()` + `itl::pc::diagonal` (Jacobi PC), compare iteration count
5. Print solution vs analytical, compute max error
6. Commentary: why CG works (SPD), what preconditioning does, iteration vs accuracy

**Key APIs**: `mat::dense2D<double>`, `vec::dense_vector<double>`, `itl::cg`, `itl::noisy_iteration`, `itl::pc::identity`, `itl::pc::diagonal`

### Example 3B: `phase3b_convection_diffusion.cpp`
**Goal**: Non-symmetric convection-diffusion → CG fails, BiCGSTAB succeeds.

**Content**:
1. Build non-symmetric matrix: -ε·u'' + u' = f, central differences
2. Show two regimes: ε=1.0 (diffusion-dominated, nearly symmetric) and ε=0.01 (convection-dominated)
3. Try `itl::cg()` on non-symmetric system → fails to converge (error code 1)
4. Solve with `itl::bicgstab()` → converges
5. Compare three iteration controllers: `basic_iteration` (silent), `cyclic_iteration` (every 10), `noisy_iteration` (every step)
6. Commentary: why solver choice depends on matrix properties

**Key APIs**: `itl::cg`, `itl::bicgstab`, `itl::basic_iteration`, `itl::cyclic_iteration`, `itl::noisy_iteration`, `itl::pc::diagonal`

### Example 4A: `phase4a_laplacian_2d.cpp`
**Goal**: Assemble 2D Laplacian with sparse inserter, solve with GMRES.

**Content**:
1. Create n²×n² `compressed2D` using `inserter` with 5-point stencil
2. Grid mapping: (ix,iy) → ix + iy*n
3. RHS from f(x,y)=2π²sin(πx)sin(πy), known solution u(x,y)=sin(πx)sin(πy)
4. Solve with `itl::gmres()` (restart=10 and restart=30), compare iteration counts
5. Solve with `itl::cg()` for comparison (valid since 2D Laplacian is SPD)
6. Show `noisy_iteration` convergence curve
7. Commentary: sparse assembly pattern, restart tradeoffs, CG vs GMRES

**Key APIs**: `mat::compressed2D<double>`, `mat::inserter<mat::compressed2D<double>>`, `itl::gmres`, `itl::cg`, `itl::noisy_iteration`, `itl::pc::diagonal`

### Example 4B: `phase4b_smoother_convergence.cpp`
**Goal**: Compare Jacobi, Gauss-Seidel, SOR convergence rates on same system.

**Content**:
1. Build diagonally-dominant sparse system (1D Poisson, n=50)
2. Run 100 sweeps each of Jacobi, GS, SOR(ω) for ω ∈ {0.5, 1.0, 1.2, 1.5, 1.8}
3. Track ‖b - Ax‖₂ / ‖b‖₂ after each sweep
4. Print convergence table: iteration vs residual for each method
5. Highlight that SOR(ω=1.0) = GS, and optimal ω ≈ 2/(1+sin(πh)) for 1D Poisson
6. Commentary: spectral radius, why smoothers are slow but useful in multigrid

**Key APIs**: `mat::compressed2D<double>`, `mat::inserter`, `itl::smoother::jacobi`, `itl::smoother::gauss_seidel`, `itl::smoother::sor`, `two_norm`

### Example 5A: `phase5a_least_squares_qr.cpp`
**Goal**: Polynomial curve fitting — QR vs normal equations for ill-conditioned Vandermonde.

**Content**:
1. Generate noisy data: y_i = sin(x_i) + noise, x_i ∈ [0, π]
2. Build Vandermonde matrix V(i,j) = x_i^j for degree p
3. **Normal equations**: Cholesky on V^T*V — works for low degree, fails for high degree
4. **QR approach**: `qr_factor(V, tau)` + `qr_solve(V, tau, c, y)` — stable at any degree
5. Compare fitted coefficients and residuals at degree 3, 5, 8, 12
6. Show condition number of V^T*V growing catastrophically
7. Commentary: why normal equations are numerically inferior, Trefethen & Bau perspective

**Key APIs**: `mat::dense2D<double>`, `qr_factor`, `qr_solve`, `qr_extract_Q`, `qr_extract_R`, `cholesky_factor`, `cholesky_solve`, `trans`, matrix multiply `*`

### Example 5B: `phase5b_solve_three_ways.cpp`
**Goal**: Solve same system via LU, Cholesky, QR — when to use which.

**Content**:
1. **SPD system**: Build 4×4 Hilbert-like SPD matrix
   - LU: `lu_factor` + `lu_solve` — works, most general
   - Cholesky: `cholesky_factor` + `cholesky_solve` — works, 2× faster
   - QR: `qr_factor` + `qr_solve` — works, most stable
   - `inv(A) * b` — works but "never do this in practice" baseline
2. **Non-SPD system**: Perturb matrix to be non-symmetric
   - LU: works
   - Cholesky: fails (returns error code > 0)
   - QR: works
3. **Preconditioner preview**: Show ILU(0) and IC(0) on sparse SPD system
4. Print solutions, residuals, and factorization return codes
5. Commentary: decision tree for choosing factorization

**Key APIs**: `lu_factor`, `lu_solve`, `cholesky_factor`, `cholesky_solve`, `qr_factor`, `qr_solve`, `inv`, `itl::pc::ilu_0`, `itl::pc::ic_0`, `two_norm`

### Example 6A: `phase6a_vibrating_string.cpp`
**Goal**: Eigenvalues of 1D Laplacian = vibration frequencies, extend to 2D via Kronecker.

**Content**:
1. Build n×n 1D Laplacian (tridiagonal: 2,-1,-1)
2. Compute eigenvalues with `eigenvalue_symmetric()`
3. Compare with analytical: λ_k = 4·sin²(kπ/(2(n+1)))
4. Build 2D Laplacian via `kron(I, T) + kron(T, I)` where I = identity, T = 1D Laplacian
5. Compute 2D eigenvalues with `eigenvalue_symmetric()`
6. Verify: 2D eigenvalues = λ_i + λ_j for all pairs (i,j) from 1D
7. Show `eigenvalue()` (general) on a non-symmetric perturbation → complex eigenvalues
8. Verify SVD singular values = |eigenvalues| for symmetric case
9. Commentary: physical interpretation, Kronecker product structure, spectral methods

**Key APIs**: `mat::dense2D<double>`, `eigenvalue_symmetric`, `eigenvalue`, `kron`, `svd`, `mat::identity2D<double>`

### Example 6B: `phase6b_pca_svd.cpp`
**Goal**: PCA on synthetic dataset via SVD, compare with covariance eigendecomposition.

**Content**:
1. Generate 8×3 data matrix (8 samples, 3 features) with known correlation structure
2. Center the data: X_centered = X - column_means
3. **SVD approach**: `svd(X_centered, U, S, V)` → principal components = columns of V
4. **Covariance approach**: C = X^T * X / (n-1), `eigenvalue_symmetric(C)` → same eigenvalues as S²/(n-1)
5. Project data onto first 2 principal components: X_reduced = X_centered * V[:, :2]
6. Reconstruct from 2 components: X_approx = X_reduced * V[:, :2]^T + means
7. Compute reconstruction error ‖X - X_approx‖_F
8. Show variance explained by each component: σ_i² / Σ σ_j²
9. Commentary: SVD is the numerically stable way to do PCA, connection to eigenvalues

**Key APIs**: `svd` (both forms), `eigenvalue_symmetric`, `trans`, matrix multiply, `two_norm`

### Example 7A: `phase7a_sparse_formats.cpp`
**Goal**: Build same matrix in COO/CRS/ELL, compare storage, round-trip via Matrix Market.

**Content**:
1. Build 5-point 2D Laplacian (10×10 grid = 100 unknowns)
2. **COO assembly**: `coordinate2D` with `insert()`, then `sort()`, `compress()` → CRS
3. **CRS assembly**: `compressed2D` with `inserter`
4. **ELL conversion**: `ell_matrix` from CRS
5. Verify all three give identical element access results
6. Print storage comparison: COO (3 arrays × nnz), CRS (2 arrays × nnz + starts), ELL (2 arrays × nrows × width)
7. **Matrix Market round-trip**: `io::mm_write_sparse` → `io::mm_read` → verify
8. `io::mm_write` dense version → `io::mm_read_dense` → verify
9. Commentary: when to use each format (assembly → COO, computation → CRS, GPU → ELL)

**Key APIs**: `mat::coordinate2D`, `mat::compressed2D`, `mat::ell_matrix`, `mat::inserter`, `io::mm_write_sparse`, `io::mm_write`, `io::mm_read`, `io::mm_read_dense`

### Example 7B: `phase7b_structured_views.cpp`
**Goal**: Exploit symmetry, bandedness, and substructure via zero-copy views.

**Content**:
1. Build 6×6 dense SPD matrix, store only upper triangle
2. **Hermitian view**: `hermitian(A)` mirrors lower from upper with conjugation
3. Verify H(i,j) == H(j,i) for all entries
4. **Banded view**: Extract tridiagonal `banded(A, 1, 1)`, diagonal-only `banded(A, 0, 0)`
5. Show outside-band elements are zero
6. **Map view**: Extract 3×3 submatrix via `mapped(A, rows, cols)`
7. Permute rows: reverse order via map
8. **Identity matrix**: `identity2D(n)` — no storage, implicit
9. Solve Hx = b with CG on the hermitian view (works because hermitian_view satisfies Matrix concept)
10. **Matrix Market I/O**: Write original + read back + compare
11. Commentary: views as zero-cost abstractions, exploiting structure for storage savings

**Key APIs**: `hermitian`, `banded`, `mapped`, `mat::identity2D`, `itl::cg`, `io::mm_write`, `io::mm_read_dense`

## CMakeLists.txt Update

`examples/CMakeLists.txt` — register all 11 executables:
```cmake
add_executable(hello_mtl5 hello_mtl5.cpp)
target_link_libraries(hello_mtl5 PRIVATE MTL5::mtl5)

# Phase 3 examples
add_executable(phase3a_heat_equation_1d phase3a_heat_equation_1d.cpp)
target_link_libraries(phase3a_heat_equation_1d PRIVATE MTL5::mtl5)

add_executable(phase3b_convection_diffusion phase3b_convection_diffusion.cpp)
target_link_libraries(phase3b_convection_diffusion PRIVATE MTL5::mtl5)

# Phase 4 examples
add_executable(phase4a_laplacian_2d phase4a_laplacian_2d.cpp)
target_link_libraries(phase4a_laplacian_2d PRIVATE MTL5::mtl5)

add_executable(phase4b_smoother_convergence phase4b_smoother_convergence.cpp)
target_link_libraries(phase4b_smoother_convergence PRIVATE MTL5::mtl5)

# Phase 5 examples
add_executable(phase5a_least_squares_qr phase5a_least_squares_qr.cpp)
target_link_libraries(phase5a_least_squares_qr PRIVATE MTL5::mtl5)

add_executable(phase5b_solve_three_ways phase5b_solve_three_ways.cpp)
target_link_libraries(phase5b_solve_three_ways PRIVATE MTL5::mtl5)

# Phase 6 examples
add_executable(phase6a_vibrating_string phase6a_vibrating_string.cpp)
target_link_libraries(phase6a_vibrating_string PRIVATE MTL5::mtl5)

add_executable(phase6b_pca_svd phase6b_pca_svd.cpp)
target_link_libraries(phase6b_pca_svd PRIVATE MTL5::mtl5)

# Phase 7 examples
add_executable(phase7a_sparse_formats phase7a_sparse_formats.cpp)
target_link_libraries(phase7a_sparse_formats PRIVATE MTL5::mtl5)

add_executable(phase7b_structured_views phase7b_structured_views.cpp)
target_link_libraries(phase7b_structured_views PRIVATE MTL5::mtl5)
```

## Implementation Order

Implement sequentially 3A → 3B → 4A → 4B → 5A → 5B → 6A → 6B → 7A → 7B, building complexity progressively. Update CMakeLists.txt at the start. Build and run each example after writing it.

## Key Patterns for All Examples

- `#include <mtl/mtl.hpp>` as sole library include
- `#include <iostream>`, `#include <cmath>`, `#include <iomanip>` for output
- `using namespace mtl;` for brevity in examples (educational context)
- Each example prints a header banner, section headers, and formatted numerical output
- Commentary as `std::cout` statements explaining what's happening and why
- Each example is self-contained and compiles independently
- Use `std::setw`, `std::setprecision`, `std::fixed`/`std::scientific` for clean output
- Target ~100-200 lines per example (enough for depth, not overwhelming)

## Verification

```bash
cmake -B build -DCMAKE_CXX_STANDARD=20
cmake --build build -j$(nproc)
# Run each example
./build/examples/phase3a_heat_equation_1d
./build/examples/phase3b_convection_diffusion
./build/examples/phase4a_laplacian_2d
./build/examples/phase4b_smoother_convergence
./build/examples/phase5a_least_squares_qr
./build/examples/phase5b_solve_three_ways
./build/examples/phase6a_vibrating_string
./build/examples/phase6b_pca_svd
./build/examples/phase7a_sparse_formats
./build/examples/phase7b_structured_views
# Also verify existing tests still pass
ctest --test-dir build --output-on-failure
```

Each example should compile cleanly (no warnings) and produce readable educational output.

  All 10 pedagogical examples + CMakeLists.txt created and verified:

|             Example              | Phase |                        Status                        |                 Key Demonstration                  |
|----------------------------------|-------|------------------------------------------------------|----------------------------------------------------|
| phase3a_heat_equation_1d.cpp     | 3     | Runs, 146→100 iters (1.5x speedup with Jacobi PC)    | Variable-coefficient heat eq, CG + preconditioning |
| phase3b_convection_diffusion.cpp | 3     | CG fails (500 iters), BiCGSTAB converges (30-65      | Non-symmetric systems, 3 iteration controllers     |
|                                  |       | iters)                                               |                                                    |
| phase4a_laplacian_2d.cpp         | 4     | GMRES(5)=85, GMRES(20)=15, CG=15 iterations          | Sparse inserter, GMRES restart tradeoff            |
| phase4b_smoother_convergence.cpp | 4     | SOR(optimal)=3e-4, Jacobi=0.75, GS=0.62 at 100       | Jacobi/GS/SOR convergence rates                    |
|                                  |       | sweeps                                               |                                                    |
| phase5a_least_squares_qr.cpp     | 5     | QR beats normal equations at degree 12 (cond=1.7e21) | Vandermonde ill-conditioning, QR stability         |
| phase5b_solve_three_ways.cpp     | 5     | LU/QR work, Cholesky fails on indefinite (code=2)    | LU vs Cholesky vs QR decision tree, ILU/IC preview |
| phase6a_vibrating_string.cpp     | 6     | 1D/2D eigenvalues match analytical to ~1e-16         | Kronecker product, complex eigenvalues, SVD        |
|                                  |       |                                                      | connection                                         |
| phase6b_pca_svd.cpp              | 6     | 97.2% variance in PC1, 99.9% with 2 components       | SVD-based PCA, covariance eigenvalue verification  |
| phase7a_sparse_formats.cpp       | 7     | COO→CRS→ELL verified identical, MM round-trip PASS   | Storage comparison, Matrix Market I/O              |
| phase7b_structured_views.cpp     | 7     | All views PASS, CG on hermitian view converges in 6  | Hermitian/banded/map views, identity matrix        |
|                                  |       | iters                                                |                                                    |

Build: All 10 examples compile cleanly with no warnings
Tests: All 31/31 existing tests pass
