# Pedagogical examples 

Based on my deep knowledge of the phases implemented and the numerical methods literature:
Here are 2 pedagogical example proposals per phase (Phases 3-7). Each is grounded in a real 
numerical methods problem that showcases the MTL5 functionality introduced in that phase.

---
## Phase 3: Basic Iterative Solvers (CG, BiCGSTAB, Iteration Control, Preconditioners)

### Example 3A: Heat Equation on a 1D Rod (Steady-State)

Numerical goal: Solve the 1D Poisson equation -u''(x) = f(x) on [0,1] with Dirichlet boundary conditions using finite differences. 
This produces a symmetric positive-definite (SPD) tridiagonal system Au = b.

Why pedagogical: This is the "hello world" of PDE numerical methods — every numerical analysis textbook (Trefethen & Bau, LeVeque) 
starts here. Students see how continuous math becomes Ax = b, and why iterative solvers matter (direct methods are O(n^3), CG is O(n*sqrt(kappa))).

MTL5 features demonstrated:

  - dense2D construction of the discretization matrix
  - dense_vector for solution and RHS
  - itl::cg() with noisy_iteration to watch convergence
  - itl::pc::diagonal preconditioner vs itl::pc::identity (no preconditioner)
  - Comparison: how many iterations with/without diagonal preconditioning

### Example 3B: Convection-Diffusion: When Symmetry Breaks

Numerical goal: Solve -epsilon * u''(x) + u'(x) = f(x) (convection-diffusion) with central differences. 
The upwind convection term makes the matrix non-symmetric, so CG cannot be used — this motivates BiCGSTAB.

Why pedagogical: Demonstrates why we need multiple solvers. Students first try CG (it fails or diverges), 
then switch to BiCGSTAB. Shows iteration control: 

  - basic_iteration for silent runs, 
  - noisy_iteration for debugging, 
  - cyclic_iteration for periodic residual checks.

MTL5 features demonstrated:

  - Non-symmetric dense system construction
  - itl::cg() fails (or converges erratically) — demonstrates solver applicability limits
  - itl::bicgstab() succeeds on the same problem
  - All three iteration controllers: basic_iteration, noisy_iteration, cyclic_iteration
  - itl::pc::diagonal preconditioner

  ---
## Phase 4: Sparse Matrices, Inserter, GMRES, BiCG, Smoothers

### Example 4A: 2D Laplacian on a Grid (Sparse Assembly + GMRES)

  Numerical goal: Solve the 2D Poisson equation -Delta u = f on the unit square using a 5-point stencil finite difference scheme. This produces a large,
  sparse, banded SPD system. Assembly via the inserter pattern demonstrates real-world sparse matrix construction.

  Why pedagogical: The 2D Laplacian is the canonical sparse systems benchmark (Saad, "Iterative Methods for Sparse Linear Systems"). Students see how the
  inserter builds CRS row-by-row, why sparse storage matters (a 100x100 grid = 10,000 unknowns but only ~50,000 nonzeros vs 10^8 dense entries), and how
  GMRES handles this.

  MTL5 features demonstrated:
  - compressed2D with inserter + update_store for 5-point stencil assembly
  - Grid-to-linear index mapping
  - itl::gmres() with restart parameter (compare restart=10 vs restart=30)
  - itl::cg() as comparison (valid since the 2D Laplacian is SPD)
  - itl::pc::diagonal preconditioning
  - noisy_iteration showing convergence history

### Example 4B: Stationary Iteration: Smoothers as Standalone Solvers

  Numerical goal: Solve a small diagonally-dominant system using classical iterative methods (Jacobi, Gauss-Seidel, SOR) and study convergence rates.
  Demonstrates the spectral radius concept: Jacobi < Gauss-Seidel < SOR(optimal omega).

  Why pedagogical: Directly from Golub & Van Loan Chapter 10 and Saad Chapter 4. Students see that smoothers are solvers (just slow ones), motivating why
  they're used as building blocks in multigrid. The SOR omega sweep is a classic homework exercise made interactive.

  MTL5 features demonstrated:
  - compressed2D with inserter for a diagonally-dominant sparse system
  - itl::smoother::jacobi, itl::smoother::gauss_seidel, itl::smoother::sor
  - Residual reduction tracking over multiple sweeps
  - SOR with varying omega (0.5, 1.0, 1.2, 1.5, optimal) — tabulated convergence comparison
  - Sparse vs dense specialization (same API, different performance)

  ---
### Phase 5: Direct Factorizations (LU, QR, LQ, Cholesky, Triangular Solvers, ILU/IC)

  Example 5A: Least Squares Curve Fitting via QR

  Numerical goal: Fit a polynomial to noisy data points using the overdetermined system V * c = y where V is a Vandermonde matrix. Compare normal equations
  (V^T V c = V^T y via Cholesky) vs QR factorization — demonstrating why QR is numerically superior.

  Why pedagogical: This is Trefethen & Bau Lecture 11 and Golub & Van Loan Section 5.3. The Vandermonde matrix is notoriously ill-conditioned, making it a
  perfect stress test. Students see the normal equations give wrong answers at high polynomial degree while QR remains stable.

  MTL5 features demonstrated:
  - dense2D Vandermonde matrix construction
  - qr_factor() + qr_extract_Q/R() for QR factorization
  - cholesky() for normal equations approach
  - upper_trisolve() / lower_trisolve() for back-substitution
  - trans() view for V^T * V
  - Side-by-side numerical comparison showing conditioning effects

### Example 5B: Solving Ax=b Three Ways: LU vs Cholesky vs QR

  Numerical goal: Given a well-conditioned SPD system, solve it using LU (general), Cholesky (SPD-exploiting), and QR (orthogonal). Compare operation counts,
   accuracy, and applicability. Then show a non-SPD system where Cholesky fails but LU/QR work.

  Why pedagogical: Directly addresses the "which factorization when?" question from Golub & Van Loan Chapter 3-5. Students build intuition for the
  factorization decision tree that every practicing numerical analyst uses.

  MTL5 features demonstrated:
  - lu_factor() + lu_solve() (with partial pivoting)
  - cholesky() + triangular solvers
  - qr_factor() + qr_solve()
  - inv() as the "never do this in practice" baseline
  - ILU(0) and IC(0) as approximate factorizations for preconditioning preview

---

## Phase 6: Eigenvalue Solvers, SVD, Kronecker, Advanced Krylov

### Example 6A: Vibrating String: Eigenvalues as Physical Frequencies

Numerical goal: Compute the natural frequencies of a vibrating string by finding 
eigenvalues of the 1D Laplacian matrix. The eigenvalues are known analytically: 

```text
lambda_k = 4 sin^2(k*pi/(2n)), 
```

providing exact verification. Then extend to a 2D membrane using the Kronecker product.

Why pedagogical: Strang's "Linear Algebra and Its Applications" Chapter 6 — eigenvalues as 
physical vibration modes is the most intuitive introduction to spectral methods. 
The Kronecker product for 2D is from Golub & Van Loan Section 4.8.

MTL5 features demonstrated:

  - eigenvalue_symmetric() on the tridiagonal Laplacian
  - Verification against analytical eigenvalues
  - kron() to build 2D Laplacian from 1D (I kron T + T kron I)
  - eigenvalue() (general) on non-symmetric perturbation showing complex eigenvalues
  - svd() to verify singular values equal absolute eigenvalues for symmetric matrices

### Example 6B: Principal Component Analysis (PCA) via SVD

Numerical goal: Perform PCA on a small dataset — compute SVD of the centered data matrix, 
extract principal components, and project data onto reduced dimensions. 
Compare with eigendecomposition of the covariance matrix.

Why pedagogical: PCA/SVD is ubiquitous in data science and appears in Trefethen & Bau Lecture 4-5. 
It makes the abstract SVD concrete: U gives sample coordinates, V gives feature loadings, 
Sigma gives variance explained. Two approaches (SVD of data vs eigendecomposition of covariance) 
connect Phase 5 and Phase 6.

MTL5 features demonstrated:

  - svd() — U, S, V decomposition
  - eigenvalue_symmetric() on X^T * X covariance matrix
  - Dense matrix arithmetic: centering, trans(), matrix multiply
  - Verification: U * diag(S) * V^T ≈ X
  - Truncated reconstruction showing information loss

---

## Phase 7: Sparse Formats, Views & I/O

### Example 7A: Sparse Format Shootout: CRS vs COO vs ELL

Numerical goal: Build the same sparse matrix (2D Laplacian) in three formats:

  - coordinate (COO for assembly), 
  - compressed (CRS for computation), and 
  - ELLPACK (ELL for GPU-like access patterns). 

Compare storage costs and access patterns. Then write/read the matrix via Matrix Market format.

Why pedagogical: Saad Chapter 3 "Sparse Matrix Computations" — understanding sparse formats is 
prerequisite to efficient large-scale computing. Students see that format choice depends on the 
operation (assembly → COO, matvec → CRS/ELL, I/O → Matrix Market).

MTL5 features demonstrated:

  - coordinate2D with insert(), sort(), compress() pipeline
  - compressed2D with inserter
  - ell_matrix constructed from compressed2D
  - io::mm_write_sparse() and io::mm_read() round-trip
  - Storage comparison: bytes per format for same matrix
  - Element access correctness verification across all three

### Example 7B: Structured Matrix Views: Exploiting Symmetry and Bandedness

Numerical goal: Start with a full dense matrix from a finite element assembly, 
then demonstrate how views reduce storage and computation. Show: 

  1. hermitian view stores only upper triangle, 
  2. banded view extracts tridiagonal structure, 
  3. map view extracts a subproblem. 

Apply CG to the hermitian-viewed matrix.

Why pedagogical: Views are a key abstraction in modern matrix libraries (MATLAB's triu, band; 
Julia's Symmetric, view). Students learn that you don't always need to copy/transform data — views 
provide zero-cost abstractions. Golub & Van Loan Section 1.2.7 on exploiting structure.

MTL5 features demonstrated:

  - hermitian() view of an upper-triangular stored matrix
  - banded() view with different bandwidth parameters
  - mapped() for submatrix extraction and row permutation
  - identity2D as implicit identity (no storage)
  - Combining views with solvers: CG on hermitian_view
  - Matrix Market I/O to save/load results

