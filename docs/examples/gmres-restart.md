# GMRES: The Generalized Minimum Residual Method

GMRES is the workhorse iterative solver for large, sparse, non-symmetric linear systems. It builds a sequence of improving approximations by searching over an expanding **Krylov subspace**, minimizing the residual norm at each step. Unlike CG (which requires a symmetric positive definite matrix), GMRES works for any nonsingular system.

This page explains why GMRES works, what the restart parameter controls, and how to use MTL5's implementation effectively -- including how to diagnose and fix the restart stalling problem.

## Why GMRES Exists

### The problem with direct solvers

For large sparse systems (n > 10,000), direct factorizations (LU, Cholesky) can be prohibitively expensive:

- **Fill-in**: L and U are typically much denser than A, requiring O(n^2) memory
- **Cost**: O(n^3) for dense, O(n * nnz) to O(n^2) for sparse depending on ordering
- **No early termination**: you pay the full cost even if you only need 3-digit accuracy

Iterative solvers avoid factorization entirely. They compute matrix-vector products (A * v) and converge to the solution incrementally, stopping as soon as the desired accuracy is reached.

### Why not just use CG?

The Conjugate Gradient method (CG) is optimal for symmetric positive definite (SPD) systems. But many important problems produce non-symmetric matrices:

- Convection-diffusion equations (fluid flow, heat transport)
- Non-symmetric discretizations (upwind schemes, DG methods)
- Preconditioned systems where the preconditioner destroys symmetry
- Control systems and Jacobians in Newton's method

GMRES handles all of these.

## How GMRES Works

### The Krylov subspace

Given a starting residual r = b - Ax_0, GMRES builds the **Krylov subspace**:

```text
K_m = span{ r, Ar, A^2 r, A^3 r, ..., A^(m-1) r }
```

At iteration m, this subspace has dimension m (assuming no lucky breakdown). GMRES finds the vector x_m in x_0 + K_m that minimizes ||b - Ax_m||.

**Intuition**: each multiplication by A "reveals" more information about A's action on the residual. After m steps, GMRES has explored m independent directions in the solution space.

### The Arnoldi process

GMRES doesn't work with the raw vectors {r, Ar, A^2 r, ...} because they quickly become nearly parallel (dominated by the eigenvector of the largest eigenvalue). Instead, it builds an **orthonormal** basis V_1, V_2, ..., V_m via the Arnoldi process:

```text
for k = 1, 2, ..., m:
    w = A * V_k                          # expand the subspace
    for j = 1 to k:                      # orthogonalize against all previous
        h_{j,k} = <V_j, w>
        w = w - h_{j,k} * V_j
    h_{k+1,k} = ||w||                    # normalize
    V_{k+1} = w / h_{k+1,k}
```

This produces an (m+1) x m upper Hessenberg matrix H such that A * V_m = V_{m+1} * H.

### The least-squares solve

The minimization ||b - Ax|| over x_0 + K_m reduces to a small (m+1) x m least-squares problem:

```text
minimize ||beta * e_1 - H * y||
```

where beta = ||r_0||. This is solved efficiently using Givens rotations as each column of H is produced -- no separate least-squares solve is needed at the end.

### Cost per iteration

| Operation | Cost |
|-----------|------|
| Matrix-vector product A * v | O(nnz) |
| Orthogonalization against k vectors | O(kn) |
| Givens rotation update | O(k) |
| **Total at iteration m** | **O(nnz + mn)** |

The critical point: orthogonalization cost grows linearly with the subspace dimension m. After m iterations, the total work is O(m * nnz + m^2 * n).

## The Restart Problem

### Why restart?

As the Krylov subspace grows, two costs accumulate:

- **Memory**: m vectors of length n must be stored (the Arnoldi basis V)
- **Orthogonalization**: each new vector must be orthogonalized against all m previous vectors

For m = 1000 on a system with n = 100,000:
- Memory: 1000 * 100,000 * 8 bytes = 800 MB just for the basis
- Orthogonalization: dominates the computation time

**Restarted GMRES** caps the subspace dimension at a parameter `restart` (typically 20-50). When m reaches `restart`:

1. Take the current best solution x_m
2. Discard all m basis vectors and the Hessenberg matrix
3. Compute the new residual r = b - Ax_m
4. Start a fresh GMRES cycle from x_m

### The stalling problem

The danger of restarting is **information loss**. If the solution requires components in Krylov directions beyond dimension `restart`, those directions are discarded and must be rediscovered in the next cycle. In pathological cases, each restart cycle rebuilds essentially the same subspace, and the residual stalls:

```text
restart = 10:
  Cycle 1: residual 1.0e+00 -> 3.2e-01 (good progress)
  Cycle 2: residual 3.2e-01 -> 1.1e-01 (progress slowing)
  Cycle 3: residual 1.1e-01 -> 9.8e-02 (stalled!)
  Cycle 4: residual 9.8e-02 -> 9.7e-02 (stuck)
  ...

restart = 50:
  Cycle 1: residual 1.0e+00 -> 2.1e-04 (much more progress per cycle)
  Cycle 2: residual 2.1e-04 -> 8.7e-09 (converged!)
```

### When does stalling happen?

GMRES(m) (restarted with dimension m) stalls when:

- The matrix has eigenvalues with **small imaginary parts** relative to their real parts -- the Krylov subspace needs many directions to capture oscillatory components
- The eigenvalues are **clustered in multiple groups** separated by gaps -- each cluster requires its own Krylov directions
- The system is **highly non-normal** (eigenvectors far from orthogonal) -- the effective condition number is much worse than the eigenvalue condition number

**Rule of thumb**: if GMRES(m) converges in k < m iterations, restart doesn't matter. If it needs k >> m iterations, the stalling penalty can be severe.

## Using GMRES in MTL5

### Basic API

```cpp
#include <mtl/mtl.hpp>
using namespace mtl;

// Solve Ax = b with GMRES
itl::noisy_iteration<double> iter(b, 1000, 1e-10);  // max 1000 iters, tol 1e-10
itl::pc::identity<mat::compressed2D<double>> no_pc(A);  // no preconditioner
itl::gmres(A, x, b, no_pc, iter, 30);  // restart = 30
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | LinearOperator | The system matrix (dense or sparse) |
| `x` | Vector | Initial guess (modified in place to the solution) |
| `b` | Vector | Right-hand side |
| `M` | Preconditioner | Left preconditioner with `M.solve(y, r)` |
| `iter` | Iteration | Controls convergence check, max iterations, output |
| `restart` | int | Max Krylov dimension per cycle (default 30) |

### Choosing the restart parameter

| restart | Memory | Best for |
|---------|--------|----------|
| 10-20 | Low | Well-preconditioned systems, memory-constrained |
| 30-50 | Moderate | General purpose (default range) |
| 100-200 | High | Difficult non-symmetric systems, stalling issues |
| n | Full GMRES | Small systems, guaranteed convergence benchmark |

### Example: 2D Convection-Diffusion

This example assembles a non-symmetric convection-diffusion matrix and demonstrates the restart effect:

```cpp
#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
using namespace mtl;

int main() {
    // 1D convection-diffusion: -eps * u'' + u' = f
    // Central differences -> non-symmetric tridiagonal
    const std::size_t n = 200;
    const double eps = 0.01;  // small diffusion -> strongly non-symmetric
    const double h = 1.0 / (n + 1);

    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0 * eps / (h * h);          // diagonal
            if (i > 0)     ins[i][i-1] << -eps/(h*h) - 1.0/(2.0*h);  // lower
            if (i + 1 < n) ins[i][i+1] << -eps/(h*h) + 1.0/(2.0*h);  // upper
        }
    }

    // RHS: constant source
    vec::dense_vector<double> b(n, 1.0);

    // Compare restart = 10 vs 50 vs full
    for (int restart : {10, 30, 50, 200}) {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::identity<mat::compressed2D<double>> no_pc(A);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::gmres(A, x, b, no_pc, iter, restart);
        std::cout << "restart=" << std::setw(4) << restart
                  << "  iters=" << std::setw(4) << iter.iterations()
                  << "  residual=" << std::scientific << iter.resid()
                  << (iter.is_converged() ? "  CONVERGED" : "  NOT CONVERGED")
                  << "\n";
    }
    return 0;
}
```

Expected output (problem-dependent, but illustrative):

```text
restart=  10  iters= 500  residual=2.31e-03  NOT CONVERGED
restart=  30  iters= 210  residual=8.42e-11  CONVERGED
restart=  50  iters= 102  residual=5.67e-11  CONVERGED
restart= 200  iters=  67  residual=3.21e-11  CONVERGED
```

With restart=10, the solver stalls before reaching tolerance. Increasing restart allows more Krylov directions per cycle, enabling convergence. Full GMRES (restart=200 >= n) converges in the fewest total iterations.

### Example: Preconditioner reduces restart sensitivity

A good preconditioner clusters eigenvalues, reducing the effective Krylov dimension needed. This makes the restart choice much less critical:

```cpp
// With ILU(0) preconditioner:
itl::pc::ilu_0<double> ilu_pc(A);
itl::basic_iteration<double> iter(b, 500, 1e-10);
itl::gmres(A, x, b, ilu_pc, iter, 10);  // even restart=10 works now
```

Typical result:

```text
Without PC:  restart=10 -> NOT CONVERGED (500 iters)
With ILU(0): restart=10 -> CONVERGED in 12 iters
```

**The preconditioner matters more than the restart parameter.** A well-chosen preconditioner can reduce a 500-iteration stalling problem to 12 iterations, even with a small restart.

## Diagnosing GMRES Issues

### Symptom: residual stalls at a plateau

**Cause**: restart parameter too small for this problem.

**Fix**: increase restart (try 2x, then 4x), or add/improve the preconditioner.

### Symptom: residual decreases slowly but steadily

**Cause**: the matrix is well-suited for GMRES but the convergence rate is limited by eigenvalue spread.

**Fix**: better preconditioner to cluster eigenvalues. ILU(0) is a good first choice; ILDL^T for symmetric systems.

### Symptom: residual oscillates or increases

**Cause**: the preconditioner may be unstable, or the matrix is severely non-normal.

**Fix**: check the preconditioner. Try a less aggressive fill level. For severely non-normal systems, consider BiCGSTAB or TFQMR as alternatives.

### Symptom: fast convergence then sudden divergence

**Cause**: loss of orthogonality in the Arnoldi basis due to finite precision.

**Fix**: this is rare with Modified Gram-Schmidt (which MTL5 uses). If it occurs, it usually indicates the system is nearly singular.

## GMRES vs Other Krylov Solvers

| Solver | Symmetry | Memory | Convergence | When to use |
|--------|----------|--------|-------------|-------------|
| **CG** | SPD only | O(n) | Optimal | SPD systems (Laplacian, elasticity) |
| **GMRES(m)** | Any | O(mn) | Monotone decrease | General non-symmetric, smooth convergence needed |
| **BiCGSTAB** | Any | O(n) | Non-monotone | Memory-limited, tolerates irregular convergence |
| **TFQMR** | Any | O(n) | Quasi-monotone | Like BiCGSTAB but smoother convergence |
| **IDR(s)** | Any | O(sn) | Tunable | Between BiCGSTAB (s=1) and GMRES (s large) |

**GMRES is the safest default** for non-symmetric systems: its residual decreases monotonically (within each cycle), it never breaks down on nonsingular systems, and the restart parameter gives explicit memory control. The tradeoff is higher memory usage compared to short-recurrence methods like BiCGSTAB.

## Further Reading

- Saad & Schultz, "GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems", SIAM J. Sci. Stat. Comput., 7(3), 1986
- Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003 (Chapters 6-7)
- Trefethen & Bau, *Numerical Linear Algebra*, SIAM, 1997 (Lecture 35)
- Barrett et al., *Templates for the Solution of Linear Systems*, SIAM, 1994 (free online)
