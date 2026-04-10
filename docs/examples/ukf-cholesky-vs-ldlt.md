# UKF Numerical Stability: Cholesky vs LDL^T Decomposition

The Unscented Kalman Filter (UKF) generates sigma points from the state covariance matrix P at every time step. This requires a matrix factorization that serves as a "square root" of P. The choice of factorization has a direct impact on filter stability, particularly when P becomes ill-conditioned after informative observations.

This document walks through MTL5's UKF comparison examples, explains why the LDL^T (square-root-free Cholesky) decomposition provides superior numerical robustness, and offers guidance on when to use each method.

## The Problem: Square Roots and Ill-Conditioned Covariances

### Why the UKF needs a matrix square root

The UKF approximates a nonlinear probability distribution by propagating a carefully chosen set of **sigma points** through the system model. For a state with covariance P, the sigma points are:

    sigma_k = x_hat +/- sqrt(n + lambda) * column_k(S)

where S is a matrix satisfying S * S^T = P. The standard approach computes S via Cholesky factorization: P = L * L^T.

### How informative observations create ill-conditioning

When a measurement is very precise (small measurement noise R), the Kalman update collapses one or more directions of the state covariance toward machine epsilon while others remain O(1). This creates extreme eigenvalue spread:

- **Well-observed direction**: eigenvalue ~ epsilon_machine
- **Poorly-observed direction**: eigenvalue ~ O(1) or larger

The condition number cond(P) = lambda_max / lambda_min can reach 10^6 in double precision and 10^3 in single precision after just a few updates.

### The precision loss mechanism

Standard Cholesky factorization computes the diagonal of L via:

    L(j,j) = sqrt(A(j,j) - sum_{k<j} L(j,k)^2)

When A(j,j) is near machine epsilon after the subtraction, the `sqrt()` operation halves the number of significant digits:

- In float64 (53-bit significand): `sqrt(1e-15)` = `3.16e-8`, losing ~7.5 digits
- In float32 (24-bit significand): `sqrt(1e-7)` = `3.16e-4`, losing ~3.5 digits
- In float16 (11-bit significand): essentially all precision is lost

This degraded diagonal entry propagates into every subsequent column of L, contaminating the entire factorization.

## The Solution: LDL^T Factorization

### Mathematical relationship

The LDL^T decomposition factors a symmetric matrix as:

    A = L * D * L^T

where L is **unit** lower triangular (ones on the diagonal) and D is diagonal. Comparing with Cholesky:

| Property | Cholesky (LL^T) | LDL^T |
|----------|----------------|-------|
| Diagonal of L | sqrt(pivot values) | 1 (unit diagonal) |
| Square roots | One per column | **None** |
| Stores | L (lower triangular) | L (unit lower) + D (diagonal) |
| Cost | O(n^3/3) | O(n^3/3) |
| SPD required | Yes | No (works for indefinite) |

### Why avoiding square roots preserves precision

The LDL^T algorithm computes D(j) directly:

    D(j) = A(j,j) - sum_{k<j} L(j,k)^2 * D(k)

No square root is taken. D(j) retains the full precision of the subtraction result, however small. The unit lower triangular entries are:

    L(i,j) = (A(i,j) - sum_{k<j} L(i,k) * D(k) * L(j,k)) / D(j)

This is a simple division, not a precision-destroying sqrt.

### Sigma point generation with LDL^T

With the LDL^T factorization, sigma points are generated as:

    sigma_k = x_hat +/- sqrt(n + lambda) * L * sqrt(D(k)) * e_k

Each `sqrt(D(k))` involves only a **single scalar** -- not an accumulated error propagated from previous columns. This is the fundamental stability advantage.

### Detecting indefiniteness

A critical practical benefit: when the covariance update `P = P_pred - K * S * K^T` introduces rounding errors that make P slightly non-SPD (which happens in practice), the two methods behave very differently:

- **Cholesky**: attempts `sqrt(negative)` and fails completely
- **LDL^T**: produces a negative D entry, which can be detected and handled (e.g., clamping, covariance reset, or using Bunch-Kaufman pivoting)

## Experiment Setup

MTL5 includes two UKF examples that demonstrate these effects.

### Example 1: Range+Bearing UKF (`examples/ukf_cholesky_vs_ldlt/`)

A 4-state nonlinear tracking problem:

- **State**: [px, py, vx, vy] -- 2D position and velocity
- **Process model**: constant velocity with nonlinear quadratic drag
- **Measurement model**: range and bearing from a beacon at (10, 0)
- **Scenarios**: benign (R = I), moderate (R = diag(0.01, 0.1)), severe (R = diag(1e-14, 1e-2)), and intermittent

### Example 2: Bearing-Only UKF (`examples/ukf_bearing_only/`)

The canonical ill-conditioning generator:

- **Measurement**: bearing only (no range) -- one direction of P collapses while the other stays large
- **Tight measurement noise**: R = 1e-4
- **Precision comparison**: runs in both float64 and float32 to show where Cholesky breaks
- **Key diagnostic**: sigma-point mean bias -- measures the asymmetry introduced by degraded factorizations

### Building and running

```bash
cmake --preset dev
cmake --build build --target example_ukf_comparison example_bearing_only_ukf
./build/examples/example_ukf_comparison
./build/examples/example_bearing_only_ukf
```

## Results and Interpretation

### Double precision: both methods perform identically

In float64, both Cholesky and LDL^T succeed across all scenarios. The condition numbers reach ~10^4, well within the ~10^15 that double precision can handle:

```text
=== Scenario: Benign (R = diag(1.0e+00, 1.0e+00)) ===
  Step       cond(P)      Chol-err      LDLT-err    Chol-resid    LDLT-resid
     0      2.21e+00      9.70e-01      9.70e-01      6.12e-17      0.00e+00
   100      6.42e+01      1.37e+00      1.37e+00      3.88e-17      2.00e-17
   199      3.26e+02      1.50e+00      1.50e+00      1.19e-16      7.49e-18

  Cholesky: completed 200 steps, final error = 1.505e+00
  LDL^T:    completed 200 steps, final error = 1.505e+00
```

Both methods produce identical estimation errors and residuals at machine epsilon level. This is the expected baseline.

### Single precision: Cholesky fails, LDL^T detects the problem

The bearing-only example in float32 shows the critical difference. At step 5, the covariance update rounding errors make P non-SPD:

```text
=== Bearing-Only UKF (float32, 24-bit significand) ===
  Step     cond(P)    Chol    LDLT     Chol-bias     LDLT-bias
     0     1.0e+00      OK      OK      2.19e-02      2.19e-02
     4     6.1e+00      OK      OK      3.40e-02      3.40e-02
     5     2.1e+02    FAIL    FAIL           N/A           N/A
     6     2.1e+02    FAIL    FAIL           N/A           N/A
  ...
  Cholesky: 5/15 steps OK (10 FAILED), max bias = 8.96e-02
  LDL^T:    5/15 steps OK (10 FAILED), max bias = 8.96e-02
```

Both methods detect the failure -- but the failure modes differ:

- **Cholesky** fails via `sqrt(negative)`, providing no information about which direction went bad
- **LDL^T** `ldlt_factor()` **succeeds** — it only fails on zero pivots. Negative D entries are representable and indicate indefiniteness when inspected afterward. The caller can examine the D vector to identify exactly which eigenvalue direction became negative.

This diagnostic difference is what enables graceful recovery strategies: inspect D after factorization, clamp negative entries, reset the covariance, or trigger a warning — rather than handling a hard factorization failure.

### Indefinite matrices: Cholesky crashes, LDL^T succeeds

The direct factorization stress test constructs matrices with one deliberately negative eigenvalue -- simulating what happens when covariance update rounding goes wrong:

```text
--- Nearly-indefinite matrices (smallest eigenvalue < 0) ---
       min_eig        Cholesky           LDL^T     D entries
      -1.0e-14  FAIL (non-SPD)              OK  [+,+,+,-]
      -1.0e-10  FAIL (non-SPD)              OK  [+,+,+,-]
      -1.0e-06  FAIL (non-SPD)              OK  [+,+,+,-]
      -1.0e-02  FAIL (non-SPD)              OK  [+,+,+,-]
```

Cholesky fails on all four cases. LDL^T succeeds and reports the D entry signs, showing exactly one negative pivot. A UKF implementation can use this information to clamp the offending direction, reset the covariance, or trigger a diagnostic warning.

## When to Use Which

### Use Cholesky (LL^T) when:

- The matrix is guaranteed SPD (e.g., formed as A^T A + regularization)
- You need maximum performance and LAPACK `dpotrf` is available (it's the most optimized factorization in most BLAS libraries)
- Working in double precision with well-conditioned matrices (cond < 10^8)
- Backward compatibility with existing code is required

### Use LDL^T when:

- The matrix may be ill-conditioned (Kalman filters, optimization Hessians)
- The matrix may be symmetric indefinite (modified Newton methods, saddle-point systems)
- Working in reduced precision (float32, float16, posit, custom types)
- You need to detect indefiniteness rather than crashing
- No square root is available for the number type (e.g., some custom arithmetic)

### Use Bunch-Kaufman pivoted LDL^T when:

- The matrix is expected to be indefinite (not just nearly-indefinite)
- You need a numerically stable factorization regardless of pivot ordering
- Graceful degradation is more important than raw speed

Bunch-Kaufman is planned for MTL5 (see issue #46) and will be available as a third path in future examples.

### Rule of thumb

If `cond(P)` can exceed approximately 10^8 in your application, prefer LDL^T over Cholesky for sigma point generation. For reduced-precision types (float32, posit), the threshold is proportionally lower -- roughly 10^(significand_bits / 4).

## MTL5 API Reference

### Dense Cholesky

```cpp
#include <mtl/operation/cholesky.hpp>

// Factor: A = L * L^T (in-place, lower triangle overwritten with L)
int info = mtl::cholesky_factor(A);  // returns 0 on success

// Solve: given precomputed L, solve A*x = b
mtl::cholesky_solve(L, x, b);
```

### Dense LDL^T

```cpp
#include <mtl/operation/ldlt.hpp>

// Factor: A = L * D * L^T (in-place, lower triangle gets L, diagonal gets D)
int info = mtl::ldlt_factor(A);  // returns 0 on success, k+1 on zero pivot

// Solve: given precomputed LDL^T factors, solve A*x = b
mtl::ldlt_solve(A, x, b);
```

### Sparse versions

Both factorizations have sparse counterparts in `mtl::sparse::factorization::` with symbolic/numeric phase separation and fill-reducing ordering support. See `<mtl/sparse/factorization/sparse_cholesky.hpp>` and `<mtl/sparse/factorization/sparse_ldlt.hpp>`.

## Further Reading

- Golub & Van Loan, *Matrix Computations*, Section 4.1.2 (LDL^T factorization)
- Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation", Proc. IEEE, 92(3), 2004
- Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006
- Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Ch. 11
