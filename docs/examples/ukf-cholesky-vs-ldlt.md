# UKF Sigma Points: Cholesky vs LDL^T

The Unscented Kalman Filter (UKF) takes its sigma points from a square root of the state covariance $P$, at every step. Two factorizations give one:

- Cholesky, $P = LL^T$;
- $LDL^T$, $P = LDL^T$.

A common claim is that $LDL^T$ is the more robust choice, because it "avoids the square root" that loses precision when $P$ is ill-conditioned. MTL5's two UKF examples test that claim, and their output doesn't support it (issue 953):

- On an SPD covariance the two factorizations give the same filter, at any condition number.
- What breaks a float32 UKF is its weights, not its factorization.
- The factorizations do differ in kind on an indefinite matrix. There, neither gives sigma points.

This page walks through the reasoning and the programs' output.

## The square root in the UKF

For a state of dimension $n$ with mean $\hat{x}$ and covariance $P$, the sigma points are

$$\sigma_0 = \hat{x}, \qquad \sigma_{\pm k} = \hat{x} \pm \gamma\, S_{:,k}, \qquad \gamma = \sqrt{n + \lambda},$$

for any $S$ with $SS^T = P$, and $\lambda = \alpha^2 (n + \kappa) - n$. The two factorizations give two such $S$:

| | Cholesky | $LDL^T$ |
|---|---|---|
| Factorization | $P = LL^T$ | $P = LDL^T$, $L$ unit lower triangular |
| $S$ | $L$ | $L\,\mathrm{diag}(\sqrt{D_k})$ |
| Square roots | $n$, one per pivot | $n$, one per $D_k$, to form the sigma points |
| Cost | $n^3/3$ | $n^3/3$ |
| Indefinite $P$ | stops at the first non-positive pivot | returns, and $D$'s signs give the inertia |

So $LDL^T$ doesn't avoid the square root here. Its sigma points take $\sqrt{D_k}$, the same $n$ scalar square roots that Cholesky takes of its pivots.

## Why the condition number is not the issue

- **Cholesky is backward stable for every SPD matrix, whatever its condition number** (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Thm 10.3). The computed $\hat{L}$ is the exact factor of $P + \Delta P$, with $\|\Delta P\| \le c\, n\, \varepsilon\, \|P\|$. There is no condition number above which it "degrades".
- **A scalar square root is perfectly conditioned.** A relative error $\varepsilon$ in its argument becomes $\varepsilon / 2$ in its result. It halves the relative error rather than growing it. A small pivot is not a less accurate pivot.
- **Unpivoted $LDL^T$ has the same kind of backward-error bound on an SPD matrix.**

So any square root of $P$ gives sigma points symmetric about $\hat{x}$ to rounding, and both factorizations give one.

## What the examples show

### Example 1: range and bearing (`examples/ukf_cholesky_vs_ldlt/`)

A 4-state tracker, [px, py, vx, vy], with constant velocity and quadratic drag. It is observed through range and bearing from a beacon at (10, 0), in four scenarios whose measurement noise R goes down to diag(1e-14, 1e-2). One filter uses Cholesky and one uses $LDL^T$.

```text
=== Scenario: Severe stress (R = diag(1.0e-14, 1.0e-02)) ===
  Step       cond(P)      Chol-err      LDLT-err    Chol-resid    LDLT-resid
     0      1.77e+02      6.12e-01      6.12e-01      8.32e-17      0.00e+00
   100      4.85e+03      2.34e-01      2.34e-01      0.00e+00      4.88e-17
   199      6.15e+03      1.25e+00      1.25e+00      4.20e-17      5.24e-18

  Cholesky: completed 200 steps, final error = 1.246e+00
  LDL^T:    completed 200 steps, final error = 1.246e+00
  largest cond(P): 6.80e+03
```

- **Both filters complete in every scenario**, and their final errors agree to 2e-7. That gap is rounding, amplified by the weights (below).
- **A tiny R doesn't make $P$ ill-conditioned.** cond($P$) never exceeds 7e3: the process noise $Q$, added at every predict step, keeps $P$'s eigenvalues well above $\varepsilon$.
- **The direct test agrees.** On SPD matrices with eigenvalues spread up to cond 1e15, both residuals stay at rounding level:

```text
--- SPD matrices, n = 8, eigenvalues spread over cond(P) ---
       cond(P)   Chol residual   LDLT residual
       1.0e+02       1.137e-16       1.022e-16
       1.0e+08       5.464e-17       6.472e-17
       1.0e+15       3.182e-17       1.504e-16
```

### Example 2: bearing only (`examples/ukf_bearing_only/`)

One measurement, the bearing from a beacon. The bearing observes one direction of the position and leaves the range unobserved. The example runs two filters, one per factorization, on the same truth and measurements, in float64 and in float32, and at α = 1e-3, 1e-2, 1e-1 and 1.

```text
=== float32, alpha = 1e-3 ===
 step    cond(P)   bias Chol   bias LDLT    |xC - xL|   err Chol   err LDLT
    0    1.0e+00    2.19e-02    2.19e-02     0.00e+00      3.310      3.310
   12    1.4e+02    9.90e-02    9.89e-02     1.51e-02      3.155      3.152
   13    1.7e+02    9.34e-02    9.34e-02     6.27e-01      3.369      3.650
   14    1.0e+03    1.09e-01    5.10e-02     1.88e+00      5.670      4.032
```

The bias is $\|\sum_k W_k \sigma_k - \hat{x}\| / \|\hat{x}\|$. The weights sum to 1, so for a symmetric set of points the exact weighted mean is $\hat{x}$, and the bias measures only the rounding of the sum.

- **At step 0 the bias is already 2.2e-2,** where $P = 100\,I$ and cond($P$) = 1. Nothing about the factorization can cause that.
- **The two float32 filters drift up to 1.9 apart.** Their estimates were exact copies of each other until step 11, and cond($P$) peaks at about 1e3.
- **In float64 the same two filters agree to 8e-8.**

### The cause: the weights at α = 1e-3

With $n = 4$ and α = 1e-3, $\lambda = \alpha^2 n - n \approx -4$, so

$$W_0 = \frac{\lambda}{n + \lambda} \approx -10^6, \qquad W_i = \frac{1}{2(n + \lambda)} \approx 1.25 \times 10^5.$$

Every weighted mean and covariance therefore adds terms about $10^6$ times the result, which cancel. In float32 that cancels six of the seven digits. Raising α removes it, for both factorizations alike:

| α | $W_0$ (float32) | step-0 bias | gap between the two float32 filters |
|---|---|---|---|
| 1e-3 | −986894 | 2.2e-2 | 1.9 |
| 1e-2 | −9997.34 | 4.9e-4 | 1.4 |
| 1e-1 | −99.0001 | 2.9e-6 | 3.8e-3 |
| 1 | 0 | 0 | 6.6e-5 |

With α = 1 both float32 filters end at error 4.189, against 4.188 in float64. In low precision, keep $W_0$ small: α near 1, or a κ for which $n + \lambda$ is not tiny.

### Indefinite matrices

A covariance update $P = P^- - K S K^T$ can lose positive definiteness to rounding. On such a $P$ the two factorizations differ in kind. The table is from Example 1's direct test:

```text
--- Indefinite matrices ---
                    matrix lambda_min  neg Cholesky  LDL^T D signs  LDL^T berr  BK neg    BK berr
    Q diag(.., -1e-14) Q^T   -1.0e-14    1    stops      [+,+,+,-]     5.4e-17       1    7.7e-17
      H_6 - 2 lambda_min I   -1.1e-07    1    stops  [+,+,+,+,+,-]     5.7e-17       1    1.5e-16
      ones, diagonal 1e-12   -1.0e+00    3    stops      [+,-,-,-]     1.8e-04       3    1.6e-16
```

- **Cholesky** stops at the first non-positive pivot.
- **Unpivoted $LDL^T$** (`ldlt_factor`) returns, and fails only on an exact zero pivot. $D$'s signs report the inertia. But it is not backward stable on an indefinite matrix: a small pivot makes $L$ large. On the matrix of ones with 1e-12 on the diagonal, its solve's backward error is 1.8e-4.
- **Bunch–Kaufman** (`ldlt_bk_factor`) pivots with 1 × 1 and 2 × 2 blocks. It finds the inertia of every matrix, with a backward error of 1e-16.

Neither gives sigma points for an indefinite $P$, since it has no real square root. Both examples' $LDL^T$ filters refuse a $D_k \le 0$ for that reason: $L\,|D|\,L^T$ would be another covariance. What $LDL^T$ adds is the diagnosis, which directions went negative.

## When to use which

- **For an SPD matrix, at any condition number, either factorization.** Cholesky is the usual one, and LAPACK's `dpotrf` is the most optimized factorization in most BLAS libraries. Neither is the more accurate.
- **$LDL^T$ when no square root is available or wanted.** For example, solving with $P$ rather than forming sigma points, or a number type without a square root.
- **$LDL^T$'s $D$ when you need the inertia** of a matrix that should be SPD, to see which directions went negative.
- **Bunch–Kaufman (`ldlt_bk_factor`) for a matrix that is, or may be, indefinite:** saddle-point systems, modified Newton methods. Unpivoted $LDL^T$ is not stable there.
- **For a UKF in low precision, look at the weights first.**

## MTL5 API reference

### Dense Cholesky

```cpp
#include <mtl/operation/cholesky.hpp>

// Factor: A = L * L^T (in-place, lower triangle overwritten with L)
int info = mtl::cholesky_factor(A);  // returns 0 on success

// Solve: given precomputed L, solve A*x = b
mtl::cholesky_solve(L, x, b);
```

### Dense $LDL^T$

```cpp
#include <mtl/operation/ldlt.hpp>

// Factor: A = L * D * L^T (in-place, lower triangle gets L, diagonal gets D)
int info = mtl::ldlt_factor(A);  // returns 0 on success, k+1 on zero pivot

// Solve: given precomputed LDL^T factors, solve A*x = b
mtl::ldlt_solve(A, x, b);
```

### Bunch–Kaufman $LDL^T$

```cpp
#include <mtl/operation/ldlt_bk.hpp>

// Factor: P A P^T = L D L^T with 1x1 and 2x2 pivots (in-place)
mtl::bk_pivot_info piv;
int info = mtl::ldlt_bk_factor(A, piv);  // 0 on success, k+1 on a singular block
mtl::ldlt_bk_solve(A, piv, x, b);
```

### Sparse versions

Both factorizations have sparse counterparts in `mtl::sparse::factorization::`, with symbolic and numeric phases and fill-reducing orderings. See `<mtl/sparse/factorization/sparse_cholesky.hpp>` and `<mtl/sparse/factorization/sparse_ldlt.hpp>`.

## Further reading

- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Ch. 10 (Cholesky) and Ch. 11 (symmetric indefinite)
- Golub & Van Loan, *Matrix Computations*, Section 4.1.2 ($LDL^T$ factorization)
- Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation", Proc. IEEE, 92(3), 2004
- Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Ch. 11
- mtl5-python's port, the "UKF sigma points: Cholesky against LDLᵀ" notebook (stillwater-sc/mtl5-python#158)
