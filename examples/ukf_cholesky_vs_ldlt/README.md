# UKF Numerical Stability: Cholesky vs LDL^T

This example demonstrates the numerical stability advantage of the LDL^T
(square-root-free Cholesky) decomposition over standard Cholesky when used
for sigma point generation in an Unscented Kalman Filter (UKF).

## The Problem

The UKF requires a "matrix square root" of the state covariance P at every
time step to generate sigma points. The standard Cholesky factorization
(P = LL^T) computes this via iterative square roots of diagonal entries. When
P is ill-conditioned -- which happens immediately after a very informative
observation drives eigenvalues of P toward machine epsilon -- the sqrt()
amplifies relative error and can cause the filter to diverge.

The LDL^T factorization (P = LDL^T) avoids all square roots. The diagonal D
captures the same information as L^2 in Cholesky, but without the
precision-destroying sqrt step. Sigma points are generated from L and sqrt(D),
where each sqrt(D_k) involves only a single scalar -- not an accumulated error.

## System Model

- **State**: [px, py, vx, vy] -- 2D position and velocity (4-state)
- **Process**: constant velocity with quadratic drag
- **Measurement**: range and bearing from a beacon at (10, 0)
- **Key knob**: measurement noise covariance R

## Scenarios

| Scenario | R | Expected behavior |
|----------|---|-------------------|
| Benign | diag(1, 1) | Both methods perform identically |
| Moderate stress | diag(0.01, 0.1) | Mild ill-conditioning; Cholesky may show slight degradation |
| Severe stress | diag(1e-14, 1e-2) | Cholesky expected to degrade or diverge |
| Intermittent | alternating | Tests recovery after ill-conditioning episodes |

## Building and Running

```bash
cmake --preset dev
cmake --build build --target example_ukf_comparison
./build/examples/example_ukf_comparison
```

## Interpreting the Output

The program prints a table for each scenario with columns:

- **Step**: time step number
- **cond(P)**: condition number of the state covariance (higher = harder)
- **Chol-err / LDLT-err**: estimation error ||x_true - x_est||
- **Chol-resid / LDLT-resid**: factorization residual ||P - reconstruct|| / ||P||

When the Cholesky variant diverges, its columns show "diverged" or "NaN".
The LDL^T variant should remain stable across all scenarios.

## References

- Golub & Van Loan, *Matrix Computations*, Section 4.1.2
- Julier & Uhlmann, *Unscented Filtering and Nonlinear Estimation*, Proc. IEEE 2004
