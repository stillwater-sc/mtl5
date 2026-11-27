# UKF sigma points: Cholesky against LDL^T

The Unscented Kalman Filter takes its sigma points from a square root of the
state covariance P at every step:

- **Cholesky**, P = L Lᵀ, uses the columns of L.
- **LDLᵀ**, P = L D Lᵀ, uses the columns L(:, k)·√D_k.

This example runs one filter with each, on the same scenarios, and factors test
matrices directly. Its conclusion is computed from the run.

## What it shows

- **On an SPD covariance the two are equally accurate, at any condition number.**
  - In all four scenarios both filters complete, and their final errors agree
    to 2e-7. The gap is rounding, amplified by the UKF weights at α = 1e-3:
    W₀ ≈ −10⁶, see [`../ukf_bearing_only`](../ukf_bearing_only).
  - On SPD matrices up to cond 1e15, both factorizations' residuals stay at
    1e-16.
  - Cholesky is backward stable for any SPD matrix (Higham, *Accuracy and
    Stability of Numerical Algorithms*, Thm 10.3). A scalar square root halves a
    relative error rather than growing it.
  - LDLᵀ doesn't avoid the square root: its sigma points take √D_k.
- **A tiny measurement noise doesn't make P ill-conditioned here.** cond(P)
  stays below 7e3 even with R = diag(1e-14, 1e-2). The process noise Q, added at
  every predict step, keeps P's eigenvalues up.
- **The two differ in kind on an indefinite matrix.**
  - Cholesky stops.
  - Unpivoted LDLᵀ returns, and D's signs report the inertia. It is not
    backward stable on an indefinite matrix, though: on ones with 1e-12 on the
    diagonal, its solve's backward error is 1.8e-4.
  - Bunch–Kaufman (`ldlt_bk`) pivots. It finds the inertia of every test matrix,
    with a backward error of 1e-16.
- **An indefinite covariance has no sigma points**, by either method. Each can
  only report it.

## System model

- **State**: [px, py, vx, vy], 2-D position and velocity
- **Process**: constant velocity with quadratic drag
- **Measurement**: range and bearing from a beacon at (10, 0). Each sigma point's
  bearing is wrapped relative to the central point's before the sums, so a set
  that straddles atan2's cut at ±π stays consistent.

| Scenario | R |
|----------|---|
| Benign | diag(1, 1) |
| Moderate | diag(0.01, 0.1) |
| Severe | diag(1e-14, 1e-2) |
| Intermittent | diag(1e-8, 1e-2) and diag(1, 1), alternating every 25 steps |

The direct tests are of two kinds:
- SPD matrices of order 8 with eigenvalues spread over cond(P) = 1e2 to 1e15.
- Indefinite matrices:
  - a random rotation of diag(1, 0.5, 0.1, λ);
  - the Hilbert matrix H₆ shifted below zero by its smallest eigenvalue;
  - ones with a tiny diagonal.

## Building and running

```bash
cmake --preset dev
cmake --build build --target example_ukf_comparison
./build/examples/example_ukf_comparison
```

## References

- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Ch. 10–11
- Golub & Van Loan, *Matrix Computations*, Section 4.1.2
- Julier & Uhlmann, *Unscented Filtering and Nonlinear Estimation*, Proc. IEEE 2004
