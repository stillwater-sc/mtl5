# Bearing-Only UKF: Cholesky vs LDL^T Stress Test

This example demonstrates the canonical ill-conditioning scenario for UKF
sigma point generation: bearing-only tracking, where a tight angular measurement
collapses one direction of the state covariance while leaving the range
direction essentially unobserved.

## Why Bearing-Only?

Range+bearing measurements constrain all state dimensions roughly equally.
Bearing-only measurements create extreme eigenvalue spread in P:

- **Small eigenvalue**: along the bearing normal (well-observed)
- **Large eigenvalue**: along the line of sight (unobserved)

After a few updates with R = 1e-4 (tight bearing), cond(P) can reach 10^6
in double, 10^3 in float -- exactly where Cholesky starts failing silently.

## The Silent Killer: Sigma-Point Mean Bias

This example measures not just whether the factorization completes, but
whether the resulting sigma points are symmetric around the mean. A degraded
factorization produces asymmetric sigma points, biasing the predicted mean.

The bias is small per-step but compounds: 10-20 UKF iterations in float32
with ill-conditioned updates can produce mean drifts of 0.5-2 sigma.

## Precision Levels

| Type | Significand bits | Expected behavior |
|------|-----------------|-------------------|
| float64 (double) | 53 | Both methods succeed; baseline |
| float32 (float) | 24 | Cholesky may degrade; LDL^T extends usable range |

For posit and custom float types, see the mixed-precision experiment in
the mtl5-python repository (stillwater-sc/mtl5-python#18).

## Building and Running

```bash
cmake --preset dev
cmake --build build --target example_bearing_only_ukf
./build/examples/example_bearing_only_ukf
```

## Output Columns

- **Step**: UKF iteration (each applies one bearing measurement)
- **cond(P)**: condition number of the state covariance
- **Chol/LDLT**: did the factorization succeed?
- **bias**: sigma-point mean asymmetry (higher = more biased prediction)
- **resid**: factorization residual ||P - reconstruct|| / ||P||

## References

- Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Ch. 11
- Julier & Uhlmann, *Unscented Filtering and Nonlinear Estimation*, Proc. IEEE 2004
