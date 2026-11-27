# Bearing-only UKF: Cholesky against LDL^T, float64 against float32

A 2-D tracker whose one measurement is the bearing from a beacon. The bearing
pins down one direction of the position and leaves the range unobserved, which
gives P a wide eigenvalue spread.

The example runs **two filters on one scenario**, the same truth and the same
measurements:
- one takes its sigma points from Cholesky;
- the other from LDLᵀ.

It runs both in float64 and in float32, and repeats both at α = 1e-3, 1e-2, 1e-1
and 1. Its conclusion is computed from the run.

## What it shows

- **The factorization doesn't decide the outcome.**
  - In float64 the two filters' estimates agree to 8e-8 at α = 1e-3, and to
    3e-13 at α = 1.
  - cond(P) reaches 3e4.
- **In float32, the weights decide it.** At α = 1e-3 the unscented transform's
  weights are W₀ = λ/(n + λ) ≈ −10⁶ and Wᵢ ≈ 1.25·10⁵ (in float32, W₀ = −986894).
  So every weighted mean and covariance adds terms about 10⁶ times its result,
  and those terms cancel six of float32's seven digits.
  - The sigma-point mean is already off by 2.2e-2 at step 0, where P = 100·I and
    cond(P) = 1.
  - The two float32 filters drift up to 1.9 apart, while cond(P) peaks at about
    1e3.

  | α | W₀ (float32) | step-0 bias | gap between the filters |
  |---|---|---|---|
  | 1e-3 | −986894 | 2.2e-2 | 1.9 |
  | 1e-2 | −9997.34 | 4.9e-4 | 1.4 |
  | 1e-1 | −99.0001 | 2.9e-6 | 3.8e-3 |
  | 1 | 0 | 0 | 6.6e-5 |

  With α = 1, both float32 filters end at error 4.189, against 4.188 in
  float64.
- **In low precision, keep W₀ small:** α near 1, or a κ for which n + λ is not
  tiny. Changing the factorization doesn't help.
- **LDLᵀ doesn't avoid the square root.** Its sigma points are
  x̂ ± γ L √D_k e_k. Where the two methods differ is an indefinite P:
  - Cholesky stops;
  - LDLᵀ's D shows the inertia, but neither gives sigma points;
  - `ldlt_bk` (Bunch–Kaufman) factors a matrix that needs pivoting.

  See [`../ukf_cholesky_vs_ldlt`](../ukf_cholesky_vs_ldlt).

The sigma-point bias is ‖Σ W_k σ_k − x̂‖ / ‖x̂‖. For a symmetric set of points
the exact weighted mean is x̂, so the bias measures only the rounding of that
sum.

Each sigma point's bearing is wrapped relative to the central point's before the
sums. The target starts near the beacon's −x axis, where atan2 jumps from π to
−π.

## Building and running

```bash
cmake --preset dev
cmake --build build --target example_bearing_only_ukf
./build/examples/example_bearing_only_ukf
```

## References

- Julier & Uhlmann, *Unscented Filtering and Nonlinear Estimation*, Proc. IEEE 2004
- Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Ch. 11
- mtl5-python's port, the "UKF sigma points: Cholesky against LDLᵀ" notebook (stillwater-sc/mtl5-python#158)
