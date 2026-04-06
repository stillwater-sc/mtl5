# CI improvements

## Current State

- 94 unit tests, all with tiny matrices (3x3 to 10x10)
- CI runs on 8 platforms in Release mode, completes in ~2 minutes
- No integration/regression tests (placeholder directory only)
- No large-scale solves — the biggest system tested is 10x10 tridiagonal
- Benchmark suite exists but is separate from testing (no pass/fail, just timing)

## Proposed Plan: Two-Tier CI

- Tier 1: Draft CI (current — fast, ~2 min)

  Keep exactly as-is. Runs on every push and draft PRs. Catches compilation errors and basic correctness across all 8 platforms.

- Tier 2: Regression CI (new — thorough, ~10-15 min)

  Runs only when a PR transitions to ready for review (not draft). Tests solver correctness at scale with diverse matrix patterns.

## What the regression tests would cover

|                 Category                  |              Sizes               |             What it validates              |
|-------------------------------------------|----------------------------------|--------------------------------------------|
| Dense LU/QR/Cholesky                      | 100, 500, 1000, 5000             | Backward stability: ‖Ax-b‖/‖b‖ < ε         |
| Dense eigenvalue/SVD                      | 100, 500, 1000                   | Orthogonality, residual checks             |
| Sparse Cholesky/LU/QR                     | 1K, 5K, 10K, 50K DOF             | Poisson/Laplacian grids, random SPD        |
| Iterative solvers (CG, GMRES, BiCGSTAB)   | 10K, 50K, 100K DOF               | Convergence to tolerance, iteration counts |
| Preconditioned solvers (ILU+GMRES, IC+CG) | 10K, 50K DOF                     | Preconditioner effectiveness               |
| Condition number sweep                    | 10 to 10^12                      | Solvers degrade gracefully                 |
| Generator matrices at scale               | Hilbert, Frank, Moler at 100-500 | Known-ill-conditioned matrices             |

## Implementation structure

```text
tests/
├── unit/           (existing — small, fast)
├── integration/    (existing placeholder)
└── regression/     (NEW)
  ├── CMakeLists.txt
  ├── dense/
  │   ├── test_lu_regression.cpp
  │   ├── test_qr_regression.cpp
  │   ├── test_cholesky_regression.cpp
  │   └── test_eigenvalue_regression.cpp
  ├── sparse/
  │   ├── test_sparse_cholesky_regression.cpp
  │   ├── test_sparse_lu_regression.cpp
  │   └── test_sparse_qr_regression.cpp
  └── itl/
      ├── test_cg_regression.cpp
      ├── test_gmres_regression.cpp
      └── test_bicgstab_regression.cpp
```

## CMake integration

```cmake
option(MTL5_BUILD_REGRESSION_TESTS "Build large-scale regression tests" OFF)
```

The ci preset stays fast. A new ci-regression preset enables regression tests.

## CI workflow changes

```yaml
# In ci.yml — add a regression job
regression:
if: >-
  github.event_name == 'pull_request' &&
  github.event.pull_request.draft == false
runs-on: ubuntu-latest   # just Linux, not full matrix
steps:
  - cmake --preset ci-regression
  - cmake --build build -j4
  - ctest --test-dir build -L regression
```

Draft PRs: only Tier 1 (fast, 8 platforms).
Ready PRs: Tier 1 + Tier 2 (regression on Linux).

## Key design decisions

1. CTest labels: regression tests labeled regression so ctest -L regression runs only them, ctest -LE regression excludes them
2. Parametric sizes: sizes controlled by #define or Catch2 GENERATE() so you can tune without recompiling
3. Residual-based validation: not exact answer checks — backward error Ax-b/(A·x) with tolerance scaling by condition number
4. Matrix generators: use existing mtl::generators (Hilbert, Moler, Frank, Lehmer) at large N, plus Poisson/Laplacian stencils for sparse
5. Timeout protection: CTest timeout per test (60s for dense, 120s for sparse) to catch performance regressions

