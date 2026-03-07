# Strengthen Linear Algebra Operator Tests with Generator Matrices

## Context

The current operator tests (LU, QR, SVD, Cholesky, eigenvalue) use only small (2×2, 3×3) hand-crafted matrices. This is adequate for basic correctness but misses critical numerical scenarios: ill-conditioning, larger sizes, known-answer verification, and structured matrices. We now have 21 generators (Tiers 1–3) purpose-built for exactly these scenarios. This plan adds generator-based tests to each operator test file.

## Approach

**Edit existing test files** — append new `TEST_CASE` blocks to the existing `test_*.cpp` files. Do NOT create new files. Each new test uses a specific generator chosen for a specific numerical reason.

## Files to Modify (7 files)

All in `tests/unit/operation/`:

### 1. `test_lu.cpp` — currently 3 tests

**Add 4 tests:**

| Test | Generator | Why |
|------|-----------|-----|
| LU solve on 8×8 Frank matrix | `frank<double>(8)` | Upper Hessenberg, known structure, larger size |
| LU solve on Moler matrix | `moler<double>(6)` | SPD with clustered eigenvalues, tests pivoting |
| LU on ill-conditioned Hilbert 6×6 | `hilbert<double>(6)` (materialized) | Classic ill-conditioned test — verify solve residual ‖Ax-b‖/‖b‖ stays reasonable |
| LU inverse of Pascal matrix | `pascal<double>(5)` | det=1, well-conditioned — verify A·A⁻¹ ≈ I |

### 2. `test_qr.cpp` — currently 4 tests

**Add 5 tests:**

| Test | Generator | Why |
|------|-----------|-----|
| QR on Kahan matrix | `kahan<double>(6)` | Upper triangular + ill-conditioned — classic QR stress test |
| QR orthogonality with randorth | `randorth<double>(8)` | QR of an already-orthogonal matrix — Q should be orthogonal, R ≈ ±I |
| QR on Frank (Hessenberg) | `frank<double>(6)` | Structured matrix — R should be upper triangular of Hessenberg |
| QR reconstruction on Vandermonde | `vandermonde<double>({1,2,3,4,5})` | Ill-conditioned — verify Q·R = A despite poor conditioning |
| QR on randsvd with known cond | `randsvd<double>(6, 100.0, 3)` | Known condition number — verify reconstruction accuracy |

### 3. `test_svd.cpp` — currently 4 tests

**Add 5 tests:**

| Test | Generator | Why |
|------|-----------|-----|
| SVD recovers prescribed singular values | `randsvd<double>(6, 6, {6,5,4,3,2,1})` | **Ground truth test** — prescribed σ must be recovered |
| SVD condition number from randsvd | `randsvd<double>(5, 50.0, 3)` | Verify σ₁/σₙ ≈ κ from geometric distribution |
| SVD of Hilbert matrix | `hilbert<double>(5)` (materialized) | Ill-conditioned — verify reconstruction ‖USVᵀ - A‖_F is small |
| SVD of rank-1 ones matrix | `ones<double>(4)` (materialized) | Rank-1 — only one nonzero singular value (= n) |
| SVD orthogonality on Moler matrix | `moler<double>(5)` | SPD with clustered eigenvalues near 0 — stress orthogonality |

### 4. `test_eigenvalue.cpp` — currently 6 tests

**Add 6 tests:**

| Test | Generator | Why |
|------|-----------|-----|
| Symmetric eigenvalues of Wilkinson W21 | `wilkinson<double>(7)` | Nearly-equal eigenvalue pairs — classic eigensolver stress test |
| Symmetric eigenvalues of randsym | `randsym<double>(5, {10,5,3,2,1})` | **Ground truth** — prescribed eigenvalues must be recovered |
| Symmetric eigenvalues of Clement | `clement<double>(6)` | Known eigenvalues: ±5, ±3, ±1 for n=6 |
| Symmetric eigenvalues of Lehmer | `lehmer<double>(5)` (materialized) | SPD — all eigenvalues positive, trace = 5 |
| General eigenvalues of Frank | `frank<double>(5)` | Known positive real eigenvalues |
| General eigenvalues of Forsythe | `forsythe<double>(5, 1e-10, 0.0)` | Near-defective — tests sensitivity to small perturbation |

### 5. `test_cholesky.cpp` — currently 3 tests

**Add 4 tests:**

| Test | Generator | Why |
|------|-----------|-----|
| Cholesky on randspd with known eigenvalues | `randspd<double>(5, {8,4,2,1,0.5})` | **Ground truth SPD** — guaranteed factorizable |
| Cholesky on Pascal matrix | `pascal<double>(6)` | SPD with det=1 — well-conditioned |
| Cholesky on Moler matrix | `moler<double>(6)` | SPD by construction (L·Lᵀ) — eigenvalues near 0 |
| Cholesky on Lehmer matrix | `lehmer<double>(6)` (materialized) | SPD — verify L·Lᵀ reconstruction accuracy |

### 6. `test_trisolve.cpp` — currently 4 tests (3 trisolve + 1 trace)

**Add 2 tests:**

| Test | Generator | Why |
|------|-----------|-----|
| Upper trisolve on Kahan matrix | `kahan<double>(6)` | Upper triangular + ill-conditioned — tests backward stability |
| Lower trisolve from Cholesky of Moler | `moler<double>(5)` | Extract L from Cholesky, solve Lx=b — natural pipeline test |

### 7. `test_norms.cpp` — currently 11 tests

**Add 3 tests:**

| Test | Generator | Why |
|------|-----------|-----|
| Frobenius norm of orthogonal matrix | `randorth<double>(6)` | ‖Q‖_F = √n for any n×n orthogonal matrix |
| One-norm of ones matrix | `ones<double>(5)` (materialized) | ‖J‖₁ = n (max col sum = n) |
| Infinity-norm of Wilkinson | `wilkinson<double>(7)` | Known structure → verify ‖W‖_∞ = m+2 where m=(n-1)/2 |

## Implementation Notes

### Materializing Implicit Generators

Tier 1 implicit generators (hilbert, lehmer, lotkin, ones, minij) are classes, not `dense2D`. Operations like `lu_factor` modify the matrix in place and require a `dense2D`. Materialize with a helper loop:

```cpp
auto H_gen = generators::hilbert<double>(n);
mat::dense2D<double> H(n, n);
for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
        H(i, j) = H_gen(i, j);
```

### Tolerance Strategy

- Well-conditioned tests (Pascal, Frank, randspd): tight tolerance (1e-10)
- Moderately conditioned (Moler, Lehmer, Vandermonde): moderate (1e-8)
- Ill-conditioned (Hilbert, Kahan, high-κ randsvd): loose (1e-4 to 1e-2) or use **relative residual** ‖Ax-b‖/(‖A‖·‖x‖)
- Eigenvalue tests: sort both computed and expected, compare pairwise

### Residual-Based Verification

For ill-conditioned systems, don't check `x ≈ x_true` (forward error); check **backward error**:
```cpp
auto residual = A * x - b;
double rel_residual = frobenius_norm(residual) / frobenius_norm(b);
REQUIRE(rel_residual < tolerance);
```

## Test Count

| File | Before | Added | After |
|------|--------|-------|-------|
| test_lu.cpp | 3 | 4 | 7 |
| test_qr.cpp | 4 | 5 | 9 |
| test_svd.cpp | 4 | 5 | 9 |
| test_eigenvalue.cpp | 6 | 6 | 12 |
| test_cholesky.cpp | 3 | 4 | 7 |
| test_trisolve.cpp | 4 | 2 | 6 |
| test_norms.cpp | 11 | 3 | 14 |
| **Total** | **35** | **29** | **64** |

Total test count: 57 existing + 0 new files = 57 executables, but each executable gains more `TEST_CASE` blocks.

## Implementation Order

1. `test_norms.cpp` — simplest, no solvers
2. `test_trisolve.cpp` — straightforward
3. `test_cholesky.cpp` — SPD generators
4. `test_lu.cpp` — general square
5. `test_qr.cpp` — square + conditioning
6. `test_svd.cpp` — ground truth + conditioning
7. `test_eigenvalue.cpp` — most complex verification logic
8. Full build + full test suite

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake --build build -j$(nproc)
ctest --test-dir build
ctest --test-dir build -R "op_test"  # operation tests specifically
```
