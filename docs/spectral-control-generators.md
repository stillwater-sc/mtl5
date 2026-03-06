# Tier 3: Spectral-Control Generators for `mtl::generators`

## Context

Tiers 1 & 2 (15 generators, committed) provide matrices with known structural properties. Tier 3 adds generators with **precise spectral control** — prescribed singular values, eigenvalues, and condition numbers. These are the most powerful tools for numerical testing because you verify results against known ground truth.

The centerpiece is `randsvd`: construct a matrix with *exact* prescribed singular values via U * Σ * V^T where U, V are random orthogonal matrices.

## New Generators (6 files)

| Generator | Returns | Description |
|-----------|---------|-------------|
| `randorth<T>(n)` | `dense2D<T>` | Random orthogonal matrix (building block) |
| `randsvd<T>(m,n,sigma)` | `dense2D<T>` | Matrix with prescribed singular values |
| `randsvd<T>(n,kappa,mode)` | `dense2D<T>` | Matrix with prescribed condition number |
| `randsym<T>(n,eigenvalues)` | `dense2D<T>` | Symmetric with prescribed eigenvalues |
| `randspd<T>(n,eigenvalues)` | `dense2D<T>` | SPD with prescribed positive eigenvalues |
| `rosser<T>()` | `dense2D<T>` | Classic 8x8 eigenvalue test matrix |
| `wilkinson<T>(n)` | `dense2D<T>` | W+ tridiagonal, tests eigenvalue sensitivity |

## New Files

```
include/mtl/generators/
    randorth.hpp        ← random orthogonal (QR of random matrix)
    randsvd.hpp         ← prescribed singular values / condition number
    randsym.hpp         ← prescribed eigenvalues (symmetric)
    randspd.hpp         ← prescribed positive eigenvalues (SPD)
    rosser.hpp          ← classic 8x8 test matrix
    wilkinson.hpp       ← Wilkinson W+ tridiagonal

tests/unit/generators/
    test_randorth.cpp
    test_randsvd.cpp
    test_randsym.cpp
    test_randspd.cpp
    test_rosser.cpp
    test_wilkinson.cpp
```

## Modified Files

- `include/mtl/generators/generators.hpp` — add 6 new `#include` lines
- No changes to `mtl_fwd.hpp` (Tier 3 are factory functions, no classes to forward-declare)
- No changes to `tests/unit/CMakeLists.txt` (glob already picks up `generators/*.cpp`)

## Implementation Details

### `randorth<T>(n)` — `include/mtl/generators/randorth.hpp`

Building block for all `rand*` generators. QR factorization of a random matrix:

```cpp
template <typename T = double>
auto randorth(std::size_t n) -> mat::dense2D<T> {
    auto A = random_matrix<T>(n, n);
    vec::dense_vector<T> tau;
    qr_factor(A, tau);
    return qr_extract_Q(A, tau);
}
```

Uses: `random_matrix()` from `operation/random.hpp`, `qr_factor()` + `qr_extract_Q()` from `operation/qr.hpp`.

### `randsvd<T>(...)` — `include/mtl/generators/randsvd.hpp`

**Overload 1: Explicit singular values**
```cpp
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, const std::vector<T>& sigma)
    -> mat::dense2D<T>;
```
Constructs A = U * Σ * V^T. U is m×m orthogonal, V is n×n orthogonal, Σ is m×n diagonal.

**Overload 2: Condition number + mode (square shorthand)**
```cpp
template <typename T = double>
auto randsvd(std::size_t n, T kappa, int mode = 3) -> mat::dense2D<T>;
```

**Overload 3: Condition number + mode (rectangular)**
```cpp
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, T kappa, int mode = 3)
    -> mat::dense2D<T>;
```

**Modes** (MATLAB `gallery('randsvd')` convention), p = min(m,n):
- **Mode 1:** One large: σ = [1, 1/κ, 1/κ, ..., 1/κ]
- **Mode 2:** One small: σ = [1, 1, ..., 1, 1/κ]
- **Mode 3:** Geometric: σ_i = κ^{-(i-1)/(p-1)} (default)
- **Mode 4:** Arithmetic: σ_i = 1 - (i-1)·(1-1/κ)/(p-1)
- **Mode 5:** Random log-uniform in [1/κ, 1]

All modes give σ₁ = 1, σ_p = 1/κ, so cond(A) = κ exactly.

Helper function `make_sigma(p, kappa, mode)` generates the σ vector from mode specification.

Uses: `randorth()`, `mat::dense2D<T>`, `operator*`, `trans()`.

### `randsym<T>(...)` — `include/mtl/generators/randsym.hpp`

```cpp
template <typename T = double>
auto randsym(std::size_t n, const std::vector<T>& eigenvalues) -> mat::dense2D<T>;
template <typename T = double>
auto randsym(std::size_t n, T kappa, int mode = 3) -> mat::dense2D<T>;
```

A = Q * Λ * Q^T. Same mode system for eigenvalue distribution. Result is symmetric by construction.

### `randspd<T>(...)` — `include/mtl/generators/randspd.hpp`

Same as `randsym` but asserts/enforces all eigenvalues > 0. Mode-based overload guarantees positivity since λ_i ∈ [1/κ, 1].

### `rosser<T>()` — `include/mtl/generators/rosser.hpp`

Hardcoded 8×8 matrix (Rosser, 1951). Known eigenvalues: {0, 1020, 1020, 1000, 1000, 10√10405, -10√10405, 510+100√26}. Tests repeated, near-equal, and zero eigenvalues.

### `wilkinson<T>(n)` — `include/mtl/generators/wilkinson.hpp`

n must be odd (2m+1). Tridiagonal:
- Diagonal: [m, m-1, ..., 1, 0, 1, ..., m-1, m]
- Sub/super-diagonal: all 1s

Tests eigenvalue sensitivity — has nearly-equal eigenvalue pairs.

## Test Strategy

| Test | Key Verifications |
|------|-------------------|
| `test_randorth` | Q^T·Q ≈ I (Frobenius norm of error < tol) |
| `test_randsvd` | SVD of result recovers prescribed σ; test all 5 modes; cond = κ exactly |
| `test_randsym` | A = A^T; eigenvalues match prescribed λ (sorted) |
| `test_randspd` | A = A^T; eigenvalues positive and match; Cholesky succeeds |
| `test_rosser` | Known 8 eigenvalues; symmetry; dimension = 8 |
| `test_wilkinson` | Tridiagonal structure; symmetry; known dimension; diagonal values |

## Implementation Order

1. `randorth.hpp` + test
2. `randsvd.hpp` + test (depends on randorth)
3. `randsym.hpp` + `randspd.hpp` + tests (depend on randorth)
4. `rosser.hpp` + `wilkinson.hpp` + tests (standalone)
5. Update `generators.hpp` umbrella
6. Full build + full test suite

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake --build build -j$(nproc)
ctest --test-dir build                # all 57+ tests pass
ctest --test-dir build -R generators  # generator tests specifically
```
