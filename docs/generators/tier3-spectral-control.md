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
auto randorth(std::size_t n, std::mt19937_64& gen) -> mat::dense2D<T> {
    auto A = random_matrix<T>(n, n, T(-1), T(1), gen);   // also: a seed, or none
    vec::dense_vector<T> tau;
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);
    qr_normalize_signs(Q, R);   // R's diagonal non-negative, so Q is unique
    return Q;
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

A = Q * Λ * Q^T, symmetric by construction.
- **The eigenvalue form** takes any real spectrum, including negative and zero eigenvalues.
- **The mode form** uses the same mode system as `randsvd`, with λ ∈ [1/κ, 1]. Its result is therefore **positive definite**, the same kind of matrix as `randspd(n, kappa, mode)`. For an indefinite matrix with this conditioning, use `randsym_indefinite`.

### `randsym_indefinite<T>(...)` — `include/mtl/generators/randsym.hpp` (#961)

```cpp
template <typename T = double>
auto randsym_indefinite(std::size_t n, T kappa, int mode, std::size_t num_negative) -> mat::dense2D<T>;
// plus the std::uint64_t seed and std::mt19937_64& forms, as for every random generator
```

It builds the mode form's spectrum and negates `num_negative` of its eigenvalues. This is the usual specification for testing indefinite solvers and inertia code (LDLᵀ, MINRES, `inertia` / `is_indefinite`).
- **cond₂(A) = κ:** the signs do not change max|λ| / min|λ|.
- **The inertia** is (n − `num_negative`, `num_negative`, 0).
- **Which eigenvalues are negated:** a random subset of the positions, drawn from the engine after the spectrum, as LAPACK's `DLATMS` does with `RSIGN = 'T'`. The draw uses the raw engine stream, so a seed negates the same positions with every compiler and standard library.
- **`num_negative = 0`** draws nothing for the subset, and gives `randsym(n, kappa, mode)`'s matrix for the same seed.
- **`num_negative > n`** throws `std::invalid_argument`, as do a κ < 1 and a mode outside 1–5.
- **Why a separate name:** an overload `randsym(n, kappa, mode, num_negative)` would have the same parameter types as the seeded `randsym(n, kappa, mode, std::uint64_t seed)`, since `std::size_t` and `std::uint64_t` are the same type on most platforms.

### `randspd<T>(...)` — `include/mtl/generators/randspd.hpp`

Same as `randsym`, but it throws `std::invalid_argument` unless all eigenvalues are > 0. The mode-based overload guarantees positivity, since λ_i ∈ [1/κ, 1].

`randsym` and `randspd` average A(i,j) and A(j,i) after forming Q·Λ·Qᵀ. The product alone is not bitwise symmetric, and the average makes the result exactly symmetric.

### Reproducible draws (#844)

Every random generator has three forms: `randorth`, `randsvd`, `randsym`, `randsym_indefinite` and `randspd`, plus `fill_random`, `random_vector` and `random_matrix` in `operation/random.hpp`.

| Form | Example | Draws |
|---|---|---|
| no seed | `randspd<double>(n, kappa, mode)` | A `std::mt19937_64` seeded from `std::random_device`. Different on every call. |
| seed | `randspd<double>(n, kappa, mode, std::uint64_t(42))` | `std::mt19937_64(42)`. The same values on every call. |
| engine | `randspd<double>(n, kappa, mode, gen)` | The caller's `std::mt19937_64&`, so several calls share one stream |

The draws come from the raw `std::mt19937_64` stream, not from `std::uniform_real_distribution`. The standard specifies the engine bit for bit, but each standard library chooses its own distribution algorithm. So a seed gives the same random matrix with libstdc++, libc++ and MSVC, as in [`gen_dot` and `gen_sum`](ill-conditioned.md).

One uniform value on [lo, hi) is computed as follows. Draw u = (gen() >> 11) · 2⁻⁵³ on [0, 1), and take `fma(hi, u, lo·(1−u))` in the value type. A value that rounds up to hi is drawn again. The explicit `fma` keeps the rounding the same whether or not the compiler contracts floating-point expressions. The bounds must be finite. Matrices are filled row by row, whatever their orientation. The orthogonal factors are then the same up to the rounding of the QR factorization.

The square `randsvd` with a kappa takes its seed through the rectangular form, `randsvd<double>(n, n, kappa, mode, seed)`. A `randsvd(n, kappa, mode, seed)` overload would be ambiguous with `randsvd(m, n, kappa, mode)`.

The argument checks throw `std::invalid_argument` in every build. They cover kappa < 1, a mode outside 1–5, a σ or eigenvalue list of the wrong length, a non-positive `randspd` eigenvalue, and hi < lo. Before #844 they were `assert`s.

### `rosser<T>()` — `include/mtl/generators/rosser.hpp`

Hardcoded 8×8 matrix (Rosser, 1951). Its eigenvalues are -10√10405, 0, 510-100√26 (≈ 0.098), 1000, 1000, 510+100√26 (≈ 1019.90), 1020 and 10√10405 (≈ 1020.05). They sum to the trace, 4040. The matrix tests a double eigenvalue, three nearly equal ones, a zero and a tiny one. (The list before #844 had 1020 twice and lacked 510-100√26, so it summed to 5059.9.)

### `wilkinson<T>(n)` — `include/mtl/generators/wilkinson.hpp`

As MATLAB's `wilkinson(n)`, for any n ≥ 1. Tridiagonal:
- Diagonal: |i − (n−1)/2|, i = 0…n−1. For odd n = 2m+1 that is [m, m-1, ..., 1, 0, 1, ..., m-1, m]. For even n it is half-integers, e.g. [1.5, 0.5, 0.5, 1.5] for n = 4.
- Sub/super-diagonal: all 1s
- n = 0 throws `std::invalid_argument`. So does an even n with an integer value type, which cannot hold the half-integers.

Tests eigenvalue sensitivity — has nearly-equal eigenvalue pairs.

## Test Strategy

| Test | Key Verifications |
|------|-------------------|
| `test_randorth` | Q^T·Q ≈ I (Frobenius norm of error < tol) |
| `test_randsvd` | SVD of result recovers prescribed σ; test all 5 modes; cond = κ exactly |
| `test_randsym` | A = A^T; eigenvalues match prescribed λ (sorted). `randsym_indefinite`: the inertia, \|λ\| equal to `make_sigma`'s in every mode (so cond = κ), bitwise symmetry, seed reproducibility, `num_negative = 0` equals `randsym`, the negated positions pinned for one seed, argument checks |
| `test_randspd` | A = A^T; eigenvalues positive and match; Cholesky succeeds |
| `test_rosser` | Symmetry; dimension = 8; trace. The 8 eigenvalues are checked in `test_generator_properties` |
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
