# Tiers 1 & 2: Structural-Property Generators for `mtl::generators`

## Context

MTL5 needs a test matrix generation facility analogous to MATLAB's `gallery()` — matrices with known mathematical properties for verifying decompositions, solvers, and eigenvalue routines. Tier 1 provides implicit (storage-free) generators; Tier 2 provides factory functions returning concrete `dense2D<T>` or `compressed2D<T>`.

## Architecture: Two-Tier Design

**Tier 1 — Implicit generators** are lightweight classes with `operator()(r, c)` that compute entries on-the-fly. They satisfy the `Matrix` concept without allocating storage. Each requires trait registration (`category<>` → `tag::dense`, `ashape<>` → `mat<V>`).

**Tier 2 — Factory generators** are free functions returning `dense2D<T>` or `compressed2D<T>`. No trait registration needed since they return concrete types.

## New Generators (15 total across 14 files)

### Tier 1: Implicit Generators (5 files)

| Generator | Formula | Key Properties |
|-----------|---------|----------------|
| `hilbert<T>(n)` | H(i,j) = 1/(i+j+1) | SPD, notoriously ill-conditioned |
| `lehmer<T>(n)` | L(i,j) = (min(i,j)+1)/(max(i,j)+1) | SPD, well-conditioned, diag=1 |
| `lotkin<T>(n)` | Row 0 all 1s, rest = Hilbert | Asymmetric, ill-conditioned |
| `ones<T>(n)` / `ones<T>(m,n)` | O(i,j) = 1 | Rank-1, supports rectangular |
| `minij<T>(n)` | M(i,j) = min(i+1, j+1) | SPD, known inverse |

### Tier 2: Dense Factory Generators (8 files)

| Generator | Formula | Key Properties |
|-----------|---------|----------------|
| `kahan<T>(n, θ, ζ)` | Upper triangular, K(i,i)=s^i·ζ, K(i,j)=-c·s^i | Ill-conditioned, tests QR |
| `frank<T>(n)` | F(i,j) = n+1-max(i+1,j+1) for j≥i-1 | Upper Hessenberg, known eigenvalues |
| `moler<T>(n, α)` | M = L·L^T, L unit lower tri with off-diag α | SPD, eigenvalues cluster near 0 |
| `pascal<T>(n)` | P(i,j) = C(i+j, i), recurrence P(i,j)=P(i-1,j)+P(i,j-1) | SPD, det=1, binomial structure |
| `clement<T>(n)` | C(i,i±1) = √((i+1)(n-1-i)), diag=0 | Tridiagonal symmetric, known eigenvalues |
| `companion<T>(coeffs)` | Subdiag=1, last col=-coeffs | Eigenvalues = polynomial roots |
| `vandermonde<T>(nodes)` | V(i,j) = nodes[i]^j | Ill-conditioned for uniform nodes |
| `forsythe<T>(n, α, λ)` | Jordan block + corner perturbation F(n-1,0)=α | Tests near-defective matrices |

### Tier 2: Sparse Factory Generator (1 file, 2 functions)

| Generator | Formula | Key Properties |
|-----------|---------|----------------|
| `laplacian_1d<T>(n)` | Tridiagonal [-1, 2, -1] | SPD sparse, known eigenvalues λ_k = 2-2cos(kπ/(n+1)) |
| `laplacian_2d<T>(nx, ny)` | 5-point stencil, diag=4, neighbors=-1 | SPD sparse, (nx·ny)×(nx·ny), ≤5 nnz/row |

## New Files

```
include/mtl/generators/
    hilbert.hpp         ← implicit, SPD, ill-conditioned
    lehmer.hpp          ← implicit, SPD, well-conditioned
    lotkin.hpp          ← implicit, asymmetric, ill-conditioned
    ones.hpp            ← implicit, rank-1
    minij.hpp           ← implicit, SPD
    kahan.hpp           ← factory → dense2D, upper triangular
    frank.hpp           ← factory → dense2D, upper Hessenberg
    moler.hpp           ← factory → dense2D, SPD, L*L^T
    pascal.hpp          ← factory → dense2D, SPD, binomial
    clement.hpp         ← factory → dense2D, tridiagonal
    companion.hpp       ← factory → dense2D, polynomial roots
    vandermonde.hpp     ← factory → dense2D, ill-conditioned
    forsythe.hpp        ← factory → dense2D, near-defective
    laplacian.hpp       ← factory → compressed2D, 1D and 2D

    generators.hpp      ← umbrella header

tests/unit/generators/
    test_hilbert.cpp
    test_lehmer.cpp
    test_lotkin.cpp
    test_ones.cpp
    test_minij.cpp
    test_kahan.cpp
    test_frank.cpp
    test_moler.cpp
    test_pascal.cpp
    test_clement.cpp
    test_companion.cpp
    test_vandermonde.cpp
    test_forsythe.cpp
    test_laplacian.cpp
```

## Modified Files

- `include/mtl/mtl_fwd.hpp` — add forward declarations for the 5 implicit generator classes
- `tests/unit/CMakeLists.txt` — add glob for `generators/*.cpp` via `compile_all()`
- No changes needed for factory generators (they return concrete types, no forward declarations needed)

## Implementation Details

### Implicit Generator Pattern (Tier 1)

Each implicit generator follows this template:

```cpp
namespace mtl::generators {

template <typename Value = double>
class hilbert {
public:
    using value_type = Value;
    using size_type  = std::size_t;

    explicit hilbert(size_type n) : n_(n) {}

    value_type operator()(size_type r, size_type c) const {
        return Value(1) / Value(r + c + 1);
    }

    size_type num_rows() const { return n_; }
    size_type num_cols() const { return n_; }
    size_type size()     const { return n_ * n_; }

private:
    size_type n_;
};

} // namespace mtl::generators
```

**Required trait registrations** (at bottom of each file):

```cpp
namespace mtl::traits {
    template <typename V>
    struct category<generators::hilbert<V>> { using type = tag::dense; };
}

namespace mtl::ashape {
    template <typename V>
    struct ashape<::mtl::generators::hilbert<V>> { using type = mat<V>; };
}
```

**Forward declarations** in `mtl_fwd.hpp`:

```cpp
namespace generators {
    template <typename Value> class hilbert;
    template <typename Value> class lehmer;
    template <typename Value> class lotkin;
    template <typename Value> class ones;
    template <typename Value> class minij;
}
```

### Dense Factory Pattern (Tier 2)

```cpp
namespace mtl::generators {

template <typename T = double>
auto frank(std::size_t n) {
    mat::dense2D<T> F(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            // compute F(i, j)
        }
    return F;
}

} // namespace mtl::generators
```

### Sparse Factory Pattern (Tier 2)

```cpp
namespace mtl::generators {

template <typename T = double>
auto laplacian_1d(std::size_t n) {
    mat::compressed2D<T> L(n, n);
    {
        mat::inserter<mat::compressed2D<T>> ins(L);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << T(2);
            if (i > 0)     ins[i][i - 1] << T(-1);
            if (i + 1 < n) ins[i][i + 1] << T(-1);
        }
    }
    return L;
}

} // namespace mtl::generators
```

### Specific Generator Formulas

#### `hilbert<T>(n)` — H(i,j) = 1/(i+j+1)
SPD. Condition number grows super-exponentially with n.

#### `lehmer<T>(n)` — L(i,j) = (min(i,j)+1)/(max(i,j)+1)
SPD. All diagonal entries = 1. All entries in (0, 1].

#### `lotkin<T>(n)` — First row all 1s, rest = Hilbert
Asymmetric: L(0,1)=1 ≠ L(1,0)=0.5. Ill-conditioned like Hilbert.

#### `ones<T>(n)` / `ones<T>(m,n)` — All entries = 1
Two constructors: square and rectangular. Rank-1 matrix.

#### `minij<T>(n)` — M(i,j) = min(i+1, j+1)
SPD. For 3×3: [[1,1,1],[1,2,2],[1,2,3]].

#### `kahan<T>(n, θ=1.2, ζ=25)` — Upper triangular
- s = sin(θ), c = cos(θ)
- K(i,i) = s^i · ζ
- K(i,j) = -c · s^i for j > i
- K(i,j) = 0 for j < i

#### `frank<T>(n)` — Upper Hessenberg
- F(i,j) = n+1-max(i+1,j+1) for j ≥ i-1, else 0
- For n=4: [[4,3,2,1],[3,3,2,1],[0,2,2,1],[0,0,1,1]]
- Subdiagonal: F(i,i-1) = n-i

#### `moler<T>(n, α=-1)` — M = L·L^T
- L = unit lower triangular with L(i,j) = α for i > j
- M(i,i) = i·α² + 1
- M(i,j) = min(i,j)·α² + α for i ≠ j
- For α=-1, n=3: [[1,-1,-1],[-1,2,0],[-1,0,3]]

#### `pascal<T>(n)` — Binomial coefficients
- P(i,0) = P(0,j) = 1
- P(i,j) = P(i-1,j) + P(i,j-1)
- Entries are C(i+j, i)

#### `clement<T>(n)` — Tridiagonal, known eigenvalues
- Diagonal = 0
- C(i,i±1) = √((i+1)(n-1-i))
- For even n, eigenvalues: ±(n-1), ±(n-3), ..., ±1

#### `companion<T>(coeffs)` — Polynomial eigenvalue matrix
- Input: coefficients [c₀, c₁, ..., c_{n-1}] of p(x) = x^n + c_{n-1}x^{n-1} + ... + c₀
- Subdiagonal = 1, last column = -coeffs, rest = 0
- Eigenvalues = roots of p(x)

#### `vandermonde<T>(nodes)` — V(i,j) = nodes[i]^j
- First column all 1s
- Ill-conditioned for uniformly-spaced nodes

#### `forsythe<T>(n, α=1e-10, λ=0)` — Perturbed Jordan block
- Diagonal = λ, superdiagonal = 1, F(n-1,0) = α
- When α=0: pure Jordan block
- Tests eigensolvers near defective matrices

#### `laplacian_1d<T>(n)` — 1D discrete Laplacian
- Tridiagonal: [-1, 2, -1]
- SPD. Eigenvalues: 2 - 2cos(kπ/(n+1))

#### `laplacian_2d<T>(nx, ny)` — 2D discrete Laplacian
- 5-point stencil on nx×ny grid → (nx·ny)×(nx·ny) sparse matrix
- Diagonal = 4, off-diagonal neighbors = -1

## Test Strategy

All tests use Catch2 with `WithinAbs` floating-point matchers.

| Test | Key Verifications |
|------|-------------------|
| `test_hilbert` | Concept check, H(i,j)=1/(i+j+1), symmetry, condition growth |
| `test_lehmer` | Concept check, L(i,j) formula, symmetry, diagonal=1, positivity |
| `test_lotkin` | Concept check, first row=1, rows 1+ match Hilbert, asymmetry |
| `test_ones` | Concept check, square/rectangular, all entries=1, integer type |
| `test_minij` | Concept check, M(i,j)=min(i+1,j+1), symmetry, known 3×3 |
| `test_kahan` | Dimensions, upper triangular, diagonal=s^i·ζ, superdiag=-c·s^i |
| `test_frank` | Dimensions, Hessenberg, known 4×4, subdiag=n-i |
| `test_moler` | Dimensions, symmetry, known 3×3, L·L^T reconstruction, custom α |
| `test_pascal` | Dimensions, symmetry, known values, first row/col=1, recurrence |
| `test_clement` | Dimensions, tridiagonal, diag=0, symmetry, known 4×4 values |
| `test_companion` | Dimensions, subdiag=1, last col=-coeffs, 2×2 quadratic roots |
| `test_vandermonde` | Dimensions, V(i,j)=nodes[i]^j, first col=1, second col=nodes |
| `test_forsythe` | Dimensions, diag=λ, superdiag=1, corner=α, zero-α Jordan |
| `test_laplacian` | 1D: tridiag [-1,2,-1], row sums; 2D: dimensions, diag=4, symmetry |

## Implementation Order

1. Implicit generators (hilbert, lehmer, lotkin, ones, minij) + trait registrations + forward declarations + tests
2. Dense factory generators (kahan, frank, moler, pascal, clement, companion, vandermonde, forsythe) + tests
3. Sparse factory generator (laplacian) + test
4. Umbrella header `generators.hpp`
5. CMakeLists.txt test registration
6. Full build + full test suite

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake -B build
cmake --build build -j$(nproc)
ctest --test-dir build                # all tests pass
ctest --test-dir build -R generators  # generator tests specifically
```
