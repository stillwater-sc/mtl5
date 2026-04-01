# Architecture Overview

MTL5 is organized as a header-only library under `include/mtl/`. The design preserves MTL4's proven architecture while modernizing with C++20 features.

## Source Layout

| Directory | Purpose |
|---|---|
| `concepts/` | C++20 concepts: `Scalar`, `Field`, `Matrix`, `Vector`, `Collection` |
| `tag/` | Compile-time tags: orientation, sparsity, shape, traversal |
| `traits/` | Type traits: `category`, `ashape`, `transposed_orientation` |
| `math/` | Algebraic identities: `zero<T>()`, `one<T>()`, operation tags |
| `mat/` | Matrix types: `dense2D`, `compressed2D`, `coordinate2D`, `ell_matrix` |
| `vec/` | Vector types: `dense_vector`, `sparse_vector` |
| `operation/` | Free functions: decompositions, norms, solvers |
| `itl/` | Iterative solvers: CG, BiCGSTAB, GMRES, preconditioners |
| `sparse/` | Sparse direct solver infrastructure |
| `interface/` | Optional BLAS/LAPACK/UMFPACK bindings |
| `io/` | Matrix Market I/O |

## Key Patterns

### C++20 Concepts

MTL5 replaces Boost.MPL enable_if chains with C++20 concepts:

```cpp
// MTL4 style
template <typename T, typename Enable = boost::enable_if<is_matrix<T>>>
void solve(const T& A, ...);

// MTL5 style
template <Matrix M>
void solve(const M& A, ...);
```

### Expression Templates

Lazy evaluation via CRTP provides zero-overhead abstractions:

```cpp
auto expr = A * x + b;  // No computation yet
vec::dense_vector<double> result(expr);  // Evaluated here
```

### Static Polymorphism

Tag dispatch and `if constexpr` replace runtime polymorphism:

```cpp
template <Matrix M>
auto norm(const M& A) {
    if constexpr (traits::is_sparse_v<M>) {
        // Sparse path
    } else {
        // Dense path — may dispatch to BLAS
    }
}
```

## Namespaces

- `mtl::` — top-level
- `mtl::mat` — matrix types and operations
- `mtl::vec` — vector types and operations
- `mtl::math` — algebraic identities
- `mtl::itl` — iterative solvers
- `mtl::itl::pc` — preconditioners
- `mtl::sparse` — sparse direct solver infrastructure
- `mtl::interface` — external library bindings
