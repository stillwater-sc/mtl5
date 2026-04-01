# C++20 Concepts

MTL5 defines a hierarchy of C++20 concepts in `include/mtl/concepts/` that provide clear, compiler-checked constraints for generic algorithms.

## Concept Hierarchy

### Scalar Concepts

- **`Scalar<T>`** — arithmetic or custom number type with basic operations
- **`Field<T>`** — scalar with division (excludes integer types)
- **`OrderedField<T>`** — field with total ordering (for pivoting)

### Collection Concepts

- **`Collection<T>`** — any container with `size()` and element access
- **`Vector<T>`** — one-dimensional collection
- **`Matrix<T>`** — two-dimensional collection with `num_rows()` and `num_cols()`

### Operator Concepts

- **`LinearOperator<T>`** — supports matrix-vector multiplication
- **`Preconditioner<T>`** — supports `solve()` and `adjoint_solve()`

### Sparse Solver Concepts

- **`FillReducingOrdering<T>`** — produces a fill-reducing permutation
- **`SparseDirectSolver<T>`** — factorize and solve sparse systems

## Usage

Concepts appear as constraints on template parameters:

```cpp
#include <mtl/concepts/matrix.hpp>

template <mtl::Matrix M, mtl::Vector V>
auto solve(const M& A, const V& b) {
    // A is guaranteed to have num_rows(), num_cols(), etc.
    // V is guaranteed to have size(), operator[], etc.
}
```

## Concept Satisfaction

Any type that models the required interface satisfies the concept — no inheritance or registration required. This enables MTL5 to work seamlessly with custom number types from the Universal library.
