# C++20 Concepts

MTL5 defines a hierarchy of C++20 concepts in `include/mtl/concepts/` that provide clear, compiler-checked constraints for generic algorithms.

## Concept Hierarchy

### Scalar Concepts

- **`Scalar<T>`** — arithmetic or custom number type with basic operations, and a `std::numeric_limits` specialization on its magnitude type (`HasNumericLimits<T>`)
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

### Every element type has `std::numeric_limits`

`Scalar<T>` requires `std::numeric_limits<magnitude_t<T>>::is_specialized` (#838). The requirement is on the magnitude type because the standard does not specialize `numeric_limits` for `std::complex`: `std::complex<T>` qualifies when `T` does.

Algorithms that need a type's precision or range (the tolerances of the factorizations, the deflation floors of the eigensolvers, the scaling of `expm` and of the Sylvester solver) read them from `numeric_limits`. There is one source for these properties and no per-algorithm fallback. Before #838 a type without limits got a measured epsilon in some places, an exact comparison in others, and no scaling elsewhere.

A custom number type therefore specializes `std::numeric_limits`, as every Universal number system does. A type with the arithmetic and no limits is not a `Scalar`, and the compiler names `HasNumericLimits` as the unsatisfied constraint.
