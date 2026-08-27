#pragma once
// MTL5 -- Vector concepts (replaces MTL4 concept/vector.hpp)
#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/scalar.hpp>
#include <cstddef>

namespace mtl {

/// A 1D collection with indexed access
template <typename T>
concept Vector = Collection<T> && requires(const T& v, std::size_t i) {
    { v(i) } -> std::convertible_to<typename T::value_type>;
};

/// A vector whose elements form a FIELD -- the requirement of every algorithm
/// that divides by a vector element or by a scalar derived from one.
///
/// The Krylov solvers in `itl/` are that case (#503). Each of them computes a
/// step length as a ratio in the element type -- `cg`'s `beta = rho / rho_1` and
/// `alpha = rho / dot(p, q)`, and eight such ratios in `qmr` -- so on an integral
/// element type the division TRUNCATES and the iteration is nonsense from the
/// first step, converging or not and returning a confident wrong answer. The
/// integers are a ring, not a field: only 1 and -1 have inverses.
///
/// The vector counterpart of `FieldMatrix`, which #430 introduced for exactly
/// this failure in the decompositions -- `lu_factor(dense2D<int>)` used to
/// compile and return truncated nonsense. Same defect, one level up.
template <typename T>
concept FieldVector = Vector<T> && Field<typename T::value_type>;

/// A dense vector with contiguous storage
template <typename T>
concept DenseVector = Vector<T>;

/// A column vector
template <typename T>
concept ColumnVector = Vector<T>;

/// A row vector
template <typename T>
concept RowVector = Vector<T>;

} // namespace mtl
