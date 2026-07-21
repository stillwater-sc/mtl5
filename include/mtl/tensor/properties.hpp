#pragma once
// MTL5 -- Rank-2 tensor property predicates (#244, batch 4):
// is_symmetric, is_antisymmetric.
//
// `tol` is an ABSOLUTE threshold on the relevant deviation; the default 0
// requires exact structure. The tests are written as !(dev <= tol) rather than
// dev > tol so a NaN component (dev unordered) fails the predicate instead of
// being silently accepted. Consistent with the matrix property predicates.
#include <cmath>
#include <cstddef>

#include <mtl/concepts/magnitude.hpp>
#include <mtl/tensor/tensor.hpp>

namespace mtl::tensor {

/// Symmetric rank-2 tensor: t(i,j) == t(j,i) (within tol) for all i < j.
template <typename V, std::size_t D>
bool is_symmetric(const tensor<V, 2, D>& t, magnitude_t<V> tol = 0) {
    using std::abs;
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = i + 1; j < D; ++j)
            if (!(abs(t(i, j) - t(j, i)) <= tol)) return false;
    return true;
}

/// Antisymmetric (skew) rank-2 tensor: t(i,j) == -t(j,i) (within tol). This
/// forces a zero diagonal (t(i,i) == -t(i,i)), which is checked explicitly.
template <typename V, std::size_t D>
bool is_antisymmetric(const tensor<V, 2, D>& t, magnitude_t<V> tol = 0) {
    using std::abs;
    for (std::size_t i = 0; i < D; ++i) {
        if (!(abs(t(i, i)) <= tol)) return false;               // diagonal must vanish
        for (std::size_t j = i + 1; j < D; ++j)
            if (!(abs(t(i, j) + t(j, i)) <= tol)) return false; // t(i,j) == -t(j,i)
    }
    return true;
}

} // namespace mtl::tensor
