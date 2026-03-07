#pragma once
// MTL5 — Element-wise signum (sign function)
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

namespace detail {
    template <typename T>
    constexpr T signum_scalar(const T& x) {
        return (x > T{0}) ? T{1} : (x < T{0}) ? T{-1} : T{0};
    }
} // namespace detail

/// Element-wise signum for vectors: +1, 0, or -1
template <Vector V>
auto signum(const V& v) {
    using T = typename V::value_type;
    vec::dense_vector<T> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        result(i) = detail::signum_scalar(v(i));
    }
    return result;
}

/// Element-wise signum for matrices
template <Matrix M>
auto signum(const M& m) {
    using T = typename M::value_type;
    mat::dense2D<T> result(m.num_rows(), m.num_cols());
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            result(r, c) = detail::signum_scalar(m(r, c));
        }
    }
    return result;
}

} // namespace mtl
