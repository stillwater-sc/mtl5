#pragma once
// MTL5 -- Vector/matrix norms: one_norm, two_norm, infinity_norm, frobenius_norm
#include <algorithm>
#include <cmath>
#include <cassert>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

// -- Vector norms --------------------------------------------------------

/// one_norm(v) = sum(|v[i]|)
template <Vector V>
auto one_norm(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    auto acc = math::zero<mag_t>();
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::abs;
        acc += abs(v(i));
    }
    return acc;
}

/// two_norm(v) = sqrt(sum(|v[i]|^2))
template <Vector V>
auto two_norm(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    auto acc = math::zero<mag_t>();
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::abs;
        auto a = abs(v(i));
        acc += a * a;
    }
    using std::sqrt;
    return sqrt(acc);
}

/// infinity_norm(v) = max(|v[i]|)
template <Vector V>
auto infinity_norm(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    auto result = math::zero<mag_t>();
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::abs;
        auto a = abs(v(i));
        if (a > result) result = a;
    }
    return result;
}

// -- Matrix norms --------------------------------------------------------

/// frobenius_norm(m) = sqrt(sum(|m[i,j]|^2))
template <Matrix M>
auto frobenius_norm(const M& m) {
    using mag_t = magnitude_t<typename M::value_type>;
    auto acc = math::zero<mag_t>();
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::abs;
            auto a = abs(m(r, c));
            acc += a * a;
        }
    }
    using std::sqrt;
    return sqrt(acc);
}

/// one_norm(m) = max column sum of |m[i,j]|
template <Matrix M>
auto one_norm(const M& m) {
    using mag_t = magnitude_t<typename M::value_type>;
    auto result = math::zero<mag_t>();
    for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
        auto col_sum = math::zero<mag_t>();
        for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
            using std::abs;
            col_sum += abs(m(r, c));
        }
        if (col_sum > result) result = col_sum;
    }
    return result;
}

/// infinity_norm(m) = max row sum of |m[i,j]|
template <Matrix M>
auto infinity_norm(const M& m) {
    using mag_t = magnitude_t<typename M::value_type>;
    auto result = math::zero<mag_t>();
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        auto row_sum = math::zero<mag_t>();
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::abs;
            row_sum += abs(m(r, c));
        }
        if (row_sum > result) result = row_sum;
    }
    return result;
}

} // namespace mtl
