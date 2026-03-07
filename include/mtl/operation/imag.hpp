#pragma once
// MTL5 — Element-wise imaginary part extraction
#include <complex>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Element-wise imag for vectors → dense_vector<magnitude_t<T>>
template <Vector V>
auto imag(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    vec::dense_vector<mag_t> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        if constexpr (is_complex_v<typename V::value_type>) {
            result(i) = v(i).imag();
        } else {
            result(i) = mag_t{0};
        }
    }
    return result;
}

/// Element-wise imag for matrices → dense2D<magnitude_t<T>>
template <Matrix M>
auto imag(const M& m) {
    using mag_t = magnitude_t<typename M::value_type>;
    mat::dense2D<mag_t> result(m.num_rows(), m.num_cols());
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            if constexpr (is_complex_v<typename M::value_type>) {
                result(r, c) = m(r, c).imag();
            } else {
                result(r, c) = mag_t{0};
            }
        }
    }
    return result;
}

} // namespace mtl
