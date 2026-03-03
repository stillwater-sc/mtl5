#pragma once
// MTL5 — Operator overloads for matrix types
#include <type_traits>
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::mat {

// ── Binary arithmetic ───────────────────────────────────────────────────

template <Matrix M1, Matrix M2>
auto operator+(const M1& a, const M2& b) {
    assert(a.num_rows() == b.num_rows() && a.num_cols() == b.num_cols());
    using result_t = std::common_type_t<typename M1::value_type, typename M2::value_type>;
    dense2D<result_t> r(a.num_rows(), a.num_cols());
    for (typename M1::size_type i = 0; i < a.num_rows(); ++i)
        for (typename M1::size_type j = 0; j < a.num_cols(); ++j)
            r(i, j) = a(i, j) + b(i, j);
    return r;
}

template <Matrix M1, Matrix M2>
auto operator-(const M1& a, const M2& b) {
    assert(a.num_rows() == b.num_rows() && a.num_cols() == b.num_cols());
    using result_t = std::common_type_t<typename M1::value_type, typename M2::value_type>;
    dense2D<result_t> r(a.num_rows(), a.num_cols());
    for (typename M1::size_type i = 0; i < a.num_rows(); ++i)
        for (typename M1::size_type j = 0; j < a.num_cols(); ++j)
            r(i, j) = a(i, j) - b(i, j);
    return r;
}

// ── Unary negation ──────────────────────────────────────────────────────

template <Matrix M>
auto operator-(const M& m) {
    using T = typename M::value_type;
    dense2D<T> r(m.num_rows(), m.num_cols());
    for (typename M::size_type i = 0; i < m.num_rows(); ++i)
        for (typename M::size_type j = 0; j < m.num_cols(); ++j)
            r(i, j) = -m(i, j);
    return r;
}

// ── Scalar-matrix multiply ──────────────────────────────────────────────
// Use std::is_arithmetic_v to avoid recursive concept evaluation with Scalar.

template <typename S, Matrix M>
    requires std::is_arithmetic_v<S>
auto operator*(const S& alpha, const M& m) {
    using result_t = std::common_type_t<S, typename M::value_type>;
    dense2D<result_t> r(m.num_rows(), m.num_cols());
    for (typename M::size_type i = 0; i < m.num_rows(); ++i)
        for (typename M::size_type j = 0; j < m.num_cols(); ++j)
            r(i, j) = alpha * m(i, j);
    return r;
}

template <Matrix M, typename S>
    requires std::is_arithmetic_v<S>
auto operator*(const M& m, const S& alpha) {
    return alpha * m;
}

template <Matrix M, typename S>
    requires std::is_arithmetic_v<S>
auto operator/(const M& m, const S& alpha) {
    using result_t = std::common_type_t<typename M::value_type, S>;
    dense2D<result_t> r(m.num_rows(), m.num_cols());
    for (typename M::size_type i = 0; i < m.num_rows(); ++i)
        for (typename M::size_type j = 0; j < m.num_cols(); ++j)
            r(i, j) = m(i, j) / alpha;
    return r;
}

// ── Matrix-vector multiply ──────────────────────────────────────────────

template <Matrix M, Vector V>
    requires (!Matrix<V>)
auto operator*(const M& A, const V& x) {
    assert(A.num_cols() == x.size());
    using result_t = std::common_type_t<typename M::value_type, typename V::value_type>;
    vec::dense_vector<result_t> y(A.num_rows());
    for (typename M::size_type r = 0; r < A.num_rows(); ++r) {
        auto acc = math::zero<result_t>();
        for (typename M::size_type c = 0; c < A.num_cols(); ++c) {
            acc += A(r, c) * x(c);
        }
        y(r) = acc;
    }
    return y;
}

// ── Matrix-matrix multiply ──────────────────────────────────────────────

template <Matrix M1, Matrix M2>
auto operator*(const M1& A, const M2& B) {
    assert(A.num_cols() == B.num_rows());
    using result_t = std::common_type_t<typename M1::value_type, typename M2::value_type>;
    dense2D<result_t> C(A.num_rows(), B.num_cols());
    for (typename M1::size_type r = 0; r < A.num_rows(); ++r) {
        for (typename M2::size_type c = 0; c < B.num_cols(); ++c) {
            auto acc = math::zero<result_t>();
            for (typename M1::size_type k = 0; k < A.num_cols(); ++k) {
                acc += A(r, k) * B(k, c);
            }
            C(r, c) = acc;
        }
    }
    return C;
}

} // namespace mtl::mat
