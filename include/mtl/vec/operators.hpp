#pragma once
// MTL5 — Operator overloads for vector types
#include <type_traits>
#include <cassert>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::vec {

// ── Binary arithmetic ───────────────────────────────────────────────────

template <Vector V1, Vector V2>
auto operator+(const V1& a, const V2& b) {
    assert(a.size() == b.size());
    using result_t = std::common_type_t<typename V1::value_type, typename V2::value_type>;
    dense_vector<result_t> r(a.size());
    for (typename V1::size_type i = 0; i < a.size(); ++i)
        r(i) = a(i) + b(i);
    return r;
}

template <Vector V1, Vector V2>
auto operator-(const V1& a, const V2& b) {
    assert(a.size() == b.size());
    using result_t = std::common_type_t<typename V1::value_type, typename V2::value_type>;
    dense_vector<result_t> r(a.size());
    for (typename V1::size_type i = 0; i < a.size(); ++i)
        r(i) = a(i) - b(i);
    return r;
}

// ── Unary negation ──────────────────────────────────────────────────────

template <Vector V>
auto operator-(const V& v) {
    using T = typename V::value_type;
    dense_vector<T> r(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i)
        r(i) = -v(i);
    return r;
}

// ── Scalar-vector multiply ──────────────────────────────────────────────
// Use std::is_arithmetic_v to avoid recursive concept evaluation with Scalar.

template <typename S, Vector V>
    requires std::is_arithmetic_v<S>
auto operator*(const S& alpha, const V& v) {
    using result_t = std::common_type_t<S, typename V::value_type>;
    dense_vector<result_t> r(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i)
        r(i) = alpha * v(i);
    return r;
}

template <Vector V, typename S>
    requires std::is_arithmetic_v<S>
auto operator*(const V& v, const S& alpha) {
    return alpha * v;
}

template <Vector V, typename S>
    requires std::is_arithmetic_v<S>
auto operator/(const V& v, const S& alpha) {
    using result_t = std::common_type_t<typename V::value_type, S>;
    dense_vector<result_t> r(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i)
        r(i) = v(i) / alpha;
    return r;
}

} // namespace mtl::vec
