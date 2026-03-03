#pragma once
// MTL5 — Matrix multiplication: mat*vec and mat*mat into pre-allocated output
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// mat*vec multiply into pre-allocated y: y = A * x
template <Matrix M, Vector VIn, Vector VOut>
void mult(const M& A, const VIn& x, VOut& y) {
    assert(A.num_cols() == x.size());
    assert(A.num_rows() == y.size());
    using T = typename VOut::value_type;
    for (typename M::size_type r = 0; r < A.num_rows(); ++r) {
        auto acc = math::zero<T>();
        for (typename M::size_type c = 0; c < A.num_cols(); ++c) {
            acc += A(r, c) * x(c);
        }
        y(r) = acc;
    }
}

/// mat*mat multiply into pre-allocated C: C = A * B
template <Matrix MA, Matrix MB, Matrix MC>
void mult(const MA& A, const MB& B, MC& C) {
    assert(A.num_cols() == B.num_rows());
    assert(A.num_rows() == C.num_rows());
    assert(B.num_cols() == C.num_cols());
    using T = typename MC::value_type;
    for (typename MC::size_type r = 0; r < C.num_rows(); ++r) {
        for (typename MC::size_type c = 0; c < C.num_cols(); ++c) {
            auto acc = math::zero<T>();
            for (typename MA::size_type k = 0; k < A.num_cols(); ++k) {
                acc += A(r, k) * B(k, c);
            }
            C(r, c) = acc;
        }
    }
}

} // namespace mtl
