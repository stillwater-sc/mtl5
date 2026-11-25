#pragma once
// MTL5 -- Apply a Krylov solver's operator: a Matrix or a matrix-free one (#278)
#include <cstddef>
#include <type_traits>
#include <mtl/concepts/linear_operator.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trans.hpp>

namespace mtl::itl {

/// y = A * x, for the ITL Krylov solvers.
///
/// A `Matrix` operand -- dense2D, compressed2D, a view -- goes through
/// `mult<Accumulator>`, as the solvers called it before #278: any accumulator
/// reaches the matvec, and `void` keeps mult's BLAS / SIMD / native dispatch, so
/// the result is byte-identical.
///
/// Any other operand must be a `LinearOperator`: only `A * x` is available, and
/// `y` takes its entries. A matrix-free operator does its own accumulation, so a
/// non-void `Accumulator` cannot reach it; it still governs the solver's dot
/// products. This is the contract of the eigensolvers' `ev_matvec` (#591).
template <typename Accumulator = void, typename LinearOp, typename VecIn, typename VecOut>
void apply_operator(const LinearOp& A, const VecIn& x, VecOut& y) {
    if constexpr (Matrix<LinearOp>) {
        mtl::mult<Accumulator>(A, x, y);
    } else {
        static_assert(LinearOperator<LinearOp, VecIn>,
                      "a Krylov solver's operator must be a Matrix or support A * x");
        const auto w = A * x;
        for (std::size_t i = 0; i < static_cast<std::size_t>(y.size()); ++i)
            y(i) = w(i);
    }
}

/// y = A^T * x, for the solvers that need the transpose (bicg, qmr).
///
/// A `Matrix` operand goes through `mult<Accumulator>(trans(A), x, y)`. A
/// matrix-free operator provides its transpose as `trans(A)`, found by
/// argument-dependent lookup, returning an operator with `operator*`; the
/// accumulator then covers only the dot products, as in `apply_operator`.
template <typename Accumulator = void, typename LinearOp, typename VecIn, typename VecOut>
void apply_transpose(const LinearOp& A, const VecIn& x, VecOut& y) {
    if constexpr (Matrix<LinearOp>) {
        mtl::mult<Accumulator>(trans(A), x, y);
    } else {
        static_assert(requires(const LinearOp& a, const VecIn& v) { trans(a) * v; },
                      "a matrix-free operator used with bicg or qmr must provide trans(A), "
                      "found by argument-dependent lookup, supporting trans(A) * x");
        // trans(A) may return an operator by value whose product is a lazy
        // expression referring to it, so keep it alive while w is read.
        auto&& At = trans(A);
        const auto w = At * x;
        for (std::size_t i = 0; i < static_cast<std::size_t>(y.size()); ++i)
            y(i) = w(i);
    }
}

/// y = A^H * x, the adjoint, for the shadow recurrences of bicg and qmr.
///
/// The solvers' dot products conjugate their first argument, so the shadow
/// sequence must be built with A^H. With A^T, bicg and qmr did not converge on
/// a complex non-Hermitian system: 200 iterations at a residual of 2.1 where
/// cgs and gmres took 3 (issue 924). For a real element type this is
/// apply_transpose. For a complex one it is A^H x = conj(A^T conj(x)), which
/// needs only the transpose, so a matrix-free operator still provides just
/// trans(A). `tmp` holds conj(x): x's size for a complex type, unused otherwise.
template <typename Accumulator = void, typename LinearOp, typename VecIn, typename VecOut, typename VecTmp>
void apply_adjoint(const LinearOp& A, const VecIn& x, VecOut& y, VecTmp& tmp) {
    using value_type = typename VecOut::value_type;
    if constexpr (!is_complex_v<value_type>) {
        apply_transpose<Accumulator>(A, x, y);
    } else {
        using conj_t = functor::scalar::conj<value_type>;
        const std::size_t n = static_cast<std::size_t>(x.size());
        for (std::size_t i = 0; i < n; ++i)
            tmp(i) = conj_t::apply(x(i));
        apply_transpose<Accumulator>(A, tmp, y);
        for (std::size_t i = 0; i < static_cast<std::size_t>(y.size()); ++i)
            y(i) = conj_t::apply(y(i));
    }
}

} // namespace mtl::itl
