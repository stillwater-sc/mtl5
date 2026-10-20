#pragma once
// MTL5 -- Apply a Krylov solver's operator: a Matrix or a matrix-free one (#278)
#include <cstddef>
#include <type_traits>
#include <mtl/concepts/linear_operator.hpp>
#include <mtl/concepts/matrix.hpp>
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
        const auto w = trans(A) * x;
        for (std::size_t i = 0; i < static_cast<std::size_t>(y.size()); ++i)
            y(i) = w(i);
    }
}

} // namespace mtl::itl
