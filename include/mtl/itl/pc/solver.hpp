#pragma once
// MTL5 — Thin free-function wrappers for preconditioner solve
#include <mtl/concepts/vector.hpp>

namespace mtl::itl::pc {

/// Free-function wrapper: solve(P, x, b) calls P.solve(x, b)
template <typename Preconditioner, typename VectorOut, typename VectorIn>
void solve(const Preconditioner& P, VectorOut& x, const VectorIn& b) {
    P.solve(x, b);
}

/// Free-function wrapper: adjoint_solve(P, x, b) calls P.adjoint_solve(x, b)
template <typename Preconditioner, typename VectorOut, typename VectorIn>
void adjoint_solve(const Preconditioner& P, VectorOut& x, const VectorIn& b) {
    P.adjoint_solve(x, b);
}

} // namespace mtl::itl::pc
