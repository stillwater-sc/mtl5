#pragma once
// MTL5 — Preconditioner concept for ITL
#include <concepts>

namespace mtl {

/// A preconditioner provides solve() and adjoint_solve()
template <typename P, typename X>
concept Preconditioner = requires(const P& p, X& x, const X& b) {
    { p.solve(x, b) };
    { p.adjoint_solve(x, b) };
};

} // namespace mtl
