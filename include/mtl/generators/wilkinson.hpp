#pragma once
// MTL5 — Wilkinson W+ tridiagonal matrix generator (factory, returns dense2D)
// Tests eigenvalue sensitivity — has nearly-equal eigenvalue pairs.
// Diagonal: [m, m-1, ..., 1, 0, 1, ..., m-1, m] where n = 2m+1
// Sub/super-diagonal: all 1s
#include <cassert>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::generators {

/// Wilkinson W+ tridiagonal matrix of size n x n (n must be odd).
/// Diagonal entries: |i - m| for i = 0..n-1, where m = (n-1)/2.
/// Sub- and super-diagonal entries: all 1s.
template <typename T = double>
auto wilkinson(std::size_t n) {
    assert(n % 2 == 1 && "Wilkinson matrix requires odd dimension");
    assert(n >= 3 && "Wilkinson matrix requires n >= 3");

    mat::dense2D<T> W(n, n);
    std::size_t m = (n - 1) / 2;

    // Initialize to zero
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            W(i, j) = math::zero<T>();

    // Diagonal: |i - m|
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t dist = (i >= m) ? (i - m) : (m - i);
        W(i, i) = T(dist);
    }

    // Sub- and super-diagonal: all 1s
    for (std::size_t i = 0; i + 1 < n; ++i) {
        W(i, i + 1) = math::one<T>();
        W(i + 1, i) = math::one<T>();
    }

    return W;
}

} // namespace mtl::generators
