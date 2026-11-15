#pragma once
// MTL5 -- Wilkinson W+ tridiagonal matrix generator (factory, returns dense2D)
// Tests eigenvalue sensitivity -- has nearly-equal eigenvalue pairs.
#include <cstddef>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::generators {

/// Wilkinson W+ tridiagonal matrix of size n x n, as MATLAB's wilkinson(n):
/// diagonal abs(-(n-1)/2 : (n-1)/2), that is |i - (n-1)/2| for i = 0..n-1, and
/// all 1s on the sub- and super-diagonal. For odd n = 2m+1 the diagonal is
/// m, m-1, ..., 1, 0, 1, ..., m; for even n it is half-integers, e.g.
/// 1.5, 0.5, 0.5, 1.5 for n = 4. Throws std::invalid_argument for n = 0, and
/// for an even n with a value type that cannot hold 1/2 (an integer type).
template <typename T = double>
auto wilkinson(std::size_t n) {
    if (n == 0)
        throw std::invalid_argument("wilkinson: order must be >= 1");
    if (n % 2 == 0 && T(1) / T(2) == T(0))
        throw std::invalid_argument("wilkinson: an even order needs a value type that holds 1/2");

    mat::dense2D<T> W(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            W(i, j) = math::zero<T>();

    // |i - (n-1)/2| = |2i - (n-1)| / 2, in integers until the last step
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t twice = (2 * i >= n - 1) ? 2 * i - (n - 1) : (n - 1) - 2 * i;
        W(i, i) = (twice % 2 == 0) ? T(twice / 2) : T(twice) / T(2);
    }

    for (std::size_t i = 0; i + 1 < n; ++i) {
        W(i, i + 1) = math::one<T>();
        W(i + 1, i) = math::one<T>();
    }
    return W;
}

} // namespace mtl::generators
