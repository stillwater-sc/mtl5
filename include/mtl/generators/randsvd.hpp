#pragma once
// MTL5 — Random matrix with prescribed singular values (factory, returns dense2D)
// Constructs A = U * Sigma * V^T where U, V are random orthogonal matrices.
// Most powerful test matrix generator: verify SVD results against known ground truth.
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/generators/randorth.hpp>

namespace mtl::generators {

namespace detail {

/// Generate a sigma vector of length p from condition number kappa and mode.
/// All modes give sigma[0] = 1, sigma[p-1] = 1/kappa.
/// Modes follow MATLAB gallery('randsvd') convention.
template <typename T>
auto make_sigma(std::size_t p, T kappa, int mode) -> std::vector<T> {
    assert(p >= 1);
    assert(kappa >= T(1));

    std::vector<T> sigma(p);

    if (p == 1) {
        sigma[0] = T(1);
        return sigma;
    }

    switch (mode) {
    case 1:
        // One large: sigma = [1, 1/kappa, 1/kappa, ..., 1/kappa]
        sigma[0] = T(1);
        for (std::size_t i = 1; i < p; ++i)
            sigma[i] = T(1) / kappa;
        break;

    case 2:
        // One small: sigma = [1, 1, ..., 1, 1/kappa]
        for (std::size_t i = 0; i + 1 < p; ++i)
            sigma[i] = T(1);
        sigma[p - 1] = T(1) / kappa;
        break;

    case 3:
        // Geometric: sigma_i = kappa^{-(i)/(p-1)}
        for (std::size_t i = 0; i < p; ++i) {
            using std::pow;
            sigma[i] = pow(kappa, -T(i) / T(p - 1));
        }
        break;

    case 4:
        // Arithmetic: sigma_i = 1 - i*(1-1/kappa)/(p-1)
        for (std::size_t i = 0; i < p; ++i)
            sigma[i] = T(1) - T(i) * (T(1) - T(1) / kappa) / T(p - 1);
        break;

    case 5: {
        // Random log-uniform in [1/kappa, 1]
        std::mt19937 gen{std::random_device{}()};
        using std::log;
        T log_inv_kappa = log(T(1) / kappa);
        std::uniform_real_distribution<T> dist(log_inv_kappa, T(0));
        for (std::size_t i = 0; i < p; ++i) {
            using std::exp;
            sigma[i] = exp(dist(gen));
        }
        // Force exact endpoints
        sigma[0] = T(1);
        sigma[p - 1] = T(1) / kappa;
        break;
    }

    default:
        assert(false && "randsvd: mode must be 1-5");
        // Fallback to mode 3
        for (std::size_t i = 0; i < p; ++i) {
            using std::pow;
            sigma[i] = pow(kappa, -T(i) / T(p - 1));
        }
        break;
    }

    return sigma;
}

} // namespace detail

/// Construct a matrix with prescribed singular values.
/// A = U * Sigma * V^T where U (m x m) and V (n x n) are random orthogonal.
/// sigma.size() must equal min(m, n).
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, const std::vector<T>& sigma) {
    std::size_t p = std::min(m, n);
    assert(sigma.size() == p);

    auto U = randorth<T>(m);
    auto V = randorth<T>(n);

    // Build Sigma as m x n diagonal
    mat::dense2D<T> S(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            S(i, j) = math::zero<T>();
    for (std::size_t i = 0; i < p; ++i)
        S(i, i) = sigma[i];

    // A = U * S * V^T
    auto US = U * S;
    mat::dense2D<T> result = US * trans(V);
    return result;
}

/// Construct a square matrix with prescribed condition number kappa and mode.
/// cond(A) = kappa exactly. See make_sigma for mode descriptions (1-5).
template <typename T = double>
auto randsvd(std::size_t n, T kappa, int mode = 3) {
    auto sigma = detail::make_sigma(n, kappa, mode);
    return randsvd<T>(n, n, sigma);
}

/// Construct a rectangular matrix with prescribed condition number kappa and mode.
/// cond(A) = kappa exactly. See make_sigma for mode descriptions (1-5).
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, T kappa, int mode = 3) {
    std::size_t p = std::min(m, n);
    auto sigma = detail::make_sigma(p, kappa, mode);
    return randsvd<T>(m, n, sigma);
}

} // namespace mtl::generators
