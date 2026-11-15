#pragma once
// MTL5 -- Random matrix with prescribed singular values (factory, returns dense2D)
// Constructs A = U * Sigma * V^T where U, V are random orthogonal matrices.
// Most powerful test matrix generator: verify SVD results against known ground truth.
//
// Every generator has a form without a seed (non-deterministic), a form taking
// a std::uint64_t seed, and a form drawing from a caller's std::mt19937_64.
// The seeded forms draw portably (operation/random.hpp), so a seed gives the
// same U, V and mode-5 sigma with every compiler and standard library (#844).
// For the square kappa form, pass the seed as randsvd(n, n, kappa, mode, seed):
// a randsvd(n, kappa, mode, seed) overload would be ambiguous with the
// rectangular randsvd(m, n, kappa, mode).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/operation/random.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/generators/randorth.hpp>

namespace mtl::generators {

namespace detail {

/// Generate a sigma vector of length p from condition number kappa and mode.
/// All modes give sigma[0] = 1, sigma[p-1] = 1/kappa (p >= 2); p = 1 gives {1}
/// and p = 0 an empty vector. Modes follow MATLAB gallery('randsvd'):
///   1  one large:  {1, 1/kappa, ..., 1/kappa}
///   2  one small:  {1, ..., 1, 1/kappa}
///   3  geometric:  sigma_i = kappa^(-i/(p-1))
///   4  arithmetic: sigma_i = 1 - i (1 - 1/kappa) / (p-1)
///   5  random, with log(sigma) uniform on [log(1/kappa), 0), drawn from `gen`
/// Throws std::invalid_argument for kappa < 1 (or NaN) and a mode outside 1-5.
template <typename T>
auto make_sigma(std::size_t p, T kappa, int mode, std::mt19937_64& gen) -> std::vector<T> {
    if (!(kappa >= T(1)))
        throw std::invalid_argument("randsvd: kappa must be >= 1");
    if (mode < 1 || mode > 5)
        throw std::invalid_argument("randsvd: mode must be 1-5, got " + std::to_string(mode));

    std::vector<T> sigma(p);
    if (p == 0) return sigma;
    if (p == 1) {
        sigma[0] = T(1);
        return sigma;
    }

    switch (mode) {
    case 1:
        sigma[0] = T(1);
        for (std::size_t i = 1; i < p; ++i)
            sigma[i] = T(1) / kappa;
        break;

    case 2:
        for (std::size_t i = 0; i + 1 < p; ++i)
            sigma[i] = T(1);
        sigma[p - 1] = T(1) / kappa;
        break;

    case 3:
        for (std::size_t i = 0; i < p; ++i) {
            using std::pow;
            sigma[i] = pow(kappa, -T(i) / T(p - 1));
        }
        break;

    case 4:
        for (std::size_t i = 0; i < p; ++i)
            sigma[i] = T(1) - T(i) * (T(1) - T(1) / kappa) / T(p - 1);
        break;

    default: {   // 5
        using std::log;
        using std::exp;
        const T log_inv_kappa = log(T(1) / kappa);
        sigma[0] = T(1);
        for (std::size_t i = 1; i + 1 < p; ++i)
            sigma[i] = exp(mtl::detail::random_uniform(gen, log_inv_kappa, T(0)));
        sigma[p - 1] = T(1) / kappa;   // exact endpoints
        break;
    }
    }
    return sigma;
}

/// make_sigma with a non-deterministic engine for mode 5.
template <typename T>
auto make_sigma(std::size_t p, T kappa, int mode) -> std::vector<T> {
    auto gen = mtl::detail::random_engine();
    return make_sigma(p, kappa, mode, gen);
}

/// An m x n matrix of zeros with `d` on its diagonal.
template <typename T>
mat::dense2D<T> diagonal_matrix(std::size_t m, std::size_t n, const std::vector<T>& d) {
    mat::dense2D<T> D(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            D(i, j) = math::zero<T>();
    for (std::size_t i = 0; i < d.size(); ++i)
        D(i, i) = d[i];
    return D;
}

/// Q * diag(lambda) * Q^T for a random orthogonal Q from `gen`, made exactly
/// symmetric: the product is not bitwise symmetric, so each pair of entries is
/// replaced by its average.
template <typename T>
mat::dense2D<T> random_symmetric(std::size_t n, const std::vector<T>& lambda,
                                 std::mt19937_64& gen) {
    auto Q = randorth<T>(n, gen);
    auto QL = Q * diagonal_matrix<T>(n, n, lambda);
    mat::dense2D<T> A = QL * trans(Q);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const T a = (A(i, j) + A(j, i)) / T(2);
            A(i, j) = a;
            A(j, i) = a;
        }
    return A;
}

} // namespace detail

/// Construct a matrix with prescribed singular values, drawing U and V from `gen`.
/// A = U * Sigma * V^T where U (m x m) and V (n x n) are random orthogonal.
/// Throws std::invalid_argument unless sigma.size() == min(m, n).
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, const std::vector<T>& sigma, std::mt19937_64& gen) {
    if (sigma.size() != std::min(m, n))
        throw std::invalid_argument("randsvd: sigma must have min(m, n) entries");

    auto U = randorth<T>(m, gen);
    auto V = randorth<T>(n, gen);
    auto US = U * detail::diagonal_matrix<T>(m, n, sigma);
    mat::dense2D<T> result = US * trans(V);
    return result;
}

/// Prescribed singular values, reproducibly from `seed`.
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, const std::vector<T>& sigma, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return randsvd<T>(m, n, sigma, gen);
}

/// Prescribed singular values, non-deterministically.
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, const std::vector<T>& sigma) {
    auto gen = mtl::detail::random_engine();
    return randsvd<T>(m, n, sigma, gen);
}

/// A rectangular matrix with condition number kappa, singular values from mode
/// (see make_sigma), drawing from `gen`.
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, T kappa, int mode, std::mt19937_64& gen) {
    auto sigma = detail::make_sigma(std::min(m, n), kappa, mode, gen);
    return randsvd<T>(m, n, sigma, gen);
}

/// A rectangular matrix with condition number kappa, reproducibly from `seed`.
/// Also the seeded form of the square matrix: randsvd(n, n, kappa, mode, seed).
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, T kappa, int mode, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return randsvd<T>(m, n, kappa, mode, gen);
}

/// A rectangular matrix with condition number kappa, non-deterministically.
/// cond(A) = kappa up to rounding. See make_sigma for the modes (1-5).
template <typename T = double>
auto randsvd(std::size_t m, std::size_t n, T kappa, int mode = 3) {
    auto gen = mtl::detail::random_engine();
    return randsvd<T>(m, n, kappa, mode, gen);
}

/// A square matrix with condition number kappa, drawing from `gen`.
template <typename T = double>
auto randsvd(std::size_t n, T kappa, int mode, std::mt19937_64& gen) {
    return randsvd<T>(n, n, kappa, mode, gen);
}

/// A square matrix with condition number kappa, non-deterministically.
/// cond(A) = kappa up to rounding. See make_sigma for the modes (1-5).
template <typename T = double>
auto randsvd(std::size_t n, T kappa, int mode = 3) {
    auto gen = mtl::detail::random_engine();
    return randsvd<T>(n, n, kappa, mode, gen);
}

} // namespace mtl::generators
