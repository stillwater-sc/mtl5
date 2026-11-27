#pragma once
// MTL5 -- Random symmetric matrix with prescribed eigenvalues (factory, returns dense2D)
// Constructs A = Q * Lambda * Q^T where Q is a random orthogonal matrix, then
// averages A(i,j) and A(j,i) so the result is exactly symmetric.
// Each generator has a form without a seed, one with a std::uint64_t seed and
// one drawing from a std::mt19937_64; see randsvd.hpp.
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/random.hpp>
#include <mtl/generators/randsvd.hpp>  // detail::make_sigma, detail::random_symmetric

namespace mtl::generators {

/// A symmetric matrix with prescribed eigenvalues, Q drawn from `gen`.
/// Throws std::invalid_argument unless eigenvalues.size() == n.
template <typename T = double>
auto randsym(std::size_t n, const std::vector<T>& eigenvalues, std::mt19937_64& gen) {
    if (eigenvalues.size() != n)
        throw std::invalid_argument("randsym: need n eigenvalues");
    return detail::random_symmetric<T>(n, eigenvalues, gen);
}

/// Prescribed eigenvalues, reproducibly from `seed`.
template <typename T = double>
auto randsym(std::size_t n, const std::vector<T>& eigenvalues, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return randsym<T>(n, eigenvalues, gen);
}

/// Prescribed eigenvalues, non-deterministically.
template <typename T = double>
auto randsym(std::size_t n, const std::vector<T>& eigenvalues) {
    auto gen = mtl::detail::random_engine();
    return randsym<T>(n, eigenvalues, gen);
}

/// A symmetric POSITIVE DEFINITE matrix with eigenvalues in [1/kappa, 1] from
/// mode (the randsvd modes 1-5, see make_sigma), drawing from `gen`: the same
/// kind of matrix as randspd. For an indefinite one with this conditioning, see
/// randsym_indefinite below.
template <typename T = double>
auto randsym(std::size_t n, T kappa, int mode, std::mt19937_64& gen) {
    auto eigenvalues = detail::make_sigma(n, kappa, mode, gen);
    return randsym<T>(n, eigenvalues, gen);
}

/// Eigenvalues from kappa and mode, reproducibly from `seed`.
template <typename T = double>
auto randsym(std::size_t n, T kappa, int mode, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return randsym<T>(n, kappa, mode, gen);
}

/// Eigenvalues from kappa and mode, non-deterministically.
template <typename T = double>
auto randsym(std::size_t n, T kappa, int mode = 3) {
    auto gen = mtl::detail::random_engine();
    return randsym<T>(n, kappa, mode, gen);
}

/// A symmetric INDEFINITE matrix: the eigenvalues of randsym(n, kappa, mode),
/// with `num_negative` of them negated (issue 961). The negated positions are
/// a random subset of the spectrum, drawn from `gen` after it, as LAPACK's
/// DLATMS does with RSIGN = 'T'. So
///   - for n >= 2, cond_2(A) = max |lambda| / min |lambda| = kappa, as for
///     randsym: the signs do not change the magnitudes;
///   - the inertia is (n - num_negative, num_negative, 0).
/// n = 1 gives the 1x1 matrix [1], or [-1] with one negative, whatever kappa
/// is (make_sigma's single value is 1), so its condition number is 1; n = 0
/// gives an empty matrix, with num_negative = 0.
/// The subset is a partial Fisher-Yates shuffle over the raw engine stream
/// (detail::random_unit), so a seed gives the same matrix with every compiler
/// and standard library. With num_negative == 0 nothing is drawn for it, and
/// the matrix is randsym(n, kappa, mode)'s for the same engine state.
/// Throws std::invalid_argument for num_negative > n, and as randsym does for
/// kappa and mode.
///
/// (A distinct name, not a randsym overload: randsym(n, kappa, mode,
/// num_negative) would have the parameter types of the seeded
/// randsym(n, kappa, mode, std::uint64_t seed) where std::size_t is
/// std::uint64_t.)
template <typename T = double>
auto randsym_indefinite(std::size_t n, T kappa, int mode, std::size_t num_negative,
                        std::mt19937_64& gen) {
    if (num_negative > n)
        throw std::invalid_argument("randsym_indefinite: num_negative must be <= n");
    auto eigenvalues = detail::make_sigma(n, kappa, mode, gen);
    if (num_negative > 0) {
        // The first num_negative entries of a partial Fisher-Yates shuffle of
        // 0..n-1: each step draws j uniform in [i, n).
        std::vector<std::size_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        for (std::size_t i = 0; i < num_negative; ++i) {
            const std::size_t span = n - i;
            std::size_t j = i + static_cast<std::size_t>(mtl::detail::random_unit(gen) * double(span));
            if (j >= n) j = n - 1;   // random_unit < 1, but guard the product's rounding
            std::swap(idx[i], idx[j]);
            eigenvalues[idx[i]] = -eigenvalues[idx[i]];
        }
    }
    return detail::random_symmetric<T>(n, eigenvalues, gen);
}

/// randsym_indefinite reproducibly from `seed`.
template <typename T = double>
auto randsym_indefinite(std::size_t n, T kappa, int mode, std::size_t num_negative,
                        std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return randsym_indefinite<T>(n, kappa, mode, num_negative, gen);
}

/// randsym_indefinite non-deterministically.
template <typename T = double>
auto randsym_indefinite(std::size_t n, T kappa, int mode, std::size_t num_negative) {
    auto gen = mtl::detail::random_engine();
    return randsym_indefinite<T>(n, kappa, mode, num_negative, gen);
}

} // namespace mtl::generators
