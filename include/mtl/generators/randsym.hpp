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

/// A symmetric matrix with eigenvalues in [1/kappa, 1] from mode (the randsvd
/// modes 1-5, see make_sigma), drawing from `gen`.
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

} // namespace mtl::generators
