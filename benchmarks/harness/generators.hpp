#pragma once
// MTL5 Benchmark Harness -- Deterministic matrix/vector generators
// Produces repeatable test data for fair cross-backend comparison.
#include <cstddef>
#include <cstdint>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::bench {

/// Simple xorshift64* PRNG for deterministic, fast generation.
/// No dependency on <random> — avoids platform-specific distributions.
class xorshift64 {
public:
    explicit xorshift64(std::uint64_t seed = 0x12345678DEADBEEF) : state_(seed) {}

    /// Generate a double in [0, 1)
    double next_double() {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        auto x = state_ * UINT64_C(0x2545F4914F6CDD1D);
        return static_cast<double>(x >> 11) * 0x1.0p-53;
    }

    /// Generate a double in [lo, hi)
    double next_double(double lo, double hi) {
        return lo + next_double() * (hi - lo);
    }

private:
    std::uint64_t state_;
};

/// Generate an n x n dense matrix with random entries in [-1, 1]
template <typename T = double>
mat::dense2D<T> make_random_matrix(std::size_t rows, std::size_t cols,
                                    std::uint64_t seed = 42) {
    xorshift64 rng(seed);
    mat::dense2D<T> A(rows, cols);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            A(r, c) = static_cast<T>(rng.next_double(-1.0, 1.0));
    return A;
}

/// Generate an n x n symmetric positive definite matrix: A = B^T * B + n*I
/// Guarantees well-conditioned SPD for Cholesky/eigenvalue benchmarks.
template <typename T = double>
mat::dense2D<T> make_spd_matrix(std::size_t n, std::uint64_t seed = 42) {
    auto B = make_random_matrix<T>(n, n, seed);
    mat::dense2D<T> A(n, n);
    // A = B^T * B
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto acc = math::zero<T>();
            for (std::size_t k = 0; k < n; ++k)
                acc += B(k, i) * B(k, j);
            A(i, j) = acc;
        }
    }
    // Add n*I for numerical stability
    for (std::size_t i = 0; i < n; ++i)
        A(i, i) += static_cast<T>(n);
    return A;
}

/// Generate a dense vector with random entries in [-1, 1]
template <typename T = double>
vec::dense_vector<T> make_random_vector(std::size_t n, std::uint64_t seed = 123) {
    xorshift64 rng(seed);
    vec::dense_vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v(i) = static_cast<T>(rng.next_double(-1.0, 1.0));
    return v;
}

// -- Column-major variants for LAPACK benchmarks ---------------------------

/// Column-major matrix parameter type
using col_major_params = mat::parameters<tag::col_major>;

/// Generate an n x n column-major dense matrix with random entries in [-1, 1]
template <typename T = double>
mat::dense2D<T, col_major_params> make_random_matrix_colmaj(
    std::size_t rows, std::size_t cols, std::uint64_t seed = 42) {
    xorshift64 rng(seed);
    mat::dense2D<T, col_major_params> A(rows, cols);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            A(r, c) = static_cast<T>(rng.next_double(-1.0, 1.0));
    return A;
}

/// Generate an n x n column-major SPD matrix: A = B^T * B + n*I
template <typename T = double>
mat::dense2D<T, col_major_params> make_spd_matrix_colmaj(
    std::size_t n, std::uint64_t seed = 42) {
    auto B = make_random_matrix_colmaj<T>(n, n, seed);
    mat::dense2D<T, col_major_params> A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto acc = math::zero<T>();
            for (std::size_t k = 0; k < n; ++k)
                acc += B(k, i) * B(k, j);
            A(i, j) = acc;
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        A(i, i) += static_cast<T>(n);
    return A;
}

} // namespace mtl::bench
