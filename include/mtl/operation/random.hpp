#pragma once
// MTL5 -- Fill collections with uniform random values
//
// Every function has three forms:
//
//   f(..., lo, hi)                        non-deterministic: a std::mt19937_64
//                                         seeded from std::random_device
//   f(..., lo, hi, std::uint64_t seed)    reproducible: std::mt19937_64(seed)
//   f(..., lo, hi, std::mt19937_64& gen)  draws from the caller's engine, so
//                                         several calls can share one stream
//
// Portable randomness (#844). Each value is built from the raw std::mt19937_64
// stream, which the standard specifies bit for bit, not from
// std::uniform_real_distribution, whose algorithm each standard library chooses
// for itself. So a seed gives the same values with every compiler and standard
// library, as in generators/ill_conditioned.hpp. A draw is u = (gen() >> 11) *
// 2^-53, uniform on [0, 1), and the value is lo * (1 - u) + hi * u, rounded to
// the value type; a value that rounds up to hi is drawn again, so every value is
// in [lo, hi). Matrices are filled row by row, whatever their orientation.
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl {

namespace detail {

/// A std::mt19937_64 seeded from std::random_device, for the calls without a seed.
inline std::mt19937_64 random_engine() {
    std::random_device rd;
    const std::uint64_t hi = rd(), lo = rd();   // random_device yields 32 bits
    return std::mt19937_64((hi << 32) ^ lo);
}

/// Uniform on [0, 1): the top 53 bits of one 64-bit draw.
inline double random_unit(std::mt19937_64& gen) {
    return static_cast<double>(gen() >> 11) * 0x1p-53;
}

/// Throws unless lo <= hi (a NaN bound fails too).
template <typename T>
void check_random_bounds(const T& lo, const T& hi, const char* who) {
    static_assert(!std::is_integral_v<T>,
                  "random fills need a non-integral value type");
    if (!(lo <= hi))
        throw std::invalid_argument(std::string(who) + ": requires lo <= hi");
}

/// One value uniform on [lo, hi), or lo when lo == hi. Bounds already checked.
template <typename T>
T random_uniform(std::mt19937_64& gen, const T& lo, const T& hi) {
    if (!(lo < hi)) return lo;
    for (;;) {
        const T u = static_cast<T>(random_unit(gen));
        const T v = lo * (T(1) - u) + hi * u;   // no hi - lo, which can overflow
        if (v < hi) return v;
    }
}

} // namespace detail

/// Fill a vector with uniform random values in [lo, hi), drawn from `gen`.
template <Vector V>
void fill_random(V& v, typename V::value_type lo, typename V::value_type hi,
                 std::mt19937_64& gen) {
    detail::check_random_bounds(lo, hi, "fill_random");
    for (typename V::size_type i = 0; i < v.size(); ++i)
        v(i) = detail::random_uniform(gen, lo, hi);
}

/// Fill a matrix with uniform random values in [lo, hi), drawn from `gen`
/// row by row.
template <Matrix M>
void fill_random(M& A, typename M::value_type lo, typename M::value_type hi,
                 std::mt19937_64& gen) {
    detail::check_random_bounds(lo, hi, "fill_random");
    for (typename M::size_type r = 0; r < A.num_rows(); ++r)
        for (typename M::size_type c = 0; c < A.num_cols(); ++c)
            A(r, c) = detail::random_uniform(gen, lo, hi);
}

/// Fill a vector with uniform random values in [lo, hi), reproducibly from `seed`.
template <Vector V>
void fill_random(V& v, typename V::value_type lo, typename V::value_type hi,
                 std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    fill_random(v, lo, hi, gen);
}

/// Fill a matrix with uniform random values in [lo, hi), reproducibly from `seed`.
template <Matrix M>
void fill_random(M& A, typename M::value_type lo, typename M::value_type hi,
                 std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    fill_random(A, lo, hi, gen);
}

/// Fill a vector with uniform random values in [lo, hi), non-deterministically.
template <Vector V>
void fill_random(V& v, typename V::value_type lo = typename V::value_type(0),
                       typename V::value_type hi = typename V::value_type(1)) {
    auto gen = detail::random_engine();
    fill_random(v, lo, hi, gen);
}

/// Fill a matrix with uniform random values in [lo, hi), non-deterministically.
template <Matrix M>
void fill_random(M& A, typename M::value_type lo = typename M::value_type(0),
                       typename M::value_type hi = typename M::value_type(1)) {
    auto gen = detail::random_engine();
    fill_random(A, lo, hi, gen);
}

/// A dense_vector of size n with uniform values in [lo, hi), drawn from `gen`.
template <typename T = double>
auto random_vector(std::size_t n, T lo, T hi, std::mt19937_64& gen) {
    vec::dense_vector<T> v(n);
    fill_random(v, lo, hi, gen);
    return v;
}

/// A dense_vector of size n with uniform values in [lo, hi), reproducibly from `seed`.
template <typename T = double>
auto random_vector(std::size_t n, T lo, T hi, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return random_vector<T>(n, lo, hi, gen);
}

/// A dense_vector of size n with uniform values in [lo, hi), non-deterministically.
template <typename T = double>
auto random_vector(std::size_t n, T lo = T(0), T hi = T(1)) {
    auto gen = detail::random_engine();
    return random_vector<T>(n, lo, hi, gen);
}

/// An m x n dense2D with uniform values in [lo, hi), drawn from `gen` row by row.
template <typename T = double>
auto random_matrix(std::size_t m, std::size_t n, T lo, T hi, std::mt19937_64& gen) {
    mat::dense2D<T> A(m, n);
    fill_random(A, lo, hi, gen);
    return A;
}

/// An m x n dense2D with uniform values in [lo, hi), reproducibly from `seed`.
template <typename T = double>
auto random_matrix(std::size_t m, std::size_t n, T lo, T hi, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return random_matrix<T>(m, n, lo, hi, gen);
}

/// An m x n dense2D with uniform values in [lo, hi), non-deterministically.
template <typename T = double>
auto random_matrix(std::size_t m, std::size_t n, T lo = T(0), T hi = T(1)) {
    auto gen = detail::random_engine();
    return random_matrix<T>(m, n, lo, hi, gen);
}

} // namespace mtl
