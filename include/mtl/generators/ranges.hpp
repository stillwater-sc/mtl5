#pragma once
// MTL5 -- range / spacing vector generators (NumPy-style)
// arange, linspace, logspace, geomspace. Each returns a vec::dense_vector<T>
// and is generic over the scalar type T (Universal-free).
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::generators {

/// arange: integer-strided sequence projected into T over the half-open
/// interval [start, stop) (NumPy semantics -- the stop value is excluded).
/// Returns an empty vector if step == 0 or the interval is empty.
template <typename T = double>
vec::dense_vector<T> arange(std::int64_t start, std::int64_t stop, std::int64_t step = 1) {
    if (step == 0) return vec::dense_vector<T>();
    // Count the elements in unsigned arithmetic so it never overflows for valid
    // int64 arguments (spans near INT64_MIN..INT64_MAX, or step == INT64_MIN).
    std::size_t n = 0;
    if (step > 0 && stop > start) {
        std::uint64_t span = static_cast<std::uint64_t>(stop) - static_cast<std::uint64_t>(start);
        std::uint64_t mag  = static_cast<std::uint64_t>(step);
        n = static_cast<std::size_t>((span + mag - 1) / mag);
    } else if (step < 0 && stop < start) {
        std::uint64_t span = static_cast<std::uint64_t>(start) - static_cast<std::uint64_t>(stop);
        std::uint64_t mag  = 0u - static_cast<std::uint64_t>(step);   // |step|, safe for INT64_MIN
        n = static_cast<std::size_t>((span + mag - 1) / mag);
    }
    vec::dense_vector<T> v(n);
    std::int64_t sample = start;
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = T(sample);
        if (i + 1 < n) sample += step;   // skip the unused final increment (would overflow)
    }
    return v;
}

/// linspace: `steps` evenly spaced samples over the interval [start, stop].
/// With endpoint == true (default, NumPy semantics) stop is the last sample;
/// with endpoint == false the interval is treated as half-open.
template <typename T = double>
vec::dense_vector<T> linspace(const T& start, const T& stop, std::size_t steps, bool endpoint = true) {
    static_assert(!std::is_integral_v<T>,
                  "linspace requires a non-integral scalar T; integral T truncates the spacing");
    if (steps == 0) return vec::dense_vector<T>();
    vec::dense_vector<T> v(steps);
    if (steps == 1) { v[0] = start; return v; }
    std::size_t divisor = endpoint ? steps - 1 : steps;   // number of segments
    T step = (stop - start) / T(divisor);
    for (std::size_t i = 0; i < steps; ++i) v[i] = start + T(i) * step;
    if (endpoint) v[steps - 1] = stop;                    // pin the exact endpoint
    return v;
}

/// logspace: `steps` samples whose exponents are evenly spaced over
/// [start, stop]; element i equals base^exponent_i. As in NumPy, `start` and
/// `stop` are exponents, not endpoints (see geomspace for endpoint semantics).
template <typename T = double>
vec::dense_vector<T> logspace(const T& start, const T& stop, std::size_t steps, bool endpoint = true, const T& base = T(10)) {
    static_assert(!std::is_integral_v<T>,
                  "logspace requires a non-integral scalar T; integral T truncates the spacing");
    using std::pow;
    vec::dense_vector<T> e = linspace(start, stop, steps, endpoint);
    for (std::size_t i = 0; i < e.size(); ++i) e[i] = pow(base, e[i]);
    return e;
}

/// geomspace: `steps` samples in a geometric progression from `start` to `stop`
/// (NumPy semantics -- start and stop are the actual endpoints). Requires
/// start and stop to be nonzero and of the same sign.
///
/// Note: unlike the historical Universal implementation (which aliased logspace
/// and therefore treated the endpoints as exponents), this computes a true
/// geometric progression: geomspace(1, 1000, 4) == {1, 10, 100, 1000}.
template <typename T = double>
vec::dense_vector<T> geomspace(const T& start, const T& stop, std::size_t steps, bool endpoint = true) {
    static_assert(!std::is_integral_v<T>,
                  "geomspace requires a non-integral scalar T; integral T truncates the spacing");
    using std::pow;
    // A geometric progression is undefined through zero or across a sign change:
    // ratio = stop/start would divide by zero or, for opposite signs, feed a
    // negative base to pow(ratio, fractional) and yield NaN. Reject up front.
    if (start == T(0) || stop == T(0))
        throw std::invalid_argument("geomspace: endpoints must be nonzero");
    if ((start < T(0)) != (stop < T(0)))
        throw std::invalid_argument("geomspace: endpoints must have the same sign");
    if (steps == 0) return vec::dense_vector<T>();
    vec::dense_vector<T> v(steps);
    if (steps == 1) { v[0] = start; return v; }
    std::size_t divisor = endpoint ? steps - 1 : steps;
    T ratio = stop / start;
    for (std::size_t i = 0; i < steps; ++i) {
        T t = T(i) / T(divisor);
        v[i] = start * pow(ratio, t);
    }
    if (endpoint) v[steps - 1] = stop;                    // pin the exact endpoint
    return v;
}

} // namespace mtl::generators
