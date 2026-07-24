#pragma once
// MTL5 -- range / spacing vector generators (NumPy-style)
// arange, linspace, logspace, geomspace. Each returns a vec::dense_vector<T>
// and is generic over the scalar type T (Universal-free).
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::generators {

/// arange: integer-strided sequence projected into T over the half-open
/// interval [start, stop) (NumPy semantics -- the stop value is excluded).
/// Returns an empty vector if step == 0 or the interval is empty.
template <typename T = double>
vec::dense_vector<T> arange(std::int64_t start, std::int64_t stop, std::int64_t step = 1) {
    if (step == 0) return vec::dense_vector<T>();
    std::size_t n = 0;
    if (step > 0 && stop > start)
        n = static_cast<std::size_t>((stop - start + step - 1) / step);
    else if (step < 0 && stop < start)
        n = static_cast<std::size_t>((start - stop + (-step) - 1) / (-step));
    vec::dense_vector<T> v(n);
    std::int64_t sample = start;
    for (std::size_t i = 0; i < n; ++i) { v[i] = T(sample); sample += step; }
    return v;
}

/// linspace: `steps` evenly spaced samples over the interval [start, stop].
/// With endpoint == true (default, NumPy semantics) stop is the last sample;
/// with endpoint == false the interval is treated as half-open.
template <typename T = double>
vec::dense_vector<T> linspace(const T& start, const T& stop, std::size_t steps, bool endpoint = true) {
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
    using std::pow;
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
