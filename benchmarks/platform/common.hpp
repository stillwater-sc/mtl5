#pragma once
// mtl5_bench -- helpers shared by every entries_*.hpp (#721).
//
// Inputs are deterministic (xorshift64 from benchmarks/harness/generators.hpp),
// so every machine times the same problem. Checks are against long-double
// references with the standard error bound gamma_k = k*eps/(1 - k*eps), via
// error_tracker: the tolerance is a theorem about the computation, not a guess,
// and it holds for every summation order -- blocked, SIMD, threaded or BLAS.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <benchmarks/harness/generators.hpp>
#include <benchmarks/platform/registry.hpp>

namespace mtl::bench::platform {

template <typename T> const char* dtype_name();
template <> inline const char* dtype_name<double>() { return "f64"; }
template <> inline const char* dtype_name<float>()  { return "f32"; }

template <typename T>
double eps() { return static_cast<double>(std::numeric_limits<T>::epsilon()); }

/// Running worst componentwise error, against the gamma_k tolerance.
class error_tracker {
public:
    error_tracker(double tol, std::string what) { r_.tolerance = tol; r_.what = std::move(what); }
    void add(long double got, long double ref, long double absterms) {
        const long double denom = absterms > 0 ? absterms : 1;
        const double e = static_cast<double>(std::fabs(got - ref) / denom);
        if (!(e <= r_.error)) r_.error = std::isnan(e) ? std::numeric_limits<double>::infinity() : e;
    }
    check_result result() { r_.ok = r_.error <= r_.tolerance; return r_; }
private:
    check_result r_;
};

/// The gamma_k bound, capped at 0.5. Uncapped, 4*k*eps exceeds 1 for f32 once
/// k reaches about 2^22, and a check whose tolerance exceeds 1 passes a result
/// of zero. With the cap, a zero result (error 1 whenever the reference is not
/// dominated by cancellation) always fails.
template <typename T> double tol(std::size_t k) { return std::min(4.0 * double(k + 2) * eps<T>(), 0.5); }

/// A vector in [0, 1): for the reductions whose check would otherwise be blind.
/// With mean-zero inputs, |sum x_i y_i| is about sqrt(n) smaller than
/// sum |x_i y_i|, so a kernel returning 0 has an error of only ~1/sqrt(n) and
/// passes any tolerance above that. With nonnegative inputs the reference
/// equals the sum of magnitudes, and a zero result has error 1.
template <typename T>
vec::dense_vector<T> nonneg_vec_of(std::size_t n, std::uint64_t seed) {
    auto v = make_random_vector<T>(n, seed);
    for (std::size_t i = 0; i < n; ++i) v(i) = v(i) < T(0) ? -v(i) : v(i);
    return v;
}

template <typename T>
vec::dense_vector<T> vec_of(std::size_t n, std::uint64_t seed) { return make_random_vector<T>(n, seed); }
template <typename T>
mat::dense2D<T> mat_of(std::size_t r, std::size_t c, std::uint64_t seed) { return make_random_matrix<T>(r, c, seed); }

/// Lower-triangular, well conditioned: diagonal in [1, 2], off-diagonal in
/// [-1/n, 1/n], zero above. Triangular solves on it lose no accuracy.
template <typename T>
mat::dense2D<T> lower_tri(std::size_t n, std::uint64_t seed) {
    xorshift64 g(seed);
    mat::dense2D<T> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = j > i ? T(0)
                    : j == i ? T(1.0 + g.next_double())
                             : T(g.next_double(-1.0, 1.0) / double(n));
    return L;
}

template <typename T>
mat::dense2D<T> symmetric(std::size_t n, std::uint64_t seed) {
    auto A = mat_of<T>(n, n, seed);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) A(j, i) = A(i, j);
    return A;
}

// Sizes. Level 1 and 2 are memory-bound, so their medium and large sizes are
// chosen to be far larger than any last-level cache in the machines this runs
// on (64 MiB per f64 vector, 128 MiB per f64 matrix at medium), so they measure
// DRAM everywhere. `small` is cache-resident on purpose: it is a smoke run. The
// runner flags any result whose working set fits in the LLC, because placing a
// cache-resident rate on the DRAM roofline reads as > 100%.
inline std::size_t n_vec(size_class c) {
    return c == size_class::small ? (std::size_t(1) << 15)
         : c == size_class::medium ? (std::size_t(1) << 23) : (std::size_t(1) << 25);
}
inline std::size_t n_mv(size_class c) {
    return c == size_class::small ? 512 : c == size_class::medium ? 4096 : 8192;
}
inline std::size_t n_mm(size_class c) {
    return c == size_class::small ? 128 : c == size_class::medium ? 512 : 1024;
}
inline std::string vshape(std::size_t n) { return "n=" + std::to_string(n); }

/// 64 (i, j) positions spread over an n x n result, deterministic.
template <typename Fn>
void for_samples(std::size_t n, Fn fn) {
    xorshift64 g(0x5A3D);
    for (int s = 0; s < 64; ++s) {
        const auto i = static_cast<std::size_t>(g.next_double() * double(n));
        const auto j = static_cast<std::size_t>(g.next_double() * double(n));
        fn(std::min(i, n - 1), std::min(j, n - 1));
    }
}

} // namespace mtl::bench::platform
