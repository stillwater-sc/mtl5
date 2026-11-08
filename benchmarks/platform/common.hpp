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
#include <utility>
#include <vector>

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

/// Both checks must pass. The one reported is the one that failed, or, when
/// both passed, the one nearer its tolerance.
inline check_result both(check_result a, check_result b) {
    const auto ratio = [](const check_result& r) {
        return r.tolerance > 0 ? r.error / r.tolerance : r.error;
    };
    const bool ok = a.ok && b.ok;
    check_result r = (!b.ok && a.ok) || (a.ok == b.ok && ratio(b) > ratio(a)) ? std::move(b) : std::move(a);
    r.ok = ok;
    return r;
}

// -- Freivalds' check -----------------------------------------------------------
//
// A sampled check reads 64 entries of an n x n result, so an error confined to
// a few rows or one edge tile -- the remainder handling of a blocked kernel, a
// wrong mirror of one triangle -- passes with high probability. The integer
// GEMM check missed exactly that in M6. Freivalds' check (1977) covers every
// entry at O(n^2) cost: for a probe vector x, compare C x with P x, where P is
// the exact product applied factor by factor (A (B x) for P = A B), never
// formed.
//
// COMPONENTWISE (freivalds). With the probe's entries positive,
//   |(C x - P x)_i| <= sum_j |C - P|_ij x_j <= gamma_k (|A| |B| x)_i,
// so the gamma_k bound on every entry of C is checked, row by row, against
// |P| x = |A| (|B| x). A single wrong entry d at (i, j) moves row i by d x_j,
// with x_j >= 1; the price of covering every entry is that the row's
// denominator sums n terms, so the smallest error seen is ~n times larger than
// an entrywise check's. The entries that have a sampled check keep it: the
// samples are tight, the probe is complete, and `both` requires each.
//
// NORMWISE (freivalds_normwise). For a residual R that must be small in norm --
// A X - I for inv, Q T Q^T - A for Schur -- |R x| <= |R| |x| in the matching
// norm: the infinity norm for a bound in |.|_inf, the 2-norm for a bound in
// |.|_F (|R|_2 <= |R|_F). That checks the normwise bound itself, which is what
// the backward-error theorems state, on all of R.
//
// The reference rounds too: C x, B x and A (B x) are long-double sums of
// length up to kref. That adds tol<long double>(kref) to the tolerance, which
// is negligible where long double is wider than double and is the honest bound
// where it is not (MSVC, Apple ARM64).

using ldvec = std::vector<long double>;

/// A deterministic probe vector with entries in [1, 2).
inline ldvec probe(std::size_t n, std::uint64_t seed) {
    xorshift64 g(seed);
    ldvec x(n);
    for (auto& v : x) v = 1.0L + g.next_double();
    return x;
}

/// op(M) x in long double, op(M) one of M, M^T, |M|, |M|^T.
template <typename M>
ldvec ld_mv(const M& A, const ldvec& x, bool trans = false, bool absval = false) {
    const std::size_t r = A.num_rows(), c = A.num_cols();
    ldvec y(trans ? c : r, 0.0L);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) {
            long double a = A(i, j);
            if (absval) a = std::fabs(a);
            if (trans) y[j] += a * x[i];
            else       y[i] += a * x[j];
        }
    return y;
}

inline const std::uint64_t freivalds_seeds[2] = {0xF2E1u, 0xF2E2u};

/// Componentwise Freivalds check. got(x) is C x; ref(x) is {P x, |P| x}.
template <typename T, typename Got, typename Ref>
check_result freivalds(std::size_t n, std::size_t k, std::size_t kref, Got got, Ref ref, std::string what) {
    error_tracker e(tol<T>(k) + tol<long double>(kref), std::move(what));
    for (const auto seed : freivalds_seeds) {
        const ldvec x = probe(n, seed);
        const ldvec cx = got(x);
        const auto [px, ax] = ref(x);
        for (std::size_t i = 0; i < cx.size(); ++i) e.add(cx[i], px[i], ax[i]);
    }
    return e.result();
}

enum class probe_norm { inf, two };

/// Normwise Freivalds check: |R x| <= tol * scale * |x| in the chosen norm,
/// where residual(x) is R x.
template <typename T, typename Res>
check_result freivalds_normwise(std::size_t n, std::size_t k, std::size_t kref, long double scale,
                                probe_norm nrm, Res residual, std::string what) {
    error_tracker e(tol<T>(k) + tol<long double>(kref), std::move(what));
    const auto norm = [nrm](const ldvec& v) {
        long double r = 0;
        for (const long double a : v) {
            if (!std::isfinite(a)) return std::numeric_limits<long double>::infinity();
            r = nrm == probe_norm::inf ? std::max(r, std::fabs(a)) : r + a * a;
        }
        return nrm == probe_norm::inf ? r : std::sqrt(r);
    };
    for (const auto seed : freivalds_seeds) {
        const ldvec x = probe(n, seed);
        e.add(norm(residual(x)), 0.0L, scale * norm(x));
    }
    return e.result();
}

} // namespace mtl::bench::platform
