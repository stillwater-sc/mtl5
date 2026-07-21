#pragma once
// MTL5 -- Spectral / condition / rank property queries (#244, batch 3):
// spectral_radius, condition_number, rcond, numerical_rank, nullity.
//
// These wrap the existing dense SVD and eigenvalue solvers (both take A by
// const reference and copy internally), so the caller's matrix is unchanged.
// Cost is that of the underlying decomposition (SVD / QR-iteration), for dense
// matrices with a real floating value type.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/eigenvalue.hpp>

namespace mtl {

namespace detail {

// Singular values of A (the diagonal of S from the SVD), length min(m,n),
// each non-negative. Order is not assumed; callers reduce over the whole set.
template <Matrix M>
std::vector<magnitude_t<typename M::value_type>> singular_values(const M& A) {
    using mag_t = magnitude_t<typename M::value_type>;
    using std::abs;
    const std::size_t mn = std::min<std::size_t>(A.num_rows(), A.num_cols());
    std::vector<mag_t> sv;
    sv.reserve(mn);
    if (mn == 0) return sv;
    auto [U, S, V] = svd(A);
    (void)U; (void)V;
    for (std::size_t i = 0; i < mn; ++i) sv.push_back(abs(S(i, i)));
    return sv;
}

} // namespace detail

/// Spectral radius: max |eigenvalue|. Requires a square matrix (precondition,
/// as for eigenvalue); the empty (0x0) matrix has spectral radius 0.
template <Matrix M>
magnitude_t<typename M::value_type> spectral_radius(const M& A) {
    using mag_t = magnitude_t<typename M::value_type>;
    using std::abs;
    assert(A.num_rows() == A.num_cols() && "spectral_radius requires a square matrix");
    if (A.num_rows() == 0) return mag_t(0);
    auto eigs = eigenvalue(A);
    mag_t r = mag_t(0);
    for (std::size_t k = 0; k < eigs.size(); ++k) {
        mag_t m = abs(eigs(k));
        if (m > r) r = m;
    }
    return r;
}

/// 2-norm condition number sigma_max / sigma_min. A rank-deficient matrix
/// (sigma_min == 0) has an infinite condition number. The degenerate empty
/// matrix returns 1 (a trivial isometry).
template <Matrix M>
magnitude_t<typename M::value_type> condition_number(const M& A) {
    using mag_t = magnitude_t<typename M::value_type>;
    const auto sv = detail::singular_values(A);
    if (sv.empty()) return mag_t(1);
    mag_t smax = sv[0], smin = sv[0];
    for (mag_t s : sv) { if (s > smax) smax = s; if (s < smin) smin = s; }
    if (!(smin > mag_t(0))) return std::numeric_limits<mag_t>::infinity();
    return smax / smin;
}

/// Reciprocal 2-norm condition number sigma_min / sigma_max, in [0, 1] (0 for a
/// rank-deficient or zero matrix). Numerically safer than condition_number when
/// the matrix may be singular. The empty matrix returns 1.
template <Matrix M>
magnitude_t<typename M::value_type> rcond(const M& A) {
    using mag_t = magnitude_t<typename M::value_type>;
    const auto sv = detail::singular_values(A);
    if (sv.empty()) return mag_t(1);
    mag_t smax = sv[0], smin = sv[0];
    for (mag_t s : sv) { if (s > smax) smax = s; if (s < smin) smin = s; }
    if (!(smax > mag_t(0))) return mag_t(0);   // zero matrix
    return smin / smax;
}

/// Numerical rank: number of singular values above a threshold. The default
/// threshold is max(m,n) * eps * sigma_max (the standard rank-revealing cutoff);
/// pass tol to set an explicit absolute threshold on the singular values.
template <Matrix M>
std::size_t numerical_rank(const M& A,
                           magnitude_t<typename M::value_type> tol = -1) {
    using mag_t = magnitude_t<typename M::value_type>;
    const auto sv = detail::singular_values(A);
    if (sv.empty()) return 0;
    mag_t smax = sv[0];
    for (mag_t s : sv) if (s > smax) smax = s;
    mag_t threshold = tol;
    if (!(tol >= mag_t(0))) {   // negative sentinel -> default cutoff
        const mag_t dim = static_cast<mag_t>(std::max<std::size_t>(A.num_rows(), A.num_cols()));
        threshold = dim * std::numeric_limits<mag_t>::epsilon() * smax;
    }
    std::size_t r = 0;
    for (mag_t s : sv) if (s > threshold) ++r;
    return r;
}

/// Nullity: dimension of the null space = num_cols - numerical_rank.
template <Matrix M>
std::size_t nullity(const M& A, magnitude_t<typename M::value_type> tol = -1) {
    const std::size_t r = numerical_rank(A, tol);
    const std::size_t ncols = static_cast<std::size_t>(A.num_cols());
    return ncols > r ? ncols - r : 0;
}

} // namespace mtl
