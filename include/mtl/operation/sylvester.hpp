#pragma once
// MTL5 -- Sylvester equations A X + X B = C and A X - X B = C (#650)
//
// sylvester_triangular() solves the equation when A and B are already upper
// quasi-triangular (a real Schur form: 1x1 and 2x2 diagonal blocks) or upper
// triangular (a complex Schur form). It is a block back substitution, the
// algorithm of Bartels & Stewart (1972) as LAPACK's trsyl implements it:
// X is computed one diagonal block at a time, and each block solves a system
// of at most four unknowns.
//
// sylvester() solves it for general real A and B, by reducing both to real
// Schur form first (Bartels-Stewart), as MATLAB's sylvester does.
//
// Both are generic in the element type: they need + - * /, comparisons, and
// abs found by ADL, so they run for posits, cfloats, LNS, double-double and
// the other user-defined types, real or complex. dense2D<float/double> and
// dense2D<std::complex<float/double>> dispatch to LAPACK trsyl when
// MTL5_HAS_LAPACK is defined.
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/schur.hpp>
#include <mtl/detail/unit_roundoff.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// The sign between the two terms of a Sylvester equation:
/// plus for A X + X B = C, minus for A X - X B = C.
enum class sylvester_sign { plus, minus };

/// The solution of a Sylvester equation.
template <typename Value>
struct sylvester_result {
    mat::dense2D<Value> X;  ///< the solution
    /// True when the equation is singular or nearly so: an eigenvalue of A is
    /// within roundoff of an eigenvalue of -B (of B, for sylvester_sign::minus).
    /// Pivots below that threshold were replaced by it, so X solves a nearby
    /// equation and may be large and inaccurate, as with LAPACK trsyl's info = 1.
    bool perturbed = false;
};

/// The solution of a Sylvester equation scaled against overflow, as LAPACK
/// trsyl returns it: X solves A X + sgn X B = scale * C with 0 < scale <= 1.
/// scale < 1 only when the solution itself, or an intermediate of its
/// computation, would exceed the range of the element type.
template <typename Value>
struct sylvester_scaled_result {
    mat::dense2D<Value> X;          ///< the solution of the scaled equation
    magnitude_t<Value> scale{1};    ///< the factor applied to C, in (0, 1]
    bool perturbed = false;         ///< as in sylvester_result
};

namespace detail {

/// The diagonal blocks of an upper quasi-triangular square matrix, as their
/// starting indices followed by n. A 2x2 block starts at k when T(k+1, k) != 0.
/// Throws std::invalid_argument when T is not square or not quasi-triangular:
/// a nonzero below the subdiagonal, or two adjacent nonzero subdiagonals.
template <typename M>
std::vector<std::size_t> quasi_triangular_blocks(const M& T, const char* who, const char* name) {
    using value_type = typename M::value_type;
    const std::size_t n = T.num_rows();
    if (T.num_cols() != n)
        throw std::invalid_argument(std::string(who) + ": " + name + " must be square");
    const value_type zero(0);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j + 2; i < n; ++i)
            if (!(T(i, j) == zero))
                throw std::invalid_argument(std::string(who) + ": " + name +
                                            " must be upper quasi-triangular");
    std::vector<std::size_t> starts;
    for (std::size_t k = 0; k < n; ) {
        starts.push_back(k);
        if (k + 1 < n && !(T(k + 1, k) == zero)) {
            if (k + 2 < n && !(T(k + 2, k + 1) == zero))
                throw std::invalid_argument(std::string(who) + ": " + name +
                                            " has two adjacent nonzero subdiagonal entries");
            k += 2;
        } else {
            k += 1;
        }
    }
    starts.push_back(n);
    return starts;
}

/// The range limits of a Sylvester solve, from numeric_limits:
///
/// - smlnum = safmin * (m n / eps), LAPACK trsyl's floor for the singularity
///   threshold. As in schur(), it is used only when it is below eps: a
///   fixed-point type's smallest value is no safe minimum, and half
///   precision's gives a floor near 0.06 m n.
/// - bignum = max * eps / (m n), the size above which the scaled solve scales.
///   For double and for posits it is LAPACK's eps / (safmin m n); unlike that
///   form it also suits a type whose safe minimum is far from 1 / max, such as
///   half. It is used only when it is above 1.
///
/// A type without numeric_limits has neither.
template <typename R>
struct sylvester_limits {
    R smlnum{};
    R bignum{};
    bool have_floor = false;
    bool have_bignum = false;
};

template <typename R>
sylvester_limits<R> make_sylvester_limits(std::size_t m, std::size_t n, const R& eps) {
    sylvester_limits<R> l;
    if constexpr (std::numeric_limits<R>::is_specialized) {
        const R mn(double(m) * double(n));
        const R smlnum = std::numeric_limits<R>::min() * (mn / eps);
        if (smlnum > R(0) && smlnum < eps) {
            l.smlnum = smlnum;
            l.have_floor = true;
        }
        const R bignum = std::numeric_limits<R>::max() * eps / mn;
        if (bignum > R(1)) {
            l.bignum = bignum;
            l.have_bignum = true;
        }
    }
    return l;
}

/// The largest power of two f <= 1 with x * f <= 1, for x > 1.
template <typename R>
R scale_down_to_one(R x) {
    R f(1);
    const R half(0.5);
    for (int i = 0; i < 100000 && x * f > R(1); ++i) f = f * half;
    return f;
}

/// Solve the p*q unknowns of a p x p block a and a q x q block b (p, q <= 2),
///
///   a y + sgn y b = scaloc * rhs,
///
/// written as the Kronecker system (I_q (x) a + sgn b^T (x) I_p) vec(y) =
/// vec(rhs), by Gaussian elimination with complete pivoting, as LAPACK's dlasy2
/// does. A pivot no larger in magnitude than smin is replaced by smin; the
/// return value says whether that happened. On return rhs holds y.
///
/// scaloc is 1 unless `scaled` and a division in the back substitution would
/// exceed lim.bignum. Then the right-hand side, and the part of y already computed,
/// are scaled by a power of two that brings the dividend to at most 1, as
/// LAPACK's 1x1 case does with 1 / |dividend|; scaloc is the product of those
/// factors.
template <typename T, typename R>
bool solve_sylvester_block(const T (&a)[2][2], int p, const T (&b)[2][2], int q, const T& sgn,
                           T (&rhs)[2][2], const R& smin, const sylvester_limits<R>& lim,
                           bool scaled, R& scaloc) {
    using std::abs;
    const int n = p * q;
    T M[4][4];
    T v[4];
    const T zero(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) M[i][j] = zero;
        v[i] = rhs[i % p][i / p];
    }
    // Row (r, c) is unknown r + p*c: sum_t a(r,t) y(t,c) + sgn sum_t y(r,t) b(t,c).
    for (int c = 0; c < q; ++c)
        for (int r = 0; r < p; ++r) {
            const int u = r + p * c;
            for (int t = 0; t < p; ++t) M[u][t + p * c] += a[r][t];
            for (int t = 0; t < q; ++t) M[u][r + p * t] += sgn * b[t][c];
        }

    bool perturbed = false;
    int col[4] = {0, 1, 2, 3};   // col[k]: the unknown held in column k
    for (int k = 0; k < n; ++k) {
        int pr = k, pc = k;
        R best = abs(M[k][k]);
        for (int i = k; i < n; ++i)
            for (int j = k; j < n; ++j)
                if (best < abs(M[i][j])) { best = abs(M[i][j]); pr = i; pc = j; }
        if (pr != k) {
            for (int j = 0; j < n; ++j) { const T t = M[k][j]; M[k][j] = M[pr][j]; M[pr][j] = t; }
            const T t = v[k]; v[k] = v[pr]; v[pr] = t;
        }
        if (pc != k) {
            for (int i = 0; i < n; ++i) { const T t = M[i][k]; M[i][k] = M[i][pc]; M[i][pc] = t; }
            const int t = col[k]; col[k] = col[pc]; col[pc] = t;
        }
        if (abs(M[k][k]) <= smin) { M[k][k] = T(smin); perturbed = true; }   // as dlasy2
        for (int i = k + 1; i < n; ++i) {
            const T f = M[i][k] / M[k][k];
            for (int j = k + 1; j < n; ++j) M[i][j] -= f * M[k][j];
            v[i] -= f * v[k];
        }
    }
    scaloc = R(1);
    T y[4];
    for (int k = n; k-- > 0; ) {
        T s = v[k];
        for (int j = k + 1; j < n; ++j) s -= M[k][j] * y[j];
        // |s / M_kk| > bignum: scale everything computed so far and the
        // remaining right-hand side, so that |s| <= 1 and |y_k| <= 1 / smin.
        if (scaled && lim.have_bignum) {
            const R as = abs(s), am = abs(M[k][k]);
            if (am < R(1) && as > R(1) && as > lim.bignum * am) {
                const R f = scale_down_to_one(as);
                const T ft(f);
                s = s * ft;
                for (int j = k + 1; j < n; ++j) y[j] = y[j] * ft;
                for (int i = 0; i < k; ++i) v[i] = v[i] * ft;
                scaloc = scaloc * f;
            }
        }
        y[k] = s / M[k][k];
    }
    for (int k = 0; k < n; ++k) rhs[col[k] % p][col[k] / p] = y[k];
    return perturbed;
}

/// The in-house solve behind sylvester_triangular (scaled == false: no
/// scaling, scale stays 1) and sylvester_triangular_scaled (scaled == true).
/// A, B and C are validated by the caller.
template <typename Accumulator, typename MA, typename MB, typename MC>
sylvester_scaled_result<typename MC::value_type>
sylvester_triangular_inhouse(const MA& A, const MB& B, const MC& C, sylvester_sign sign,
                             const std::vector<std::size_t>& ablocks,
                             const std::vector<std::size_t>& bblocks, bool scaled) {
    using value_type = typename MC::value_type;
    using real_type  = magnitude_t<value_type>;
    using size_type  = std::size_t;
    using std::abs;
    const size_type m = A.num_rows(), n = B.num_rows();

    // LAPACK trsyl's threshold: smin = max(eps * max(|a_ij|, |b_ij|), smlnum).
    const real_type eps = machine_epsilon<real_type>();
    real_type norm(0);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
            if (norm < abs(A(i, j))) norm = abs(A(i, j));
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            if (norm < abs(B(i, j))) norm = abs(B(i, j));
    const auto lim = make_sylvester_limits<real_type>(m, n, eps);
    real_type smin = eps * norm;
    if (lim.have_floor && smin < lim.smlnum) smin = lim.smlnum;
    if (!(smin > real_type(0))) smin = eps;   // A and B are zero

    using Acc = std::conditional_t<std::is_void_v<Accumulator>, value_type, Accumulator>;
    using AT  = math::accumulator_traits<Acc, value_type>;
    const bool plus = sign == sylvester_sign::plus;
    const value_type sgn = plus ? value_type(1) : value_type(-1);

    sylvester_scaled_result<value_type> r{mat::dense2D<value_type>(m, n), real_type(1), false};
    auto& X = r.X;
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j) X(i, j) = value_type(0);
    for (size_type lb = 0; lb + 1 < bblocks.size(); ++lb) {
        const size_type l = bblocks[lb];
        const int q = static_cast<int>(bblocks[lb + 1] - l);
        for (size_type kb = ablocks.size() - 1; kb-- > 0; ) {
            const size_type k = ablocks[kb];
            const int p = static_cast<int>(ablocks[kb + 1] - k);
            value_type a[2][2], b[2][2], rhs[2][2];
            for (int i = 0; i < p; ++i)
                for (int j = 0; j < p; ++j) a[i][j] = A(k + i, k + j);
            for (int i = 0; i < q; ++i)
                for (int j = 0; j < q; ++j) b[i][j] = B(l + i, l + j);
            // rhs = scale C_kl - sum_{j past block k} A(k, j) X(j, l)
            //                  - sgn sum_{j before block l} X(k, j) B(j, l)
            const bool c_scaled = !(r.scale == real_type(1));
            const value_type sc(r.scale);
            for (int i = 0; i < p; ++i)
                for (int c = 0; c < q; ++c) {
                    const size_type row = k + static_cast<size_type>(i);
                    const size_type cl  = l + static_cast<size_type>(c);
                    Acc acc;
                    AT::assign(acc, c_scaled ? value_type(C(row, cl) * sc) : value_type(C(row, cl)));
                    for (size_type j = k + static_cast<size_type>(p); j < m; ++j)
                        AT::add_product(acc, -A(row, j), X(j, cl));
                    for (size_type j = 0; j < l; ++j)
                        AT::add_product(acc, X(row, j), plus ? -B(j, cl) : B(j, cl));
                    rhs[i][c] = AT::template value<value_type>(acc);
                }
            real_type scaloc(1);
            if (solve_sylvester_block(a, p, b, q, sgn, rhs, smin, lim, scaled, scaloc))
                r.perturbed = true;
            if (!(scaloc == real_type(1))) {
                // Everything solved so far belongs to the old scale.
                const value_type f(scaloc);
                for (size_type i = 0; i < m; ++i)
                    for (size_type j = 0; j < n; ++j) X(i, j) = X(i, j) * f;
                r.scale = r.scale * scaloc;
            }
            for (int i = 0; i < p; ++i)
                for (int c = 0; c < q; ++c)
                    X(k + static_cast<size_type>(i), l + static_cast<size_type>(c)) = rhs[i][c];
        }
    }
    return r;
}

#ifdef MTL5_HAS_LAPACK
/// LAPACK trsyl on float/double or complex<float/double> dense matrices, for
/// op(A) = A and op(B) = B.
template <typename MA, typename MB, typename MC>
sylvester_scaled_result<typename MC::value_type> lapack_trsyl(const MA& A, const MB& B, const MC& C,
                                                              sylvester_sign sign) {
    using value_type = typename MC::value_type;
    using real_type  = magnitude_t<value_type>;
    const std::size_t m = C.num_rows(), n = C.num_cols();
    const int mi = static_cast<int>(m), ni = static_cast<int>(n);
    std::vector<value_type> a(m * m), b(n * n), c(m * n);
    for (std::size_t j = 0; j < m; ++j)
        for (std::size_t i = 0; i < m; ++i) a[j * m + i] = A(i, j);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i) b[j * n + i] = B(i, j);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < m; ++i) c[j * m + i] = C(i, j);
    real_type scale(1);
    const int isgn = sign == sylvester_sign::plus ? 1 : -1;
    const int info = interface::lapack::trsyl('N', 'N', isgn, mi, ni, a.data(), mi > 0 ? mi : 1,
                                              b.data(), ni > 0 ? ni : 1, c.data(),
                                              mi > 0 ? mi : 1, &scale);
    if (info < 0)
        throw std::logic_error("mtl::sylvester_triangular: LAPACK trsyl rejected argument " +
                               std::to_string(-info));
    sylvester_scaled_result<value_type> r{mat::dense2D<value_type>(m, n), scale, info == 1};
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < m; ++i) r.X(i, j) = c[j * m + i];
    return r;
}
#endif // MTL5_HAS_LAPACK

} // namespace detail

namespace detail {

/// Validate, dispatch and solve, for sylvester_triangular (scaled == false) and
/// sylvester_triangular_scaled (scaled == true).
template <typename Accumulator, typename MA, typename MB, typename MC>
sylvester_scaled_result<typename MC::value_type>
sylvester_triangular_solve(const MA& A, const MB& B, const MC& C, sylvester_sign sign, bool scaled) {
    using value_type = typename MC::value_type;
    using real_type  = magnitude_t<value_type>;
    static_assert(std::is_same_v<typename MA::value_type, value_type> &&
                  std::is_same_v<typename MB::value_type, value_type>,
                  "mtl::sylvester_triangular: A, B and C must have the same element type");
    constexpr const char* who = "mtl::sylvester_triangular";
    const auto ablocks = quasi_triangular_blocks(A, who, "A");
    const auto bblocks = quasi_triangular_blocks(B, who, "B");
    const std::size_t m = A.num_rows(), n = B.num_rows();
    if (C.num_rows() != m || C.num_cols() != n)
        throw std::invalid_argument("mtl::sylvester_triangular: C must be A.num_rows() x B.num_rows()");
    auto finite = [](const value_type& x) { return x - x == value_type(0); };
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j)
            if (!finite(A(i, j))) throw std::invalid_argument("mtl::sylvester_triangular: A has a NaN or infinite entry");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!finite(B(i, j))) throw std::invalid_argument("mtl::sylvester_triangular: B has a NaN or infinite entry");
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!finite(C(i, j))) throw std::invalid_argument("mtl::sylvester_triangular: C has a NaN or infinite entry");

    if (m == 0 || n == 0)
        return sylvester_scaled_result<value_type>{mat::dense2D<value_type>(m, n), real_type(1), false};

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::accumulator_allows_blas_v<Accumulator>) {
        if constexpr (interface::BlasDenseMatrix<MA> && interface::BlasDenseMatrix<MB> &&
                      interface::BlasDenseMatrix<MC>) {
            return lapack_trsyl(A, B, C, sign);
        } else if constexpr (interface::BlasHermitianMatrix<MA> && interface::BlasHermitianMatrix<MB> &&
                             interface::BlasHermitianMatrix<MC>) {
            // ctrsyl/ztrsyl read A and B as upper triangular: a 2x2 block would
            // be ignored, so such input takes the in-house path.
            if (ablocks.size() == m + 1 && bblocks.size() == n + 1)
                return lapack_trsyl(A, B, C, sign);
        }
    }
#endif
    return sylvester_triangular_inhouse<Accumulator>(A, B, C, sign, ablocks, bblocks, scaled);
}

} // namespace detail

/// Solve A X + X B = scale * C (sylvester_sign::plus, the default) or
/// A X - X B = scale * C (sylvester_sign::minus) for X and a scale factor
/// 0 < scale <= 1, as LAPACK trsyl does. A and B are as for
/// sylvester_triangular(). Use as
/// `auto [X, scale, perturbed] = mtl::sylvester_triangular_scaled(TA, TB, C);`.
///
/// scale is 1 unless the solution would exceed the range of the element type.
/// Then the solve scales the right-hand side down by powers of two as it goes,
/// so that X stays representable, and reports the product of those factors.
/// This is for callers that need only the direction or the relative size of X,
/// such as condition estimators, which solve many nearly singular equations;
/// sylvester_triangular() is the call for X itself.
///
/// The solve scales once a block's solution would exceed max * eps / (m n),
/// from numeric_limits; for double and for posits that is LAPACK's
/// eps / (safmin m n). The margin of eps / (m n) below the largest value keeps
/// the later sums of products of A, B and X in range. A type without
/// numeric_limits has no threshold and always gets scale = 1. Scaling is by
/// powers of two, so entries of X far smaller than its largest can become
/// subnormal or zero; that is the price of a representable X.
///
/// Throws std::invalid_argument as sylvester_triangular() does. Dispatches to
/// LAPACK trsyl in the same cases, passing its scale through.
template <typename Accumulator = void, FieldMatrix MA, FieldMatrix MB, FieldMatrix MC>
auto sylvester_triangular_scaled(const MA& A, const MB& B, const MC& C,
                                 sylvester_sign sign = sylvester_sign::plus) {
    return detail::sylvester_triangular_solve<Accumulator>(A, B, C, sign, true);
}

/// Solve A X + X B = C (sylvester_sign::plus, the default) or A X - X B = C
/// (sylvester_sign::minus) for X, where A (m x m) and B (n x n) are upper
/// quasi-triangular, as schur() returns them, or upper triangular, as
/// rsf2csf() returns them. C and X are m x n. Use as
/// `auto [X, perturbed] = mtl::sylvester_triangular(TA, TB, C);`.
///
/// The equation has a unique solution exactly when A and -B (B, for minus)
/// have no eigenvalue in common. When they come within roundoff of one,
/// the result's `perturbed` flag is set; see sylvester_result.
///
/// Throws std::invalid_argument when A or B is not square or not upper
/// quasi-triangular, when C is not m x n, or when an entry of A, B or C is NaN
/// or infinite. Throws std::overflow_error when X is too large to represent in
/// the element type: an entry that is infinite, NaN, or at the type's largest
/// value, where a posit saturates. Without that check an IEEE type would return
/// infinities and a posit maxpos, silently. sylvester_triangular_scaled()
/// returns such an X scaled down instead. The in-house path does not scale, so
/// it never loses the small entries of an X with a wide range.
///
/// In-house path, for every element type, real or complex:
///   - X is computed block column by block column of B from the left, and in
///     each, block row by block row of A from the bottom. Each block solves
///     A_kk Y + sgn Y B_ll = C_kl - sum_j A_kj X_jl - sgn sum_j X_kj B_jl, of at
///     most 2x2 unknowns, by complete pivoting (LAPACK dlasy2).
///   - The sums are accumulated in `Accumulator` (leading, default void: plain
///     arithmetic in the element type), so a quire makes each one exact.
///   - The singularity threshold is LAPACK trsyl's, eps * max(max|a_ij|,
///     max|b_ij|), with its safe-minimum floor when numeric_limits gives one
///     that is below eps.
///
/// Dispatches to LAPACK trsyl when MTL5_HAS_LAPACK is defined, the accumulator
/// is void and A, B and C are dense float/double, or dense complex<float/double>
/// with A and B upper triangular.
template <typename Accumulator = void, FieldMatrix MA, FieldMatrix MB, FieldMatrix MC>
auto sylvester_triangular(const MA& A, const MB& B, const MC& C,
                          sylvester_sign sign = sylvester_sign::plus) {
    using value_type = typename MC::value_type;
    using real_type  = magnitude_t<value_type>;
    using std::abs;
    constexpr const char* overflow_message =
        "mtl::sylvester_triangular: the solution is too large to represent in the element "
        "type (sylvester_triangular_scaled returns it scaled)";
    auto s = detail::sylvester_triangular_solve<Accumulator>(A, B, C, sign, false);
    const std::size_t m = s.X.num_rows(), n = s.X.num_cols();
    if (!(s.scale == real_type(1))) {
        // Only LAPACK trsyl scales here. X = X_scaled / scale is returned when
        // it is representable.
        bool fits = false;
        if constexpr (std::numeric_limits<real_type>::is_specialized) {
            real_type amax(0);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    if (amax < abs(s.X(i, j))) amax = abs(s.X(i, j));
            fits = amax <= std::numeric_limits<real_type>::max() * s.scale;
        }
        if (!fits) throw std::overflow_error(overflow_message);
        const value_type sc(s.scale);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) s.X(i, j) = s.X(i, j) / sc;
    }
    // The in-house solve does not scale, so overflow shows in X itself: an
    // infinity or NaN in an IEEE type, or an entry at the largest value in a
    // type that saturates (a posit's maxpos). The inputs are finite.
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const value_type& x = s.X(i, j);
            bool bad = !(x - x == value_type(0));
            if constexpr (std::numeric_limits<real_type>::is_specialized)
                bad = bad || !(abs(x) < std::numeric_limits<real_type>::max());
            if (bad) throw std::overflow_error(overflow_message);
        }
    return sylvester_result<value_type>{std::move(s.X), s.perturbed};
}

/// Solve A X + X B = C (sylvester_sign::plus, the default) or A X - X B = C
/// (sylvester_sign::minus) for general real square A (m x m) and B (n x n),
/// and C (m x n), by the Bartels-Stewart method: A = Q_A T_A Q_A^T and
/// B = Q_B T_B Q_B^T by schur(), then T_A Y + Y T_B = Q_A^T C Q_B by
/// sylvester_triangular(), and X = Q_A Y Q_B^T. Use as
/// `auto [X, perturbed] = mtl::sylvester(A, B, C);`.
///
/// Throws as schur() and sylvester_triangular() do. `Accumulator` (leading,
/// default void) is forwarded to both and to the four matrix products.
/// Real element types only, until the complex Schur form of a complex matrix
/// is available (#650); for complex triangular A and B, call
/// sylvester_triangular() directly.
template <typename Accumulator = void, FieldMatrix MA, FieldMatrix MB, FieldMatrix MC>
auto sylvester(const MA& A, const MB& B, const MC& C, sylvester_sign sign = sylvester_sign::plus) {
    using value_type = typename MC::value_type;
    using size_type  = std::size_t;
    static_assert(!is_complex_v<value_type>,
                  "mtl::sylvester: complex A and B need the complex Schur form, which is "
                  "not available yet (#650); use sylvester_triangular for triangular input");
    static_assert(std::is_same_v<typename MA::value_type, value_type> &&
                  std::is_same_v<typename MB::value_type, value_type>,
                  "mtl::sylvester: A, B and C must have the same element type");
    const size_type m = A.num_rows(), n = B.num_rows();
    if (A.num_cols() != m || B.num_cols() != n)
        throw std::invalid_argument("mtl::sylvester: A and B must be square");
    if (C.num_rows() != m || C.num_cols() != n)
        throw std::invalid_argument("mtl::sylvester: C must be A.num_rows() x B.num_rows()");
    if (m == 0 || n == 0)
        return sylvester_result<value_type>{mat::dense2D<value_type>(m, n), false};

    auto [QA, TA] = schur<Accumulator>(A);
    auto [QB, TB] = schur<Accumulator>(B);

    auto transpose = [](const mat::dense2D<value_type>& Q) {
        mat::dense2D<value_type> Qt(Q.num_cols(), Q.num_rows());
        for (size_type i = 0; i < Q.num_rows(); ++i)
            for (size_type j = 0; j < Q.num_cols(); ++j) Qt(j, i) = Q(i, j);
        return Qt;
    };
    mat::dense2D<value_type> Cm(m, n);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j) Cm(i, j) = C(i, j);

    // F = Q_A^T C Q_B
    mat::dense2D<value_type> W(m, n), F(m, n);
    mult<Accumulator>(transpose(QA), Cm, W);
    mult<Accumulator>(W, QB, F);

    auto r = sylvester_triangular<Accumulator>(TA, TB, F, sign);

    // X = Q_A Y Q_B^T
    mult<Accumulator>(QA, r.X, W);
    mult<Accumulator>(W, transpose(QB), r.X);
    return r;
}

} // namespace mtl
