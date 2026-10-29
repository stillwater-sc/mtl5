#pragma once
// MTL5 -- gemmt: C <- alpha*A*B + beta*C on one triangle of C (#656), and
// congruence: C <- alpha*A*P*A^T + beta*C built on it.
//
// gemmt is the triangular-output product (MKL/OpenBLAS/BLIS ?gemmt): the
// caller knows A*B is symmetric, so only the `uplo` triangle of C, diagonal
// included, is formed. By default it is then mirrored to the other triangle,
// so the result is exactly symmetric -- syrk's full-result convention.
// gemmt_fill::keep leaves the other triangle as it was.
//
// Generic path: each C(i,j) in the triangle is summed exactly as
// mult_generic sums it -- zero, then + A(i,0)*B(0,j), + A(i,1)*B(1,j), ...
// in ascending k with the same generic_fma -- so the triangle rounds as mult
// does, bit for bit. Row-major B and C take the i-k-j form of that loop, as
// mult does, with the same per-entry order. alpha/beta follow the BLAS
// convention: beta == 0 does
// not read C (NaN in C does not propagate), and alpha == 0 does not read A
// or B.
//
// Mixed precision: an explicit `Accumulator` sums each entry in that
// precision and rounds it out to C's element type once, before the
// alpha/beta combine, as in syrk (#261).
//
// BLAS: ?gemmt is an extension, not reference BLAS before 3.12, so dispatch
// needs MTL5_BLAS_HAS_GEMMT, which CMake defines when the library exports
// dgemmt_. Dense float/double operands of any orientation, and transposed
// views of them, map onto its transa/transb arguments. Gated on
// `interface::accumulator_allows_blas_v`, like the rest of L3.
//
// B may be trans(compressed2D): entry (i,j) is then the sum over the stored
// entries of row j of the CRS matrix, in ascending column order. That is the
// dense order with the unstored (zero) terms left out, so it equals the dense
// result whenever A's entries are finite.
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/detail/wrapping_arithmetic.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/view/transposed_view.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trans.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Which triangle of a square matrix an operation forms or reads.
enum class uplo { lower, upper };

/// What gemmt does with the triangle it does not form.
enum class gemmt_fill {
    mirror,   ///< copy the formed triangle across the diagonal (default)
    keep      ///< leave it as it was
};

namespace detail {

template <typename M>
inline constexpr bool is_transposed_view_v = false;
template <typename M>
inline constexpr bool is_transposed_view_v<mat::view::transposed_view<M>> = true;

template <typename M>
inline constexpr bool is_transposed_crs_v = false;
template <typename V, typename P>
inline constexpr bool is_transposed_crs_v<mat::view::transposed_view<mat::compressed2D<V, P>>> = true;

/// Copy the `t` triangle of the n x n matrix C across the diagonal.
template <typename MC>
void mirror_triangle(MC& C, uplo t) {
    using size_type = typename MC::size_type;
    const size_type n = C.num_rows();
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < i; ++j) {
            if (t == uplo::lower)
                C(j, i) = C(i, j);
            else
                C(i, j) = C(j, i);
        }
}

/// The generic triangle loop. `terms(i, j, f)` calls f(a, b) for each product
/// of entry (i, j) in ascending k; this function owns the summation, the
/// accumulator and the alpha/beta combine.
template <typename Accumulator, typename Value, typename S, typename MC, typename Terms>
void gemmt_triangle(const S& alpha, const S& beta, MC& C, uplo t, Terms terms) {
    using Result = typename MC::value_type;
    using size_type = typename MC::size_type;
    const size_type n = C.num_rows();
    const bool alpha_zero = (alpha == math::zero<S>());
    const bool beta_zero  = (beta == math::zero<S>());
    for (size_type i = 0; i < n; ++i) {
        const size_type jb = (t == uplo::lower) ? size_type{0} : i;
        const size_type je = (t == uplo::lower) ? i + 1 : n;
        for (size_type j = jb; j < je; ++j) {
            if (alpha_zero) {
                C(i, j) = beta_zero ? math::zero<Result>() : Result(beta * C(i, j));
                continue;
            }
            Result s;
            if constexpr (std::is_void_v<Accumulator>) {
                auto acc = math::zero<Result>();
                terms(i, j, [&](const auto& a, const auto& b) { acc = generic_fma<Result>(acc, a, b); });
                s = acc;
            } else {
                using AT = math::accumulator_traits<Accumulator, Value>;
                Accumulator acc{};
                AT::clear(acc);
                terms(i, j, [&](const auto& a, const auto& b) {
                    AT::add_product(acc, static_cast<Value>(a), static_cast<Value>(b));
                });
                s = AT::template value<Result>(acc);
            }
            C(i, j) = beta_zero ? Result(alpha * s) : Result(alpha * s + beta * C(i, j));
        }
    }
}

#if defined(MTL5_HAS_BLAS) && defined(MTL5_BLAS_HAS_GEMMT)
/// How a column-major BLAS routine sees an operand X: op(buffer) == X with
/// op = 'N' or 'T', and the buffer's leading dimension.
struct blas_operand {
    char op;
    int ld;
};

template <typename M>
inline constexpr bool gemmt_blas_dense_v = [] {
    if constexpr (interface::BlasDenseMatrix<M>)
        return requires { typename M::orientation; };
    else
        return false;
}();

template <typename M>
inline constexpr bool gemmt_blas_operand_v = [] {
    if constexpr (is_transposed_view_v<M>)
        return gemmt_blas_dense_v<std::remove_cvref_t<decltype(std::declval<const M&>().base())>>;
    else
        return gemmt_blas_dense_v<M>;
}();

template <typename M>
const auto* gemmt_blas_data(const M& X) {
    if constexpr (is_transposed_view_v<M>)
        return X.base().data();
    else
        return X.data();
}

template <typename M>
blas_operand gemmt_blas_describe(const M& X) {
    auto ld = [](std::size_t d) { return d < 1 ? 1 : static_cast<int>(d); };
    if constexpr (is_transposed_view_v<M>) {
        const auto& Y = X.base();   // X = Y^T
        if constexpr (interface::is_row_major_v<std::remove_cvref_t<decltype(Y)>>)
            return {'N', ld(Y.num_cols())};   // buffer read column-major is Y^T
        else
            return {'T', ld(Y.num_rows())};
    } else if constexpr (interface::is_row_major_v<M>) {
        return {'T', ld(X.num_cols())};       // buffer read column-major is X^T
    } else {
        return {'N', ld(X.num_rows())};
    }
}

inline blas_operand flip(blas_operand d) { return {d.op == 'N' ? 'T' : 'N', d.ld}; }
#endif

/// Whether gemmt<Accumulator>(alpha, A, B, beta, C) calls ?gemmt (for sizes
/// that fit an int).
template <typename Accumulator, typename MA, typename MB, typename MC>
inline constexpr bool gemmt_dispatches_blas_v = [] {
#if defined(MTL5_HAS_BLAS) && defined(MTL5_BLAS_HAS_GEMMT)
    if constexpr (interface::accumulator_allows_blas_v<Accumulator> &&
                  gemmt_blas_dense_v<MC> && gemmt_blas_operand_v<MA> && gemmt_blas_operand_v<MB>)
        return interface::is_blas_scalar_v<typename MC::value_type> &&
               std::is_same_v<typename MA::value_type, typename MC::value_type> &&
               std::is_same_v<typename MB::value_type, typename MC::value_type>;
    else
        return false;
#else
    return false;
#endif
}();

} // namespace detail

/// Triangular-output product: the `t` triangle of C = alpha*A*B + beta*C.
///
/// C is n x n, A is n x k, B is k x n. The other triangle is mirrored from
/// the formed one (`fill = mirror`, the default) or left untouched (`keep`).
/// B may be a dense matrix, trans() of one, or trans() of a compressed2D.
template <typename Accumulator = void, Scalar S, Matrix MA, Matrix MB, Matrix MC>
void gemmt(const S& alpha, const MA& A, const MB& B, const S& beta, MC& C,
           uplo t = uplo::lower, gemmt_fill fill = gemmt_fill::mirror) {
    using value_type = typename MC::value_type;
    using size_type  = typename MC::size_type;
    const size_type n = C.num_rows();
    const std::size_t k = A.num_cols();
    assert(C.num_cols() == n);
    assert(static_cast<std::size_t>(A.num_rows()) == static_cast<std::size_t>(n));
    assert(static_cast<std::size_t>(B.num_cols()) == static_cast<std::size_t>(n));
    assert(static_cast<std::size_t>(B.num_rows()) == k);
    if (n == 0)
        return;

    auto finish = [&] {
        if (fill == gemmt_fill::mirror)
            detail::mirror_triangle(C, t);
    };

#if defined(MTL5_HAS_BLAS) && defined(MTL5_BLAS_HAS_GEMMT)
    if constexpr (detail::gemmt_dispatches_blas_v<Accumulator, MA, MB, MC>) {
        const auto imax = static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (static_cast<std::size_t>(n) <= imax && k <= imax) {
            using T = value_type;
            const int ni = static_cast<int>(n);
            const int ki = static_cast<int>(k);
            const int ldc = ni;
            if constexpr (interface::is_row_major_v<MC>) {
                // The buffer read column-major is C^T = B^T A^T, and C's lower
                // triangle is C^T's upper one.
                const auto da = detail::flip(detail::gemmt_blas_describe(B));
                const auto db = detail::flip(detail::gemmt_blas_describe(A));
                interface::blas::gemmt(t == uplo::lower ? 'U' : 'L', da.op, db.op, ni, ki,
                                       static_cast<T>(alpha), detail::gemmt_blas_data(B), da.ld,
                                       detail::gemmt_blas_data(A), db.ld,
                                       static_cast<T>(beta), C.data(), ldc);
            } else {
                const auto da = detail::gemmt_blas_describe(A);
                const auto db = detail::gemmt_blas_describe(B);
                interface::blas::gemmt(t == uplo::lower ? 'L' : 'U', da.op, db.op, ni, ki,
                                       static_cast<T>(alpha), detail::gemmt_blas_data(A), da.ld,
                                       detail::gemmt_blas_data(B), db.ld,
                                       static_cast<T>(beta), C.data(), ldc);
            }
            finish();
            return;
        }
    }
#endif

    using Value = std::common_type_t<typename MA::value_type, typename MB::value_type>;
    if constexpr (std::is_void_v<Accumulator> &&
                  detail::row_major_contiguous_v<MB> && detail::row_major_contiguous_v<MC>) {
        // Row-major B and C: the i-k-j form, as mult_generic_rows (#549). A row
        // buffer is the accumulator and each k adds A(i,k) * B.row(k) over the
        // triangle's columns -- a contiguous inner loop the compiler
        // vectorizes. Every entry is still zero, + A(i,0)*B(0,j), + ... in
        // ascending k with the same generic_fma, so the bits do not change.
        const bool alpha_zero = (alpha == math::zero<S>());
        const bool beta_zero  = (beta == math::zero<S>());
        std::vector<value_type> acc(static_cast<std::size_t>(n));
        for (size_type i = 0; i < n; ++i) {
            const size_type jb = (t == uplo::lower) ? size_type{0} : i;
            const size_type je = (t == uplo::lower) ? i + 1 : n;
            value_type* crow = &C(i, 0);
            if (alpha_zero) {
                for (size_type j = jb; j < je; ++j)
                    crow[j] = beta_zero ? math::zero<value_type>() : value_type(beta * crow[j]);
                continue;
            }
            value_type* a = acc.data();
            for (size_type j = jb; j < je; ++j)
                a[j] = math::zero<value_type>();
            for (std::size_t l = 0; l < k; ++l) {
                const auto ail = A(static_cast<typename MA::size_type>(i), static_cast<typename MA::size_type>(l));
                const auto* brow = &B(static_cast<typename MB::size_type>(l), 0);
                for (size_type j = jb; j < je; ++j)
                    a[j] = detail::generic_fma<value_type>(a[j], ail, brow[j]);
            }
            for (size_type j = jb; j < je; ++j)
                crow[j] = beta_zero ? value_type(alpha * a[j]) : value_type(alpha * a[j] + beta * crow[j]);
        }
    } else if constexpr (detail::is_transposed_crs_v<MB>) {
        // B = R^T for a CRS R: B(k, j) = R(j, k), the stored entries of row j.
        const auto& R = B.base();
        const auto& starts  = R.ref_major();
        const auto& indices = R.ref_minor();
        const auto& data    = R.ref_data();
        detail::gemmt_triangle<Accumulator, Value>(alpha, beta, C, t,
            [&](size_type i, size_type j, auto&& f) {
                for (auto p = starts[j]; p < starts[j + 1]; ++p)
                    f(A(i, static_cast<typename MA::size_type>(indices[p])), data[p]);
            });
    } else {
        detail::gemmt_triangle<Accumulator, Value>(alpha, beta, C, t,
            [&](size_type i, size_type j, auto&& f) {
                for (std::size_t l = 0; l < k; ++l)
                    f(A(i, static_cast<typename MA::size_type>(l)),
                      B(static_cast<typename MB::size_type>(l), j));
            });
    }
    finish();
}

namespace detail {

/// The intermediate T of a congruence: dense, C's element type and orientation.
template <typename MC>
using congruence_temp_t = mat::dense2D<typename MC::value_type,
                                       mat::parameters<typename MC::orientation>>;

/// The second product of a congruence, C = alpha * T*A^T + beta*C, lower
/// triangle mirrored. A CRS A goes to gemmt as trans(A), which walks its rows.
/// So does a dense A that ?gemmt takes, through its transb argument. Any
/// other dense A is copied out as A^T first: an O(mn) copy that lets a
/// row-major C take gemmt's i-k-j form, where trans(A) would walk a column of
/// A per entry. The entries summed, and their order, are the same either way.
template <typename Accumulator = void, typename S, typename MT, typename MA, typename MC>
void congruence_second(const S& alpha, const MT& T, const MA& A, const S& beta, MC& C) {
    if constexpr (interface::is_compressed2D_v<MA> ||
                  gemmt_dispatches_blas_v<Accumulator, MT, mat::view::transposed_view<MA>, MC>) {
        gemmt<Accumulator>(alpha, T, trans(A), beta, C);
    } else {
        using At_type = mat::dense2D<typename MA::value_type,
                                     mat::parameters<typename MC::orientation>>;
        using size_type = typename At_type::size_type;
        At_type At(A.num_cols(), A.num_rows());
        for (size_type i = 0; i < At.num_rows(); ++i)
            for (size_type j = 0; j < At.num_cols(); ++j)
                At(i, j) = A(j, i);
        gemmt<Accumulator>(alpha, T, At, beta, C);
    }
}

} // namespace detail

/// Congruence: C = alpha * A*P*A^T + beta*C, for a symmetric m x m result.
///
/// A is m x n, dense or compressed2D; P is n x n. Computed as T = A*P with
/// mult, then gemmt(T, A^T), so C is exactly symmetric and its lower
/// triangle rounds as mult(mult(A, P), trans(A)) does on the generic path.
/// P's symmetry is the caller's promise; it is not checked or exploited.
template <typename Accumulator = void, Scalar S, Matrix MA, Matrix MP, Matrix MC>
void congruence(const S& alpha, const MA& A, const MP& P, const S& beta, MC& C) {
    assert(P.num_rows() == A.num_cols() && P.num_cols() == A.num_cols());
    assert(C.num_rows() == A.num_rows() && C.num_cols() == A.num_rows());
    detail::congruence_temp_t<MC> T(A.num_rows(), A.num_cols());
    mult<Accumulator>(A, P, T);
    detail::congruence_second<Accumulator>(alpha, T, A, beta, C);
}

/// Congruence with a diagonal middle term: C = alpha * A*diag(r)*A^T + beta*C.
///
/// The Joseph-form term K*R*K^T for a diagonal R. T(i,k) = A(i,k)*r(k) is
/// formed directly (one rounding per entry, as mult by diag(r) gives), then
/// gemmt(T, A^T) as in the dense-middle overload.
template <typename Accumulator = void, Scalar S, Matrix MA, Vector VR, Matrix MC>
    requires (!Matrix<VR>)
void congruence(const S& alpha, const MA& A, const VR& r, const S& beta, MC& C) {
    using value_type = typename MC::value_type;
    using size_type  = typename detail::congruence_temp_t<MC>::size_type;
    assert(static_cast<std::size_t>(r.size()) == static_cast<std::size_t>(A.num_cols()));
    assert(C.num_rows() == A.num_rows() && C.num_cols() == A.num_rows());
    detail::congruence_temp_t<MC> T(A.num_rows(), A.num_cols());
    if constexpr (interface::is_compressed2D_v<MA>) {
        for (size_type i = 0; i < T.num_rows(); ++i)
            for (size_type j = 0; j < T.num_cols(); ++j)
                T(i, j) = math::zero<value_type>();
        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data    = A.ref_data();
        for (std::size_t i = 0; i < static_cast<std::size_t>(A.num_rows()); ++i)
            for (auto p = starts[i]; p < starts[i + 1]; ++p)
                T(static_cast<size_type>(i), static_cast<size_type>(indices[p])) =
                    value_type(data[p] * r(indices[p]));
    } else {
        for (size_type i = 0; i < T.num_rows(); ++i)
            for (size_type j = 0; j < T.num_cols(); ++j)
                T(i, j) = value_type(A(i, j) * r(j));
    }
    detail::congruence_second<Accumulator>(alpha, T, A, beta, C);
}

/// C = A*P*A^T.
template <typename Accumulator = void, Matrix MA, Matrix MP, Matrix MC>
void congruence(const MA& A, const MP& P, MC& C) {
    using value_type = typename MC::value_type;
    congruence<Accumulator>(math::one<value_type>(), A, P, math::zero<value_type>(), C);
}

/// C = A*diag(r)*A^T.
template <typename Accumulator = void, Matrix MA, Vector VR, Matrix MC>
    requires (!Matrix<VR>)
void congruence(const MA& A, const VR& r, MC& C) {
    using value_type = typename MC::value_type;
    congruence<Accumulator>(math::one<value_type>(), A, r, math::zero<value_type>(), C);
}

} // namespace mtl
