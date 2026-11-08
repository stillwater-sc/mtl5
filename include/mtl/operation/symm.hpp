#pragma once
// MTL5 -- symm: C <- alpha*A*B + beta*C  for symmetric A  (BLAS level-3, left side)
// A is m x m symmetric; B, C are m x n. Left side. Only A's lower triangle is
// read, on every path, as BLAS symm with uplo 'L' reads it (#692): A(i,k) for
// k > i is taken as A(k,i), so the strict upper triangle may hold anything.
// Before #692 the generic path read the full matrix and so disagreed with the
// BLAS path for a non-symmetric A. BLAS dispatch for dense float/double when
// available (#724): any orientation of A (a row-major A's lower triangle is its
// buffer's upper one), and B and C both column-major (side 'L') or both
// row-major (side 'R' on the transposes: C^T = alpha B^T A + beta C^T).
//
// Mixed precision: pass an explicit `Accumulator` to form each (row, col)
// reduction sum_k A(i,k)*B(k,j) in a precision distinct from the operand type
// (#261, Part C); the result is rounded out to C's element type before the
// alpha/beta combine, which happens once outside the reduction -- same shape
// as symv's row loop (#515), extended to a second free index (B's columns)
// instead of a single vector. Default `Accumulator = void` keeps the BLAS /
// generic dispatch byte for byte.
//
// Gated on `interface::accumulator_allows_blas_v`, same as symv/trmm/trsm: a
// custom accumulator cannot be honored by external BLAS.
//
// alpha/beta follow the BLAS convention, as gemmt does (#692): beta == 0 does
// not read C, so a NaN or uninitialized C does not propagate, and alpha == 0
// does not read A or B, leaving C = beta*C.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
#include <mtl/detail/gemm_blocked.hpp>
#endif

namespace mtl {

namespace detail {

/// Whether symm<Accumulator>(alpha, A, B, beta, C) calls ?symm (for sizes that
/// fit an int). B and C must share an orientation; A's may be either.
template <typename Accumulator, typename MA, typename MB, typename MC>
inline constexpr bool symm_dispatches_blas_v = [] {
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::accumulator_allows_blas_v<Accumulator> &&
                  interface::BlasDenseMatrix<MA> && interface::BlasDenseMatrix<MB> && interface::BlasDenseMatrix<MC>)
        return std::is_same_v<typename MC::value_type, typename MA::value_type> &&
               std::is_same_v<typename MC::value_type, typename MB::value_type> &&
               interface::is_row_major_v<MB> == interface::is_row_major_v<MC>;
    else
        return false;
#else
    return false;
#endif
}();

/// Whether symm<void> on this build expands A and calls the native blocked GEMM
/// (#724): MTL5_NATIVE_FAST_GEMM, float or double, dense A and B, row-major C.
template <typename MA, typename MB, typename MC>
inline constexpr bool symm_uses_blocked_v = [] {
#ifdef MTL5_NATIVE_FAST_GEMM
    if constexpr (interface::SimdDenseMatrix<MA> && interface::SimdDenseMatrix<MB> &&
                  interface::SimdDenseMatrix<MC> && requires { typename MB::orientation; })
        return interface::is_blas_scalar_v<typename MC::value_type> &&
               std::is_same_v<typename MA::value_type, typename MC::value_type> &&
               std::is_same_v<typename MB::value_type, typename MC::value_type> &&
               row_major_contiguous_v<MC>;
    else
        return false;
#else
    return false;
#endif
}();

/// Whether the generic symm<void> forms C row by row over a row-major B (#724).
template <typename MB>
inline constexpr bool symm_uses_row_loop_v = row_major_contiguous_v<MB>;

} // namespace detail

/// Symmetric matrix-matrix product: C = alpha*A*B + beta*C, A assumed symmetric.
template <typename Accumulator = void, Scalar S, Matrix MA, Matrix MB, Matrix MC>
void symm(const S& alpha, const MA& A, const MB& B, const S& beta, MC& C) {
    using value_type = typename MC::value_type;
    using size_type  = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type n = B.num_cols();
    assert(A.num_cols() == m && B.num_rows() == m && C.num_rows() == m && C.num_cols() == n);
    const bool alpha_zero = (alpha == math::zero<S>());
    const bool beta_zero  = (beta == math::zero<S>());
    // The lower triangle, read as a symmetric matrix.
    auto a = [&A](size_type i, size_type k) -> decltype(auto) { return k <= i ? A(i, k) : A(k, i); };

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<value_type, typename MA::value_type, typename MB::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j) {
                if (alpha_zero) {
                    C(i, j) = beta_zero ? math::zero<value_type>() : value_type(beta * C(i, j));
                    continue;
                }
                Accumulator acc{};
                AT::clear(acc);
                for (size_type k = 0; k < m; ++k)
                    AT::add_product(acc, static_cast<Value>(a(i, k)), static_cast<Value>(B(k, j)));
                const value_type reduced = AT::template value<value_type>(acc);
                C(i, j) = beta_zero ? value_type(alpha * reduced)
                                    : value_type(alpha * reduced + beta * C(i, j));
            }
        return;
    }

#ifdef MTL5_HAS_BLAS
    if constexpr (detail::symm_dispatches_blas_v<Accumulator, MA, MB, MC>) {
        using T = value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m <= imax && n <= imax) {
            const int mi = static_cast<int>(m), ni = static_cast<int>(n);
            const int lda = std::max(1, mi);
            constexpr char uplo = interface::blas_lower_uplo<MA>();
            if constexpr (interface::is_row_major_v<MC>) {
                const int ld = std::max(1, ni);
                interface::blas::symm('R', uplo, ni, mi, static_cast<T>(alpha), A.data(), lda,
                                      B.data(), ld, static_cast<T>(beta), C.data(), ld);
            } else {
                interface::blas::symm('L', uplo, mi, ni, static_cast<T>(alpha), A.data(), lda,
                                      B.data(), lda, static_cast<T>(beta), C.data(), lda);
            }
            return;
        }
    }
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
    if constexpr (detail::symm_uses_blocked_v<MA, MB, MC>) {
        if (!alpha_zero && m > 0 && n > 0) {
            // Expand A's lower triangle to the full symmetric matrix (O(m^2)),
            // then one blocked GEMM, C := beta C + alpha F B (#724). beta == 0
            // does not read C, as on the other paths. The rounding is the
            // blocked GEMM's, as for mult on this build.
            using T = typename MC::value_type;
            const std::size_t ms = static_cast<std::size_t>(m), ns = static_cast<std::size_t>(n);
            std::vector<T> F(ms * ms);
            for (size_type i = 0; i < m; ++i)
                for (size_type k = 0; k < m; ++k) F[static_cast<std::size_t>(i) * ms + k] = a(i, k);
            const std::ptrdiff_t b_rs = interface::is_row_major_v<MB> ? static_cast<std::ptrdiff_t>(ns) : 1;
            const std::ptrdiff_t b_cs = interface::is_row_major_v<MB> ? 1 : static_cast<std::ptrdiff_t>(ms);
            detail::gemm_blocked<T>(ms, ns, ms, static_cast<T>(alpha), F.data(), static_cast<std::ptrdiff_t>(ms), 1,
                                    B.data(), b_rs, b_cs, static_cast<T>(beta), C.data(), ns,
                                    detail::gemm_default_threads());
            return;
        }
    }
#endif
    if constexpr (detail::symm_uses_row_loop_v<MB>) {
        // Row-major B: the i-k-j form (#724), as mult's generic GEMM (#549).
        // `row` is row i's accumulator and each k adds a(i, k) * B.row(k): a
        // contiguous inner loop, where the i-j-k loop below reads B(k, j) down
        // a column and a(i, k) once per (i, j, k). Every C(i, j) is still
        // zero, + a(i,0)*B(0,j), + a(i,1)*B(1,j), ... in ascending k, then the
        // same alpha/beta combine, so the result is bit-identical
        // (contraction aside: see detail::trsm_rows).
        if (n == 0) return;
        std::vector<value_type> row(static_cast<std::size_t>(n));
        for (size_type i = 0; i < m; ++i) {
            if (alpha_zero) {
                for (size_type j = 0; j < n; ++j)
                    C(i, j) = beta_zero ? math::zero<value_type>() : value_type(beta * C(i, j));
                continue;
            }
            for (size_type j = 0; j < n; ++j) row[j] = math::zero<value_type>();
            for (size_type k = 0; k < m; ++k) {
                const auto& aik = a(i, k);
                const auto* bk = &B(k, 0);
                for (size_type j = 0; j < n; ++j) row[j] += aik * bk[j];
            }
            for (size_type j = 0; j < n; ++j)
                C(i, j) = beta_zero ? value_type(alpha * row[j]) : value_type(alpha * row[j] + beta * C(i, j));
        }
        return;
    }
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j) {
            if (alpha_zero) {
                C(i, j) = beta_zero ? math::zero<value_type>() : value_type(beta * C(i, j));
                continue;
            }
            auto acc = math::zero<value_type>();
            for (size_type k = 0; k < m; ++k)
                acc += a(i, k) * B(k, j);
            C(i, j) = beta_zero ? value_type(alpha * acc) : value_type(alpha * acc + beta * C(i, j));
        }
}

} // namespace mtl
