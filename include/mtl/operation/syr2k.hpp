#pragma once
// MTL5 -- syr2k: C <- alpha*(A*B^T + B*A^T) + beta*C  (BLAS level-3 symmetric
// rank-2k). C is m x m and treated as symmetric; the full symmetric result is
// produced (both triangles set). A, B are m x k. BLAS dispatch for dense
// float/double when available, for A and B of one orientation and C of any
// (#724); BLAS writes the lower triangle, which is then mirrored.
//
// Mixed precision: pass an explicit `Accumulator` to form each lower-triangle
// reduction in a precision distinct from the operand type (#261, Part C).
// Same lower-triangle/mirror shape as syrk, but sums two product terms
// (A(i,l)*B(j,l) and B(i,l)*A(j,l)) per l into the same accumulator before
// rounding out once, before the alpha/beta combine. The mirror to the upper
// triangle afterward is a plain unrounded copy, not a re-sum. Default
// `Accumulator = void` keeps the BLAS / generic dispatch byte for byte.
//
// Gated on `interface::accumulator_allows_blas_v`, same as syrk/symm: a
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

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

namespace detail {

/// Whether syr2k<Accumulator>(alpha, A, B, beta, C) calls ?syr2k (for sizes that
/// fit an int). A and B must share an orientation; C's may be either (#724).
template <typename Accumulator, typename MA, typename MB, typename MC>
inline constexpr bool syr2k_dispatches_blas_v = [] {
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::accumulator_allows_blas_v<Accumulator> &&
                  interface::BlasDenseMatrix<MA> && interface::BlasDenseMatrix<MB> && interface::BlasDenseMatrix<MC>)
        return std::is_same_v<typename MC::value_type, typename MA::value_type> &&
               std::is_same_v<typename MC::value_type, typename MB::value_type> &&
               interface::is_row_major_v<MA> == interface::is_row_major_v<MB>;
    else
        return false;
#else
    return false;
#endif
}();

} // namespace detail

/// Symmetric rank-2k update: C = alpha*(A*B^T + B*A^T) + beta*C, C symmetric.
template <typename Accumulator = void, Scalar S, Matrix MA, Matrix MB, Matrix MC>
void syr2k(const S& alpha, const MA& A, const MB& B, const S& beta, MC& C) {
    using value_type = typename MC::value_type;
    using size_type  = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type k = A.num_cols();
    assert(B.num_rows() == m && B.num_cols() == k && C.num_rows() == m && C.num_cols() == m);
    const bool alpha_zero = (alpha == math::zero<S>());
    const bool beta_zero  = (beta == math::zero<S>());

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<value_type, typename MA::value_type, typename MB::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j <= i; ++j) {
                if (alpha_zero) {
                    C(i, j) = beta_zero ? math::zero<value_type>() : value_type(beta * C(i, j));
                    continue;
                }
                Accumulator acc{};
                AT::clear(acc);
                for (size_type l = 0; l < k; ++l) {
                    AT::add_product(acc, static_cast<Value>(A(i, l)), static_cast<Value>(B(j, l)));
                    AT::add_product(acc, static_cast<Value>(B(i, l)), static_cast<Value>(A(j, l)));
                }
                const value_type reduced = AT::template value<value_type>(acc);
                C(i, j) = beta_zero ? value_type(alpha * reduced)
                                    : value_type(alpha * reduced + beta * C(i, j));
            }
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < i; ++j)
                C(j, i) = C(i, j);
        return;
    }

#ifdef MTL5_HAS_BLAS
    if constexpr (detail::syr2k_dispatches_blas_v<Accumulator, MA, MB, MC>) {
        using T = value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m <= imax && k <= imax) {
            // As syrk: row-major A and B are their k x m transposes (trans 'T',
            // ld = k), and a row-major C's lower triangle is its buffer's upper.
            const int mi = static_cast<int>(m), ki = static_cast<int>(k);
            constexpr bool ab_rm = interface::is_row_major_v<MA>;
            const int ldab = std::max(1, ab_rm ? ki : mi);
            interface::blas::syr2k(interface::blas_lower_uplo<MC>(), ab_rm ? 'T' : 'N', mi, ki,
                                   static_cast<T>(alpha), A.data(), ldab, B.data(), ldab,
                                   static_cast<T>(beta), C.data(), std::max(1, mi));
            for (size_type i = 0; i < m; ++i)
                for (size_type j = 0; j < i; ++j)
                    C(j, i) = C(i, j);
            return;
        }
    }
#endif
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j <= i; ++j) {
            if (alpha_zero) {
                C(i, j) = beta_zero ? math::zero<value_type>() : value_type(beta * C(i, j));
                continue;
            }
            auto acc = math::zero<value_type>();
            for (size_type l = 0; l < k; ++l)
                acc += A(i, l) * B(j, l) + B(i, l) * A(j, l);
            C(i, j) = beta_zero ? value_type(alpha * acc) : value_type(alpha * acc + beta * C(i, j));
        }
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < i; ++j)
            C(j, i) = C(i, j);
}

} // namespace mtl
