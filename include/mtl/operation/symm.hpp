#pragma once
// MTL5 -- symm: C <- alpha*A*B + beta*C  for symmetric A  (BLAS level-3, left side)
// A is m x m symmetric; B, C are m x n. Left side. Only A's lower triangle is
// read, on every path, as BLAS symm with uplo 'L' reads it (#692): A(i,k) for
// k > i is taken as A(k,i), so the strict upper triangle may hold anything.
// Before #692 the generic path read the full matrix and so disagreed with the
// BLAS path for a non-symmetric A. BLAS dispatch for column-major dense
// float/double when available.
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

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

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
    if constexpr (interface::BlasDenseMatrix<MA> && interface::BlasDenseMatrix<MB> &&
                  interface::BlasDenseMatrix<MC> && !interface::is_row_major_v<MA> &&
                  !interface::is_row_major_v<MB> && !interface::is_row_major_v<MC> &&
                  std::is_same_v<value_type, typename MA::value_type> &&
                  std::is_same_v<value_type, typename MB::value_type>) {
        using T = value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m <= imax && n <= imax) {
            const int ld = std::max(1, static_cast<int>(m));
            interface::blas::symm('L', 'L', static_cast<int>(m), static_cast<int>(n),
                                  static_cast<T>(alpha), A.data(), ld, B.data(), ld,
                                  static_cast<T>(beta), C.data(), ld);
            return;
        }
    }
#endif
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
