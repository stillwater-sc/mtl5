#pragma once
// MTL5 -- trsv: solve A*x = b for triangular A  (BLAS level-2)
// A BLAS-style facade over the forward/back-substitution solvers, with an
// in-place (solve A*x = x) form. A is upper (upper=true) or lower triangular;
// unit_diag treats the diagonal as all ones. No transpose in this variant.
//
// BLAS dispatch for dense float/double of either orientation (#747): a
// row-major buffer read column-major is A^T, so a row-major lower A is
// ?trsv('U', 'T', ...) on its buffer and an upper one ?trsv('L', 'T', ...).
//
// MTL5_NATIVE_FAST_GEMM without BLAS (#747): above detail::triangular_block
// rows, A's diagonal is walked in blocks of that many rows. The panel of each
// block against the rows already solved goes through the SIMD gemv into a
// buffer, and the diagonal block is the substitution loop, which subtracts
// the buffer last. The blocks go top-down for a lower A and bottom-up for an
// upper one. Rounding is the SIMD gemv's.
//
// Mixed precision: pass an explicit `Accumulator`, forwarded through to
// lower_trisolve/upper_trisolve (#261, Part C). Gated on
// `interface::accumulator_allows_blas_v`, same as symv/trmv -- a custom
// accumulator cannot be honored by external BLAS, so the BLAS dispatch below
// is now conditional on the gate rather than unconditional.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <vector>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
#include <mtl/detail/gemv.hpp>
#endif

namespace mtl {

namespace detail {

/// Whether trsv<Accumulator>(A, x, ...) calls ?trsv (for n that fits an int).
template <typename Accumulator, typename M, typename VX>
inline constexpr bool trsv_dispatches_blas_v = [] {
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::accumulator_allows_blas_v<Accumulator> && interface::BlasDenseMatrix<M> &&
                  interface::BlasDenseVector<VX>)
        return std::is_same_v<typename VX::value_type, typename M::value_type>;
    else
        return false;
#else
    return false;
#endif
}();

/// Whether trsv<Accumulator>(A, x, ...) takes the blocked SIMD path for
/// n > triangular_block (#747): native-fast, no BLAS, float/double.
template <typename Accumulator, typename M, typename VX>
inline constexpr bool trsv_uses_blocked_v = [] {
#if defined(MTL5_NATIVE_FAST_GEMM) && !defined(MTL5_HAS_BLAS)
    if constexpr (std::is_void_v<Accumulator> && interface::BlasDenseMatrix<M> &&
                  interface::BlasDenseVector<VX> && requires { typename M::orientation; })
        return std::is_same_v<typename VX::value_type, typename M::value_type>;
    else
        return false;
#else
    return false;
#endif
}();

#if defined(MTL5_NATIVE_FAST_GEMM) && !defined(MTL5_HAS_BLAS)
/// The blocked trsv: solve A x = x in place on n > triangular_block rows (#747).
template <typename T, typename M, typename VX>
void trsv_blocked(const M& A, VX& x, bool upper, bool unit_diag) {
    const std::size_t n = A.num_rows();
    const std::size_t nb = triangular_block;
    constexpr bool rm = interface::is_row_major_v<M>;
    const T* a = A.data();
    T* xp = x.data();
    std::vector<T> t(nb);
    const auto at = [&](std::size_t i, std::size_t j) -> T { return rm ? a[i * n + j] : a[j * n + i]; };
    if (!upper) {
        // Top-down: x(0:i0) is solved.
        for (std::size_t i0 = 0; i0 < n; i0 += nb) {
            const std::size_t i1 = i0 + nb < n ? i0 + nb : n;
            gemv_panel<T>(rm, n, a, i0, 0, i1 - i0, i0, xp, t.data());   // A(I, 0:i0) x(0:i0)
            for (std::size_t i = i0; i < i1; ++i) {
                T sum = T(0);
                for (std::size_t j = i0; j < i; ++j) sum += at(i, j) * xp[j];
                const T r = xp[i] - (sum + t[i - i0]);
                xp[i] = unit_diag ? r : T(r / at(i, i));
            }
        }
    } else {
        // Bottom-up: x(i1:n) is solved.
        for (std::size_t i1 = n; i1 > 0;) {
            const std::size_t i0 = i1 > nb ? i1 - nb : 0;
            gemv_panel<T>(rm, n, a, i0, i1, i1 - i0, n - i1, xp + i1, t.data());   // A(I, i1:n) x(i1:n)
            for (std::size_t i = i1; i-- > i0;) {
                T sum = T(0);
                for (std::size_t j = i + 1; j < i1; ++j) sum += at(i, j) * xp[j];
                const T r = xp[i] - (sum + t[i - i0]);
                xp[i] = unit_diag ? r : T(r / at(i, i));
            }
            i1 = i0;
        }
    }
}
#endif

} // namespace detail

/// Solve A*x = x in place, A upper/lower triangular.
template <typename Accumulator = void, Matrix M, Vector VX>
void trsv(const M& A, VX& x, bool upper, bool unit_diag = false) {
    using value_type = typename VX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n);

    if constexpr (interface::accumulator_allows_blas_v<Accumulator>) {
#ifdef MTL5_HAS_BLAS
        if constexpr (detail::trsv_dispatches_blas_v<Accumulator, M, VX>) {
            if (n == 0) return;   // x is empty; skip BLAS (#693)
            if (n <= static_cast<size_type>(std::numeric_limits<int>::max())) {
                // BLAS requires lda >= 1 even for an empty A (#693). A row-major
                // buffer read column-major is A^T: the other triangle, transposed.
                constexpr bool rm = interface::is_row_major_v<M>;
                const char uplo = (upper != rm) ? 'U' : 'L';
                interface::blas::trsv(uplo, rm ? 'T' : 'N', unit_diag ? 'U' : 'N',
                                      static_cast<int>(n), A.data(), std::max(1, static_cast<int>(n)),
                                      x.data(), 1);
                return;
            }
        }
#endif
#if defined(MTL5_NATIVE_FAST_GEMM) && !defined(MTL5_HAS_BLAS)
        if constexpr (detail::trsv_uses_blocked_v<Accumulator, M, VX>) {
            if (static_cast<std::size_t>(n) > detail::triangular_block) {
                detail::trsv_blocked<value_type>(A, x, upper, unit_diag);
                return;
            }
        }
#endif
    }
    if (upper)
        upper_trisolve<Accumulator>(A, x, unit_diag);
    else
        lower_trisolve<Accumulator>(A, x, unit_diag);
}

/// Solve A*x = b, writing the solution into x (b unchanged).
template <typename Accumulator = void, Matrix M, Vector VX, Vector VB>
void trsv(const M& A, VX& x, const VB& b, bool upper, bool unit_diag = false) {
    using size_type = typename M::size_type;
    for (size_type i = 0; i < b.size(); ++i) x(i) = b(i);
    trsv<Accumulator>(A, x, upper, unit_diag);
}

} // namespace mtl
