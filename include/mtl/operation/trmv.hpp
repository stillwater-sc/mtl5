#pragma once
// MTL5 -- trmv: x <- A*x  for triangular A  (BLAS level-2)
// A is upper (upper=true) or lower triangular; the opposite triangle is ignored.
// unit_diag treats the diagonal as all ones. No transpose in this variant.
//
// BLAS dispatch for dense float/double of either orientation (#747): a
// row-major n x n buffer read column-major is A^T, so a row-major lower A is
// ?trmv('U', 'T', ...) on its buffer and an upper one ?trmv('L', 'T', ...).
//
// MTL5_NATIVE_FAST_GEMM without BLAS (#747): above detail::triangular_block
// rows, A's diagonal is walked in blocks of that many rows. Each block's
// off-diagonal panel goes through the SIMD gemv into a buffer, and its
// diagonal block is the row loop, which adds the buffer last. The blocks go
// bottom-up for a lower A and top-down for an upper one, so every x(j) a
// block reads still holds its original value. Rounding is the SIMD gemv's.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
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

/// Whether trmv<Accumulator>(A, x, ...) calls ?trmv (for n that fits an int).
template <typename Accumulator, typename M, typename VX>
inline constexpr bool trmv_dispatches_blas_v = [] {
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

/// Whether trmv<Accumulator>(A, x, ...) takes the blocked SIMD path for
/// n > triangular_block (#747): native-fast, no BLAS, float/double.
template <typename Accumulator, typename M, typename VX>
inline constexpr bool trmv_uses_blocked_v = [] {
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
/// The blocked trmv: x := A x on n > triangular_block rows (#747).
template <typename T, typename M, typename VX>
void trmv_blocked(const M& A, VX& x, bool upper, bool unit_diag) {
    const std::size_t n = A.num_rows();
    const std::size_t nb = triangular_block;
    constexpr bool rm = interface::is_row_major_v<M>;
    const T* a = A.data();
    T* xp = x.data();
    std::vector<T> t(nb);
    const auto at = [&](std::size_t i, std::size_t j) -> T { return rm ? a[i * n + j] : a[j * n + i]; };
    if (!upper) {
        // Bottom-up: rows above the block are not overwritten yet.
        for (std::size_t i1 = n; i1 > 0;) {
            const std::size_t i0 = i1 > nb ? i1 - nb : 0;
            gemv_panel<T>(rm, n, a, i0, 0, i1 - i0, i0, xp, t.data());   // A(I, 0:i0) x(0:i0)
            for (std::size_t i = i1; i-- > i0;) {                        // descending: x(j<i) still old
                T acc = unit_diag ? xp[i] : T(at(i, i) * xp[i]);
                for (std::size_t j = i0; j < i; ++j) acc += at(i, j) * xp[j];
                xp[i] = acc + t[i - i0];
            }
            i1 = i0;
        }
    } else {
        // Top-down: rows below the block are not overwritten yet.
        for (std::size_t i0 = 0; i0 < n; i0 += nb) {
            const std::size_t i1 = i0 + nb < n ? i0 + nb : n;
            gemv_panel<T>(rm, n, a, i0, i1, i1 - i0, n - i1, xp + i1, t.data());   // A(I, i1:n) x(i1:n)
            for (std::size_t i = i0; i < i1; ++i) {                                  // ascending: x(j>i) still old
                T acc = unit_diag ? xp[i] : T(at(i, i) * xp[i]);
                for (std::size_t j = i + 1; j < i1; ++j) acc += at(i, j) * xp[j];
                xp[i] = acc + t[i - i0];
            }
        }
    }
}
#endif

} // namespace detail

/// Triangular matrix-vector product: x = A*x, A upper/lower triangular.
///
/// Mixed precision: pass an explicit `Accumulator` to form each row's
/// reduction in a precision distinct from the operand type (#261, Part C).
/// Default `Accumulator = void` keeps the BLAS / generic dispatch byte for
/// byte.
///
/// Gated on `interface::accumulator_allows_blas_v`, same as symv/gemv: a
/// custom accumulator cannot be honored by external BLAS.
///
/// The diagonal term (1*x(i) for unit_diag, A(i,i)*x(i) otherwise) is fed
/// through `add_product` as the first term of a `clear`-seeded reduction,
/// same shape as symv's row loop -- not special-cased as a seed -- so the
/// diagonal gets the same rounding treatment as every off-diagonal term.
///
/// The traversal order (forward for upper, reverse for lower) is an
/// in-place-overwrite hazard, not a precision concern: x(j) for the
/// off-diagonal terms must still hold its original value when read, which
/// this order guarantees regardless of Accumulator. Unchanged by this patch.
template <typename Accumulator = void, Matrix M, Vector VX>
void trmv(const M& A, VX& x, bool upper, bool unit_diag = false) {
    using value_type = typename VX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n);

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<value_type, typename M::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        if (upper) {
            for (size_type i = 0; i < n; ++i) {
                Accumulator acc{};
                AT::clear(acc);
                const Value diag = unit_diag ? Value(1) : static_cast<Value>(A(i, i));
                AT::add_product(acc, diag, static_cast<Value>(x(i)));
                for (size_type j = i + 1; j < n; ++j)
                    AT::add_product(acc, static_cast<Value>(A(i, j)), static_cast<Value>(x(j)));
                x(i) = AT::template value<value_type>(acc);
            }
        } else {
            for (size_type ii = n; ii-- > 0; ) {
                Accumulator acc{};
                AT::clear(acc);
                const Value diag = unit_diag ? Value(1) : static_cast<Value>(A(ii, ii));
                AT::add_product(acc, diag, static_cast<Value>(x(ii)));
                for (size_type j = 0; j < ii; ++j)
                    AT::add_product(acc, static_cast<Value>(A(ii, j)), static_cast<Value>(x(j)));
                x(ii) = AT::template value<value_type>(acc);
            }
        }
        return;
    }

#ifdef MTL5_HAS_BLAS
    if constexpr (detail::trmv_dispatches_blas_v<Accumulator, M, VX>) {
        if (n == 0) return;   // x is empty; skip BLAS (#693)
        if (n <= static_cast<size_type>(std::numeric_limits<int>::max())) {
            // BLAS requires lda >= 1 even for an empty A (#693). A row-major
            // buffer read column-major is A^T: the other triangle, transposed.
            constexpr bool rm = interface::is_row_major_v<M>;
            const char uplo = (upper != rm) ? 'U' : 'L';
            interface::blas::trmv(uplo, rm ? 'T' : 'N', unit_diag ? 'U' : 'N',
                                  static_cast<int>(n), A.data(), std::max(1, static_cast<int>(n)),
                                  x.data(), 1);
            return;
        }
    }
#endif
#if defined(MTL5_NATIVE_FAST_GEMM) && !defined(MTL5_HAS_BLAS)
    if constexpr (detail::trmv_uses_blocked_v<Accumulator, M, VX>) {
        if (static_cast<std::size_t>(n) > detail::triangular_block) {
            detail::trmv_blocked<value_type>(A, x, upper, unit_diag);
            return;
        }
    }
#endif
    if (upper) {
        // x_i depends on x_j for j >= i; forward order is safe.
        for (size_type i = 0; i < n; ++i) {
            auto acc = unit_diag ? x(i) : A(i, i) * x(i);
            for (size_type j = i + 1; j < n; ++j)
                acc += A(i, j) * x(j);
            x(i) = acc;
        }
    } else {
        // x_i depends on x_j for j <= i; reverse order is safe.
        for (size_type ii = n; ii-- > 0; ) {
            auto acc = unit_diag ? x(ii) : A(ii, ii) * x(ii);
            for (size_type j = 0; j < ii; ++j)
                acc += A(ii, j) * x(j);
            x(ii) = acc;
        }
    }
}

} // namespace mtl
