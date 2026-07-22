#pragma once
// MTL5 -- Matrix multiplication: mat*vec and mat*mat into pre-allocated output
// Optional BLAS dispatch when MTL5_HAS_BLAS is defined and types qualify.
#include <cassert>
#include <type_traits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/mat/compressed2D.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
#include <cstddef>
#include <type_traits>
#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/gemv.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <algorithm>
#endif

namespace mtl {

namespace detail {

/// Generic mat*vec: y = A * x. With an explicit `Accumulator`, each y element is
/// summed in that precision and rounded out (fused convert) to y's element type.
template <typename Accumulator = void, Matrix M, Vector VIn, Vector VOut>
void mult_generic(const M& A, const VIn& x, VOut& y) {
    using Result = typename VOut::value_type;
    if constexpr (std::is_void_v<Accumulator>) {
        for (typename M::size_type r = 0; r < A.num_rows(); ++r) {
            auto acc = math::zero<Result>();
            for (typename M::size_type c = 0; c < A.num_cols(); ++c) {
                acc += A(r, c) * x(c);
            }
            y(r) = acc;
        }
    } else {
        using Value = std::common_type_t<typename M::value_type, typename VIn::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (typename M::size_type r = 0; r < A.num_rows(); ++r) {
            Accumulator acc{};
            AT::clear(acc);
            for (typename M::size_type c = 0; c < A.num_cols(); ++c) {
                AT::add_product(acc, static_cast<Value>(A(r, c)), static_cast<Value>(x(c)));
            }
            y(r) = AT::template value<Result>(acc);
        }
    }
}

/// Sparse CRS mat*vec: y = A * x, iterating only stored nonzeros. Mirrors
/// mat::operator*(compressed2D, dense_vector)'s traversal but adds
/// Accumulator support (accumulator_traits), so mixed-precision / quire
/// accumulation works on sparse matrices too, not just dense ones.
template <typename Accumulator = void, typename V, typename P, typename VIn, typename VOut>
void mult_sparse_crs(const mat::compressed2D<V, P>& A, const VIn& x, VOut& y) {
    using Result = typename VOut::value_type;
    using size_type = typename mat::compressed2D<V, P>::size_type;
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();
    const std::size_t nrows = A.num_rows();
    if constexpr (std::is_void_v<Accumulator>) {
        using Value = std::common_type_t<V, typename VIn::value_type>;
        for (std::size_t r = 0; r < nrows; ++r) {
            auto acc = math::zero<Result>();
            for (size_type k = starts[r]; k < starts[r + 1]; ++k)
                acc += static_cast<Result>(static_cast<Value>(data[k]) * static_cast<Value>(x(indices[k])));
            y(r) = acc;
        }
    } else {
        using Value = std::common_type_t<V, typename VIn::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (std::size_t r = 0; r < nrows; ++r) {
            Accumulator acc{};
            AT::clear(acc);
            for (size_type k = starts[r]; k < starts[r + 1]; ++k)
                AT::add_product(acc, static_cast<Value>(data[k]), static_cast<Value>(x(indices[k])));
            y(r) = AT::template value<Result>(acc);
        }
    }
}

/// Generic mat*mat: C = A * B.
///
/// With `Accumulator = void` (default) each C element is summed in C's own value
/// type (unchanged). With an explicit `Accumulator`, the inner product is summed
/// in that precision via mtl::math::accumulator_traits and the result is rounded
/// out (fused convert) to C's element type on store -- the Element -> Accumulate
/// -> Result model, with the result type inferred from C.
template <typename Accumulator = void, Matrix MA, Matrix MB, Matrix MC>
void mult_generic(const MA& A, const MB& B, MC& C) {
    using Result = typename MC::value_type;
    if constexpr (std::is_void_v<Accumulator>) {
        for (typename MC::size_type r = 0; r < C.num_rows(); ++r) {
            for (typename MC::size_type c = 0; c < C.num_cols(); ++c) {
                auto acc = math::zero<Result>();
                for (typename MA::size_type k = 0; k < A.num_cols(); ++k) {
                    acc += A(r, k) * B(k, c);
                }
                C(r, c) = acc;
            }
        }
    } else {
        using Value = std::common_type_t<typename MA::value_type, typename MB::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (typename MC::size_type r = 0; r < C.num_rows(); ++r) {
            for (typename MC::size_type c = 0; c < C.num_cols(); ++c) {
                Accumulator acc{};
                AT::clear(acc);
                for (typename MA::size_type k = 0; k < A.num_cols(); ++k) {
                    AT::add_product(acc, static_cast<Value>(A(r, k)), static_cast<Value>(B(k, c)));
                }
                C(r, c) = AT::template value<Result>(acc);   // fused convert to C's type
            }
        }
    }
}

} // namespace detail

/// mat*vec multiply into pre-allocated y: y = A * x.
///
/// Mixed precision: pass an explicit `Accumulator` to sum each y element in a
/// precision distinct from the operand element type; the result is rounded out to
/// y's element type. Default `Accumulator = void` keeps the BLAS / native-fast /
/// generic dispatch unchanged.
template <typename Accumulator = void, Matrix M, Vector VIn, Vector VOut>
void mult(const M& A, const VIn& x, VOut& y) {
    assert(A.num_cols() == x.size());
    assert(A.num_rows() == y.size());

    if constexpr (interface::is_compressed2D_v<M>) {
        detail::mult_sparse_crs<Accumulator>(A, x, y);
        return;
    } else if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        detail::mult_generic<Accumulator>(A, x, y);
        return;
    } else {

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<M> &&
                  interface::BlasDenseVector<VIn> &&
                  interface::BlasDenseVector<VOut>) {
        using T = typename M::value_type;
        int m = static_cast<int>(A.num_rows());
        int n = static_cast<int>(A.num_cols());
        T alpha = math::one<T>();
        T beta  = math::zero<T>();
        if constexpr (interface::is_row_major_v<M>) {
            // Row-major: A_row is A_col^T, so y = A_row * x => y = A_col^T * x
            // BLAS: gemv('T', n, m, ..., A_data, n, x, 1, ..., y, 1)
            interface::blas::gemv('T', n, m, alpha,
                                  A.data(), n, x.data(), 1,
                                  beta, y.data(), 1);
        } else {
            interface::blas::gemv('N', m, n, alpha,
                                  A.data(), m, x.data(), 1,
                                  beta, y.data(), 1);
        }
        return;
    }
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
    // Native SIMD GEMV: preferred over the generic scalar loop for dense
    // contiguous float/double when no external BLAS handled it above.
    if constexpr (interface::BlasDenseMatrix<M> &&
                  interface::BlasDenseVector<VIn> &&
                  interface::BlasDenseVector<VOut> &&
                  std::is_same_v<typename M::value_type, typename VIn::value_type> &&
                  std::is_same_v<typename M::value_type, typename VOut::value_type>) {
        using T = typename M::value_type;
        const std::size_t mm = A.num_rows();
        const std::size_t nn = A.num_cols();
        if constexpr (interface::is_row_major_v<M>) {
            // Parallelize over output rows: each y[i] is an independent dot, so
            // the result is bit-identical across thread counts. Grain balances
            // ~64K flops per chunk. Serial by default (MTL5_NUM_THREADS=1).
            const std::size_t grain =
                nn ? std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / nn) : std::size_t{1};
            const T* Ap = A.data();
            const T* xp = x.data();
            T* yp = y.data();
            detail::thread_pool::instance().parallel_for(mm, grain,
                [&](std::size_t b, std::size_t e) {
                    detail::gemv_rowmajor<T>(e - b, nn, Ap + b * nn, nn, xp, yp + b);
                });
        } else {
            // Partition over output rows: each y[i] accumulates its columns in
            // the same order regardless of the row sub-block, so the result is
            // bit-identical across thread counts. The column stride (lda = mm) is
            // preserved for the sub-block; A + b offsets to row b.
            const std::size_t grain =
                nn ? std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / nn) : std::size_t{1};
            const T* Ap = A.data();
            const T* xp = x.data();
            T* yp = y.data();
            detail::thread_pool::instance().parallel_for(mm, grain,
                [&](std::size_t b, std::size_t e) {
                    detail::gemv_colmajor<T>(e - b, nn, Ap + b, mm, xp, yp + b);
                });
        }
        return;
    }
#endif
    detail::mult_generic(A, x, y);
    }
}

/// mat*mat multiply into pre-allocated C: C = A * B.
///
/// Mixed precision: pass an explicit `Accumulator` to sum each C element in a
/// precision distinct from the operand element type (e.g. `mult<float>(A, B, C)`
/// with bfloat16 A/B accumulates in fp32); the accumulator is rounded out to C's
/// element type on store, so C's type selects the output precision. The default
/// `Accumulator = void` keeps the BLAS / native-fast / generic dispatch
/// unchanged. The mixed path is the generic kernel (scalar; SIMD is #165).
template <typename Accumulator = void, Matrix MA, Matrix MB, Matrix MC>
void mult(const MA& A, const MB& B, MC& C) {
    assert(A.num_cols() == B.num_rows());
    assert(A.num_rows() == C.num_rows());
    assert(B.num_cols() == C.num_cols());

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
#ifdef MTL5_NATIVE_FAST_GEMM
        // SIMD widening fast path (#176): float operands accumulated in fp64
        // through the blocked GEMM, reusing the micro-kernel's widening load.
        // Restricted to the float->double case on dense contiguous matrices;
        // every other custom accumulator uses the generic scalar kernel below.
        if constexpr (std::is_same_v<Accumulator, double> &&
                      std::is_same_v<typename MA::value_type, float> &&
                      std::is_same_v<typename MB::value_type, float> &&
                      std::is_same_v<typename MC::value_type, double> &&
                      interface::BlasDenseMatrix<MA> &&
                      interface::BlasDenseMatrix<MB> &&
                      interface::BlasDenseMatrix<MC>) {
            const std::size_t M = A.num_rows();
            const std::size_t N = B.num_cols();
            const std::size_t K = A.num_cols();
            const std::ptrdiff_t a_rs = interface::is_row_major_v<MA> ? static_cast<std::ptrdiff_t>(A.num_cols()) : 1;
            const std::ptrdiff_t a_cs = interface::is_row_major_v<MA> ? 1 : static_cast<std::ptrdiff_t>(A.num_rows());
            const std::ptrdiff_t b_rs = interface::is_row_major_v<MB> ? static_cast<std::ptrdiff_t>(B.num_cols()) : 1;
            const std::ptrdiff_t b_cs = interface::is_row_major_v<MB> ? 1 : static_cast<std::ptrdiff_t>(B.num_rows());
            const unsigned nthreads = detail::gemm_default_threads();
            if constexpr (interface::is_row_major_v<MC>) {
                detail::gemm_blocked<double, float>(M, N, K, math::one<double>(),
                                                    A.data(), a_rs, a_cs,
                                                    B.data(), b_rs, b_cs,
                                                    math::zero<double>(), C.data(), N, nthreads);
            } else {
                detail::gemm_blocked<double, float>(N, M, K, math::one<double>(),
                                                    B.data(), b_cs, b_rs,
                                                    A.data(), a_cs, a_rs,
                                                    math::zero<double>(), C.data(), M, nthreads);
            }
            return;
        }
#endif
        // Custom accumulator: external BLAS / native-fast GEMM use hardware-fixed
        // accumulation, so route to the accumulator-aware generic kernel.
        detail::mult_generic<Accumulator>(A, B, C);
        return;
    } else {

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<MA> &&
                  interface::BlasDenseMatrix<MB> &&
                  interface::BlasDenseMatrix<MC> &&
                  interface::is_row_major_v<MA> == interface::is_row_major_v<MC> &&
                  interface::is_row_major_v<MB> == interface::is_row_major_v<MC>) {
        using T = typename MC::value_type;
        int m = static_cast<int>(A.num_rows());
        int n = static_cast<int>(B.num_cols());
        int k = static_cast<int>(A.num_cols());
        T alpha = math::one<T>();
        T beta  = math::zero<T>();
        if constexpr (interface::is_row_major_v<MC>) {
            // Row-major: C_row = A_row * B_row
            // C = A*B in row-major = (B^T * A^T)^T in col-major
            // So call gemm with swapped A and B pointers.
            interface::blas::gemm('N', 'N', n, m, k, alpha,
                                  B.data(), n, A.data(), k,
                                  beta, C.data(), n);
        } else {
            interface::blas::gemm('N', 'N', m, n, k, alpha,
                                  A.data(), m, B.data(), k,
                                  beta, C.data(), m);
        }
        return;
    }
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
    // Native blocked GEMM: preferred over the generic triple loop for dense
    // contiguous float/double matrices when no external BLAS handled it above.
    if constexpr (interface::BlasDenseMatrix<MA> &&
                  interface::BlasDenseMatrix<MB> &&
                  interface::BlasDenseMatrix<MC>) {
        using T = typename MC::value_type;
        const std::size_t M = A.num_rows();
        const std::size_t N = B.num_cols();
        const std::size_t K = A.num_cols();
        // Tightly-packed dense layout: ld = ncols (row-major) or nrows (col-major).
        const std::ptrdiff_t a_rs = interface::is_row_major_v<MA> ? static_cast<std::ptrdiff_t>(A.num_cols()) : 1;
        const std::ptrdiff_t a_cs = interface::is_row_major_v<MA> ? 1 : static_cast<std::ptrdiff_t>(A.num_rows());
        const std::ptrdiff_t b_rs = interface::is_row_major_v<MB> ? static_cast<std::ptrdiff_t>(B.num_cols()) : 1;
        const std::ptrdiff_t b_cs = interface::is_row_major_v<MB> ? 1 : static_cast<std::ptrdiff_t>(B.num_rows());
        const unsigned nthreads = detail::gemm_default_threads();
        if constexpr (interface::is_row_major_v<MC>) {
            detail::gemm_blocked<T>(M, N, K, math::one<T>(),
                                    A.data(), a_rs, a_cs,
                                    B.data(), b_rs, b_cs,
                                    math::zero<T>(), C.data(), N, nthreads);
        } else {
            // Col-major C: compute C^T = B^T * A^T into the same buffer, viewed as
            // a row-major N x M matrix (ld = M). Pack picks up B^T/A^T by swapping
            // each operand's strides.
            detail::gemm_blocked<T>(N, M, K, math::one<T>(),
                                    B.data(), b_cs, b_rs,
                                    A.data(), a_cs, a_rs,
                                    math::zero<T>(), C.data(), M, nthreads);
        }
        return;
    }
#endif
    detail::mult_generic(A, B, C);
    }
}

} // namespace mtl
