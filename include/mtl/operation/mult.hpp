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
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
#include <cstddef>
#include <type_traits>
#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/gemv.hpp>
#endif

namespace mtl {

namespace detail {

/// Generic mat*vec: y = A * x
template <Matrix M, Vector VIn, Vector VOut>
void mult_generic(const M& A, const VIn& x, VOut& y) {
    using T = typename VOut::value_type;
    for (typename M::size_type r = 0; r < A.num_rows(); ++r) {
        auto acc = math::zero<T>();
        for (typename M::size_type c = 0; c < A.num_cols(); ++c) {
            acc += A(r, c) * x(c);
        }
        y(r) = acc;
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

/// mat*vec multiply into pre-allocated y: y = A * x
template <Matrix M, Vector VIn, Vector VOut>
void mult(const M& A, const VIn& x, VOut& y) {
    assert(A.num_cols() == x.size());
    assert(A.num_rows() == y.size());

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
            detail::gemv_rowmajor<T>(mm, nn, A.data(), nn, x.data(), y.data());
        } else {
            detail::gemv_colmajor<T>(mm, nn, A.data(), mm, x.data(), y.data());
        }
        return;
    }
#endif
    detail::mult_generic(A, x, y);
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

    if constexpr (!std::is_void_v<Accumulator>) {
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
