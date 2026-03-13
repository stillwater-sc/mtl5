#pragma once
// MTL5 Benchmark Harness -- BLAS-level operation wrappers (Level 1/2/3)
// Each operation is a struct template parameterized by Backend.
// The Native specialization always calls detail::*_generic directly.
// The Blas specialization calls BLAS when available.

#include <benchmarks/harness/backend.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

// Pull in generic implementations from mtl::detail
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>

namespace mtl::bench::op {

// ═══════════════════════════════════════════════════════════════════════════
// GEMV: y = A * x
// ═══════════════════════════════════════════════════════════════════════════

template <typename Backend>
struct gemv;

/// Native: always use the generic C++ loops
template <>
struct gemv<Native> {
    template <Matrix M, Vector VIn, Vector VOut>
    static void run(const M& A, const VIn& x, VOut& y) {
        detail::mult_generic(A, x, y);
    }
    static constexpr double flops(std::size_t m, std::size_t n) {
        return static_cast<double>(2 * m * n); // multiply + add per element
    }
};

#ifdef MTL5_HAS_BLAS
/// BLAS: dispatch to sgemv_/dgemv_
template <>
struct gemv<Blas> {
    template <Matrix M, Vector VIn, Vector VOut>
    static void run(const M& A, const VIn& x, VOut& y) {
        if constexpr (interface::BlasDenseMatrix<M> &&
                      interface::BlasDenseVector<VIn> &&
                      interface::BlasDenseVector<VOut>) {
            using T = typename M::value_type;
            int m = static_cast<int>(A.num_rows());
            int n = static_cast<int>(A.num_cols());
            T alpha = math::one<T>();
            T beta  = math::zero<T>();
            if constexpr (interface::is_row_major_v<M>) {
                interface::blas::gemv('T', n, m, alpha,
                                      A.data(), n, x.data(), 1,
                                      beta, y.data(), 1);
            } else {
                interface::blas::gemv('N', m, n, alpha,
                                      A.data(), m, x.data(), 1,
                                      beta, y.data(), 1);
            }
        } else {
            // Fallback for non-eligible types
            detail::mult_generic(A, x, y);
        }
    }
    static constexpr double flops(std::size_t m, std::size_t n) {
        return static_cast<double>(2 * m * n);
    }
};
#endif

// ═══════════════════════════════════════════════════════════════════════════
// GEMM: C = A * B
// ═══════════════════════════════════════════════════════════════════════════

template <typename Backend>
struct gemm;

template <>
struct gemm<Native> {
    template <Matrix MA, Matrix MB, Matrix MC>
    static void run(const MA& A, const MB& B, MC& C) {
        detail::mult_generic(A, B, C);
    }
    static constexpr double flops(std::size_t m, std::size_t n, std::size_t k) {
        return static_cast<double>(2 * m * n * k);
    }
};

#ifdef MTL5_HAS_BLAS
template <>
struct gemm<Blas> {
    template <Matrix MA, Matrix MB, Matrix MC>
    static void run(const MA& A, const MB& B, MC& C) {
        if constexpr (interface::BlasDenseMatrix<MA> &&
                      interface::BlasDenseMatrix<MB> &&
                      interface::BlasDenseMatrix<MC>) {
            using T = typename MC::value_type;
            int m = static_cast<int>(A.num_rows());
            int n = static_cast<int>(B.num_cols());
            int k = static_cast<int>(A.num_cols());
            T alpha = math::one<T>();
            T beta  = math::zero<T>();
            if constexpr (interface::is_row_major_v<MC>) {
                interface::blas::gemm('N', 'N', n, m, k, alpha,
                                      B.data(), n, A.data(), k,
                                      beta, C.data(), n);
            } else {
                interface::blas::gemm('N', 'N', m, n, k, alpha,
                                      A.data(), m, B.data(), k,
                                      beta, C.data(), m);
            }
        } else {
            detail::mult_generic(A, B, C);
        }
    }
    static constexpr double flops(std::size_t m, std::size_t n, std::size_t k) {
        return static_cast<double>(2 * m * n * k);
    }
};
#endif

// ═══════════════════════════════════════════════════════════════════════════
// TWO_NORM: ||v||_2
// ═══════════════════════════════════════════════════════════════════════════

template <typename Backend>
struct two_norm_op;

template <>
struct two_norm_op<Native> {
    template <Vector V>
    static auto run(const V& v) {
        using mag_t = magnitude_t<typename V::value_type>;
        auto acc = math::zero<mag_t>();
        for (typename V::size_type i = 0; i < v.size(); ++i) {
            using std::abs;
            auto a = abs(v(i));
            acc += a * a;
        }
        using std::sqrt;
        return sqrt(acc);
    }
    static constexpr double flops(std::size_t n) {
        return static_cast<double>(2 * n + 1); // n mult + n add + 1 sqrt
    }
};

#ifdef MTL5_HAS_BLAS
template <>
struct two_norm_op<Blas> {
    template <Vector V>
    static auto run(const V& v) {
        if constexpr (interface::BlasDenseVector<V>) {
            return interface::blas::nrm2(static_cast<int>(v.size()), v.data(), 1);
        } else {
            return two_norm_op<Native>::run(v);
        }
    }
    static constexpr double flops(std::size_t n) {
        return static_cast<double>(2 * n + 1);
    }
};
#endif

// ═══════════════════════════════════════════════════════════════════════════
// DOT: v1 . v2
// ═══════════════════════════════════════════════════════════════════════════

template <typename Backend>
struct dot_op;

template <>
struct dot_op<Native> {
    template <Vector V1, Vector V2>
    static auto run(const V1& v1, const V2& v2) {
        return mtl::dot_real(v1, v2);
    }
    static constexpr double flops(std::size_t n) {
        return static_cast<double>(2 * n); // n mult + n add
    }
};

#ifdef MTL5_HAS_BLAS
template <>
struct dot_op<Blas> {
    template <Vector V1, Vector V2>
    static auto run(const V1& v1, const V2& v2) {
        if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2>) {
            return interface::blas::dot(static_cast<int>(v1.size()),
                                         v1.data(), 1, v2.data(), 1);
        } else {
            return mtl::dot_real(v1, v2);
        }
    }
    static constexpr double flops(std::size_t n) {
        return static_cast<double>(2 * n);
    }
};
#endif

} // namespace mtl::bench::op
