#pragma once
// MTL5 -- Matrix multiplication: mat*vec and mat*mat into pre-allocated output
// Optional BLAS dispatch when MTL5_HAS_BLAS is defined and types qualify.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/detail/wrapping_arithmetic.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/view/transposed_view.hpp>
// The pool is NOT part of the native-fast-GEMM option. Threading the generic
// row loop is orthogonal to routing through the blocked/SIMD kernels (#446), and
// every other threaded operation header -- dot, axpy, scale, norms, lu,
// cholesky -- includes it unconditionally.
#include <mtl/detail/thread_pool.hpp>
#include <vector>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/gemv.hpp>
#endif

namespace mtl {

namespace detail {

/// Generic mat*vec over the OUTPUT ROW RANGE [rb, re): the loop body of
/// `mult_generic` below, split out so a threaded caller can hand one contiguous
/// band of rows to each worker (#446).
///
/// Each y(r) is an independent dot product read out of A and x and written
/// nowhere else, so a partition of [0, num_rows()) into bands computes exactly
/// the values the whole-range call would, in exactly the same summation order --
/// the result is bit-identical to the serial loop, not merely close to it.
/// Preconditions the serial call already had: y must not alias A or x.
///
/// The dimension assertions are `mult`'s, restated. They are not redundant
/// paperwork: with assertions live they are also the only place the optimizer
/// learns that these extents agree, and losing them across the extra call layer
/// this refactor introduced measured as a 1.8x slowdown of the serial kernel on
/// a class-typed scalar (-O2 without NDEBUG; every optimized CMake config
/// defines NDEBUG and was unaffected). A row band is a band of the SAME matrices,
/// so every equality still holds inside it.
template <typename Accumulator = void, Matrix M, Vector VIn, Vector VOut>
void mult_generic_rows(const M& A, const VIn& x, VOut& y,
                       std::size_t rb, std::size_t re) {
    assert(A.num_cols() == x.size());
    assert(A.num_rows() == y.size());
    assert(re <= static_cast<std::size_t>(A.num_rows()));
    using Result = typename VOut::value_type;
    using size_type = typename M::size_type;
    const size_type ncols = A.num_cols();
    if constexpr (std::is_void_v<Accumulator>) {
        for (std::size_t r = rb; r < re; ++r) {
            auto acc = math::zero<Result>();
            for (size_type c = 0; c < ncols; ++c) {
                acc = generic_fma<Result>(acc, A(static_cast<size_type>(r), c), x(c));
            }
            y(static_cast<typename VOut::size_type>(r)) = acc;
        }
    } else {
        using Value = std::common_type_t<typename M::value_type, typename VIn::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (std::size_t r = rb; r < re; ++r) {
            Accumulator acc{};
            AT::clear(acc);
            for (size_type c = 0; c < ncols; ++c) {
                AT::add_product(acc, static_cast<Value>(A(static_cast<size_type>(r), c)),
                                     static_cast<Value>(x(c)));
            }
            y(static_cast<typename VOut::size_type>(r)) = AT::template value<Result>(acc);
        }
    }
}

/// Generic mat*vec: y = A * x. With an explicit `Accumulator`, each y element is
/// summed in that precision and rounded out (fused convert) to y's element type.
///
/// Stays SERIAL by name: two GEMM tests use this as the naive reference kernel a
/// fast path is compared against, and a reference that silently threads is no
/// longer a reference. `mult_generic_par` is the threading entry point.
template <typename Accumulator = void, Matrix M, Vector VIn, Vector VOut>
void mult_generic(const M& A, const VIn& x, VOut& y) {
    mult_generic_rows<Accumulator>(A, x, y, std::size_t{0},
                                   static_cast<std::size_t>(A.num_rows()));
}

/// Sparse CRS mat*vec over the OUTPUT ROW RANGE [rb, re): the loop body of
/// `mult_sparse_crs` below, split out for the threaded caller exactly as the
/// dense kernels are.
///
/// Each y(r) reads only row r's slice of the CSR arrays and writes only y(r), so
/// a partition into row bands is bit-identical to the serial loop. The GATHER
/// from x is what makes this safe to parallelize and its transpose not: reads of
/// x(indices[k]) may collide across bands, and concurrent reads are not a race.
/// y must not alias x.
///
/// The assertions are `mult`'s, restated -- see `mult_generic_rows` for why that
/// matters beyond documentation.
template <typename Accumulator = void, typename V, typename P, typename VIn, typename VOut>
void mult_sparse_crs_rows(const mat::compressed2D<V, P>& A, const VIn& x, VOut& y,
                          std::size_t rb, std::size_t re) {
    assert(A.num_cols() == x.size());
    assert(A.num_rows() == y.size());
    assert(re <= static_cast<std::size_t>(A.num_rows()));
    using Result = typename VOut::value_type;
    using size_type = typename mat::compressed2D<V, P>::size_type;
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();
    if constexpr (std::is_void_v<Accumulator>) {
        using Value = std::common_type_t<V, typename VIn::value_type>;
        for (std::size_t r = rb; r < re; ++r) {
            auto acc = math::zero<Result>();
            for (size_type k = starts[r]; k < starts[r + 1]; ++k)
                acc += static_cast<Result>(static_cast<Value>(data[k]) * static_cast<Value>(x(indices[k])));
            y(r) = acc;
        }
    } else {
        using Value = std::common_type_t<V, typename VIn::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (std::size_t r = rb; r < re; ++r) {
            Accumulator acc{};
            AT::clear(acc);
            for (size_type k = starts[r]; k < starts[r + 1]; ++k)
                AT::add_product(acc, static_cast<Value>(data[k]), static_cast<Value>(x(indices[k])));
            y(r) = AT::template value<Result>(acc);
        }
    }
}

/// Sparse CRS mat*vec: y = A * x, iterating only stored nonzeros. Mirrors
/// mat::operator*(compressed2D, dense_vector)'s traversal but adds
/// Accumulator support (accumulator_traits), so mixed-precision / quire
/// accumulation works on sparse matrices too, not just dense ones.
///
/// Serial; `mult_sparse_crs_par` is the threading entry point.
///
/// THIS LOOP IS DUPLICATED from `mult_sparse_crs_rows` above, deliberately, and
/// the duplication is worth more than it costs. Expressing it as a whole-range
/// call to the row-range kernel -- the shape the dense path uses -- measured 6.5%
/// SLOWER here (0.004396s -> 0.004681s, n=200000 nnz=2.4M, min of 5): starting
/// the loop at a parameter rather than a literal 0 costs the optimizer something
/// on the `starts[r] / starts[r+1]` pair the CSR traversal indexes with, and the
/// restated assertions that recovered the dense case did not recover this one.
/// That 6.5% lands on `dense_vector<double>` SpMV at MTL5_NUM_THREADS=1, which is
/// the default and is the inner loop of every Krylov iteration in itl/ -- the one
/// place in this file where a few percent is not a rounding error. The threaded
/// path has no such constraint: it is already paying for a pool handoff.
///
/// Keep the two bodies in step. They are the same arithmetic, and a test asserts
/// they agree exactly.
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

/// Sparse CRS mat*vec with the output rows spread over the pool.
///
/// `mat::operator*(compressed2D, dense_vector)` has threaded this same traversal
/// for any value type since #221; this is the accumulator-aware entry point
/// catching up, so `mult(compressed2D, x, y)` and `A * x` no longer disagree
/// about whether a sparse matvec uses the machine.
///
/// GRAIN IS AVERAGED, and for a sparse matrix that is an approximation rather
/// than a measurement: a row's cost is its own nnz, so contiguous bands sized off
/// nnz/nrows balance well for a banded or roughly regular matrix and badly for a
/// power-law one, where a handful of dense rows can dominate a band. Averaging is
/// what the operator* path already does, and the alternative -- partitioning on
/// the prefix sum of `starts` so each band holds equal nnz -- changes which rows
/// land together without changing any result, so it is a pure scheduling
/// improvement that belongs behind a measurement rather than in this change.
template <typename Accumulator = void, typename V, typename P, typename VIn, typename VOut>
void mult_sparse_crs_par(const mat::compressed2D<V, P>& A, const VIn& x, VOut& y) {
    if constexpr (interface::ThreadableDenseVector<VIn> &&
                  interface::ThreadableDenseVector<VOut>) {
        const std::size_t nrows = static_cast<std::size_t>(A.num_rows());
        const std::size_t nnz   = A.ref_data().size();
        const std::size_t avg   = nrows ? std::max(std::size_t{1}, nnz / nrows) : std::size_t{1};
        const std::size_t grain = interface::row_grain<V>(avg);
        thread_pool& pool = thread_pool::instance();
        if (pool.size() > 1 && nrows >= grain * 2) {
            pool.parallel_for(nrows, grain,
                [&](std::size_t b, std::size_t e) {
                    mult_sparse_crs_rows<Accumulator>(A, x, y, b, e);
                });
            return;
        }
    }
    mult_sparse_crs<Accumulator>(A, x, y);
}

/// Generic mat*mat over the OUTPUT ROW RANGE [rb, re) of C -- the mat*mat
/// counterpart of `mult_generic_rows` above, with the same bit-identity argument:
/// C(r, c) is written by whichever band owns row r, out of an inner k loop that
/// does not depend on the band. C must not alias A or B. The assertions are
/// `mult`'s, restated here for the reason given on the mat*vec overload above.
template <typename Accumulator = void, Matrix MA, Matrix MB, Matrix MC>
void mult_generic_rows(const MA& A, const MB& B, MC& C,
                       std::size_t rb, std::size_t re) {
    assert(A.num_cols() == B.num_rows());
    assert(A.num_rows() == C.num_rows());
    assert(B.num_cols() == C.num_cols());
    assert(re <= static_cast<std::size_t>(C.num_rows()));
    using Result = typename MC::value_type;
    using size_type = typename MC::size_type;
    const size_type ncols = C.num_cols();
    const typename MA::size_type kdim = A.num_cols();
    if constexpr (std::is_void_v<Accumulator>) {
        for (std::size_t rr = rb; rr < re; ++rr) {
            const size_type r = static_cast<size_type>(rr);
            for (size_type c = 0; c < ncols; ++c) {
                auto acc = math::zero<Result>();
                for (typename MA::size_type k = 0; k < kdim; ++k) {
                    acc = generic_fma<Result>(acc, A(r, k), B(k, c));
                }
                C(r, c) = acc;
            }
        }
    } else {
        using Value = std::common_type_t<typename MA::value_type, typename MB::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (std::size_t rr = rb; rr < re; ++rr) {
            const size_type r = static_cast<size_type>(rr);
            for (size_type c = 0; c < ncols; ++c) {
                Accumulator acc{};
                AT::clear(acc);
                for (typename MA::size_type k = 0; k < kdim; ++k) {
                    AT::add_product(acc, static_cast<Value>(A(r, k)), static_cast<Value>(B(k, c)));
                }
                C(r, c) = AT::template value<Result>(acc);   // fused convert to C's type
            }
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
///
/// Serial, for the same reason as the mat*vec overload: this is the naive triple
/// loop tests compare a fast path against.
template <typename Accumulator = void, Matrix MA, Matrix MB, Matrix MC>
void mult_generic(const MA& A, const MB& B, MC& C) {
    mult_generic_rows<Accumulator>(A, B, C, std::size_t{0},
                                   static_cast<std::size_t>(C.num_rows()));
}

// -- Threaded generic kernels (#446) -----------------------------------------
//
// These are the generic loops above with their output rows spread over the pool
// -- NOT a reroute into `gemm_blocked`. That distinction is the whole point.
// Blocking pays for itself through cache tiles plus a vectorized micro-kernel,
// and Highway vectorizes float/double lanes; it cannot help a software-emulated
// scalar, and the blocked kernel measured ~3.4x SLOWER than the generic one for
// such a type. Threading is the separable half: it parallelizes whichever kernel
// runs, vectorized or not.
//
// Both fall back to the serial loop when the operands are not plain dense
// storage, so this is always correct and only sometimes faster.
//
// WHY THE SERIAL CASE IS DECIDED HERE and not left to `parallel_for`, which
// already runs a single `body(0, n)` when the pool is serial or the problem is
// below the grain: because reaching the loop through the pool's callback COSTS
// something even when nothing is dispatched. `mult` asserts the three dimension
// equalities on entry, and with assertions live those give the optimizer facts
// about the loop bounds -- facts that do not survive being handed to a kernel as
// an opaque [b, e) pair from inside a lambda. Measured on a class-typed scalar,
// GEMM n=384, one thread: 0.117s calling the loop directly against 0.211s
// through the pool's serial path, for identical arithmetic. So the serial case
// calls exactly what it called before this change and the wrapper disappears;
// only a genuinely parallel run pays for the indirection, and it has 4x to spend.
// The `mm >= grain * 2` test duplicates `parallel_for`'s own -- deliberately,
// since agreeing with it is what makes this a pure fast path rather than a
// second policy.

/// mat*vec: y = A * x, output rows partitioned across the pool when A, x and y
/// are plain dense storage and there is more than one chunk's worth of work.
/// Bit-identical to `mult_generic` in every case.
template <typename Accumulator = void, Matrix M, Vector VIn, Vector VOut>
void mult_generic_par(const M& A, const VIn& x, VOut& y) {
    if constexpr (interface::ThreadableDenseMatrix<M> &&
                  interface::ThreadableDenseVector<VIn> &&
                  interface::ThreadableDenseVector<VOut>) {
        const std::size_t mm = static_cast<std::size_t>(A.num_rows());
        const std::size_t nn = static_cast<std::size_t>(A.num_cols());
        // One output row costs nn multiply-adds.
        const std::size_t grain = interface::row_grain<typename M::value_type>(nn);
        thread_pool& pool = thread_pool::instance();
        if (pool.size() > 1 && mm >= grain * 2) {
            pool.parallel_for(mm, grain,
                [&](std::size_t b, std::size_t e) {
                    mult_generic_rows<Accumulator>(A, x, y, b, e);
                });
            return;
        }
    }
    mult_generic<Accumulator>(A, x, y);
}

/// mat*mat: C = A * B, output rows of C partitioned across the pool when A, B and
/// C are plain dense storage and there is more than one chunk's worth of work.
/// Bit-identical to `mult_generic` in every case.
template <typename Accumulator = void, Matrix MA, Matrix MB, Matrix MC>
void mult_generic_par(const MA& A, const MB& B, MC& C) {
    if constexpr (interface::ThreadableDenseMatrix<MA> &&
                  interface::ThreadableDenseMatrix<MB> &&
                  interface::ThreadableDenseMatrix<MC>) {
        const std::size_t mm = static_cast<std::size_t>(C.num_rows());
        const std::size_t nn = static_cast<std::size_t>(C.num_cols());
        const std::size_t kk = static_cast<std::size_t>(A.num_cols());
        // One output row of C costs N*K multiply-adds.
        const std::size_t grain = interface::row_grain<typename MC::value_type>(nn * kk);
        thread_pool& pool = thread_pool::instance();
        if (pool.size() > 1 && mm >= grain * 2) {
            pool.parallel_for(mm, grain,
                [&](std::size_t b, std::size_t e) {
                    mult_generic_rows<Accumulator>(A, B, C, b, e);
                });
            return;
        }
    }
    mult_generic<Accumulator>(A, B, C);
}


/// Transposed sparse CRS mat*vec: y = A^T * x, O(nnz), scatter into y.
/// Mirrors mat::operator*(transposed_view<compressed2D>, dense_vector) but
/// adds Accumulator support so mixed-precision / quire accumulation works
/// on transposed sparse matvecs too. Scatter pattern means each y(j)
/// receives contributions from multiple rows r, so accumulation is done
/// per-output-element rather than per-row.
///
/// STAYS SERIAL, and the scatter is exactly why (#446 follow-up). The forward
/// kernel above threads because a row band owns its output elements outright;
/// here row r contributes to y(indices[k]) for every stored column, so two bands
/// racing to update the same y(j) is the normal case, not an edge one. The fixes
/// both cost something this kernel is not willing to pay: atomics do not exist
/// for a custom number type and would not be deterministic if they did, and
/// per-thread privatization needs O(threads * y.size()) accumulators -- already
/// the expensive part here, since a configuration-3 super-accumulator is one to
/// two orders of magnitude larger per element than the value type -- and changes
/// the summation grouping, which forfeits the bit-identity every other threaded
/// path in this file guarantees. Threading this wants a different algorithm (a
/// CSC view, or a symbolic split of the output range), not a partition of this
/// loop.
template <typename Accumulator = void, typename V, typename P, typename VIn, typename VOut>
void mult_sparse_crs_transposed(const mat::view::transposed_view<mat::compressed2D<V, P>>& At,
                                 const VIn& x, VOut& y) {
    using Result = typename VOut::value_type;
    using size_type = typename mat::compressed2D<V, P>::size_type;
    const auto& A = At.base();
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();
    const std::size_t nrows = A.num_rows();
    for (typename VOut::size_type i = 0; i < y.size(); ++i)
        y(i) = math::zero<Result>();
    if constexpr (std::is_void_v<Accumulator>) {
        using Value = std::common_type_t<V, typename VIn::value_type>;
        for (std::size_t r = 0; r < nrows; ++r)
            for (size_type k = starts[r]; k < starts[r + 1]; ++k)
                y(indices[k]) += static_cast<Result>(static_cast<Value>(data[k]) * static_cast<Value>(x(r)));
    } else {
        using Value = std::common_type_t<V, typename VIn::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        std::vector<Accumulator> accs(y.size());
        for (auto& a : accs) AT::clear(a);
        for (std::size_t r = 0; r < nrows; ++r)
            for (size_type k = starts[r]; k < starts[r + 1]; ++k)
                AT::add_product(accs[indices[k]], static_cast<Value>(data[k]), static_cast<Value>(x(r)));
        for (typename VOut::size_type j = 0; j < y.size(); ++j)
            y(j) = AT::template value<Result>(accs[j]);
    }
}

} // namespace detail

/// mat*vec multiply into pre-allocated y: y = A * x.
///
/// Mixed precision: pass an explicit `Accumulator` to sum each y element in a
/// precision distinct from the operand element type; the result is rounded out to
/// y's element type. Default `Accumulator = void` keeps the BLAS / native-fast /
/// generic dispatch unchanged.
/// mat*vec multiply for a transposed sparse view: y = A^T * x, O(nnz).
/// Mirrors mat::operator*(transposed_view<compressed2D>, dense_vector) but
/// adds Accumulator support for mixed-precision / quire accumulation.
template <typename Accumulator = void, typename V, typename P, typename VIn, typename VOut>
void mult(const mat::view::transposed_view<mat::compressed2D<V, P>>& At, const VIn& x, VOut& y) {
    assert(At.num_cols() == x.size());
    assert(At.num_rows() == y.size());
    detail::mult_sparse_crs_transposed<Accumulator>(At, x, y);
}

template <typename Accumulator = void, Matrix M, Vector VIn, Vector VOut>
void mult(const M& A, const VIn& x, VOut& y) {
    assert(A.num_cols() == x.size());
    assert(A.num_rows() == y.size());

    if constexpr (interface::is_compressed2D_v<M>) {
        detail::mult_sparse_crs_par<Accumulator>(A, x, y);
        return;
    } else if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        detail::mult_generic_par<Accumulator>(A, x, y);
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
    // contiguous lane types when no external BLAS handled it above.
    //
    // Gated on the Simd* concepts, not the Blas* ones, so the integer lanes are
    // reachable (#451 phase 0). There is no external ?gemv for int32, so this
    // native path is the only one they can take; the BLAS branch above is
    // unreachable for them and needs no extra guard. The result is exact mod
    // 2^32 and bit-identical across lane counts and thread partitions -- the
    // per-row partitioning below is already order-preserving.
    if constexpr (interface::SimdDenseMatrix<M> &&
                  interface::SimdDenseVector<VIn> &&
                  interface::SimdDenseVector<VOut> &&
                  std::is_same_v<typename M::value_type, typename VIn::value_type> &&
                  std::is_same_v<typename M::value_type, typename VOut::value_type>) {
        using T = typename M::value_type;
        const std::size_t mm = A.num_rows();
        const std::size_t nn = A.num_cols();
        if constexpr (interface::is_row_major_v<M>) {
            // Parallelize over output rows: each y[i] is an independent dot, so
            // the result is bit-identical across thread counts. Grain balances
            // ~64K flops per chunk. Serial by default (MTL5_NUM_THREADS=1).
            const std::size_t grain = interface::row_grain<T>(nn);
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
            const std::size_t grain = interface::row_grain<T>(nn);
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
    // Everything the accelerated paths could not take -- every software-emulated
    // format among them -- lands here, and lands here THREADED (#446).
    detail::mult_generic_par(A, x, y);
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
#ifdef MTL5_NATIVE_FAST_GEMM
        // The integer analogue: 8- or 16-bit operands accumulated in 32 bits
        // through the same blocked nest and the same widening load.
        //
        // This is the WIDEN-ON-LOAD path, not the hardware quad product. It
        // promotes narrow operands into int32 lanes and does an ordinary
        // multiply-add, so it captures the memory-traffic win -- an int8 GEMM
        // reads an eighth of the bytes of an fp64 one -- but not `vpdpbusd`.
        //
        // That kernel now exists: see `mult_quad` below, which is FASTER here
        // (1.26x symmetric, 1.64x for u8 x i8) even on a machine that only
        // emulates the instruction. This path is deliberately NOT rerouted to
        // it -- both are correct for an (i8,i8) pair, so switching silently
        // would destroy the within-machine control the benchmark needs, and
        // nothing in a timing can tell the two apart afterwards. The default
        // moves when there is a measurement behind it.
        //
        // Same signedness on both operands and the accumulator, matching
        // batch::load_widen: a mixed pair is a question about the caller's
        // intent, not something to guess at.
        if constexpr (std::is_integral_v<Accumulator> &&
                      simd::is_lane_v<Accumulator> &&
                      std::is_same_v<typename MC::value_type, Accumulator> &&
                      std::is_same_v<typename MA::value_type, typename MB::value_type> &&
                      std::is_integral_v<typename MA::value_type> &&
                      (sizeof(typename MA::value_type) < sizeof(Accumulator)) &&
                      (std::is_signed_v<typename MA::value_type> ==
                       std::is_signed_v<Accumulator>) &&
                      interface::SimdDenseMatrix<MC> &&
                      interface::ContiguousMatrixData<MA> &&
                      interface::ContiguousMatrixData<MB>) {
            using TAB = typename MA::value_type;
            const std::size_t M = A.num_rows();
            const std::size_t N = B.num_cols();
            const std::size_t K = A.num_cols();
            const std::ptrdiff_t a_rs = interface::is_row_major_v<MA> ? static_cast<std::ptrdiff_t>(A.num_cols()) : 1;
            const std::ptrdiff_t a_cs = interface::is_row_major_v<MA> ? 1 : static_cast<std::ptrdiff_t>(A.num_rows());
            const std::ptrdiff_t b_rs = interface::is_row_major_v<MB> ? static_cast<std::ptrdiff_t>(B.num_cols()) : 1;
            const std::ptrdiff_t b_cs = interface::is_row_major_v<MB> ? 1 : static_cast<std::ptrdiff_t>(B.num_rows());
            const unsigned nthreads = detail::gemm_default_threads();
            if constexpr (interface::is_row_major_v<MC>) {
                detail::gemm_blocked<Accumulator, TAB>(M, N, K, math::one<Accumulator>(),
                                                       A.data(), a_rs, a_cs,
                                                       B.data(), b_rs, b_cs,
                                                       math::zero<Accumulator>(), C.data(), N, nthreads);
            } else {
                detail::gemm_blocked<Accumulator, TAB>(N, M, K, math::one<Accumulator>(),
                                                       B.data(), b_cs, b_rs,
                                                       A.data(), a_cs, a_rs,
                                                       math::zero<Accumulator>(), C.data(), M, nthreads);
            }
            return;
        }
#endif
        // Custom accumulator: external BLAS / native-fast GEMM use hardware-fixed
        // accumulation, so route to the accumulator-aware generic kernel -- which
        // still partitions its output rows across the pool (#446).
        detail::mult_generic_par<Accumulator>(A, B, C);
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
    // contiguous matrices when no external BLAS handled it above.
    //
    // Gated on the Simd concepts, so int32 matrices reach the blocked nest too.
    // There is no external ?gemm for int32, so the BLAS branch above cannot have
    // taken them, and the generic triple loop is what they were falling to.
    if constexpr (interface::SimdDenseMatrix<MA> &&
                  interface::SimdDenseMatrix<MB> &&
                  interface::SimdDenseMatrix<MC> &&
                  std::is_same_v<typename MA::value_type, typename MC::value_type> &&
                  std::is_same_v<typename MB::value_type, typename MC::value_type>) {
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
    // Same as the mat*vec tail: whatever the blocked/BLAS gates rejected still
    // gets its output rows spread over the pool (#446).
    detail::mult_generic_par(A, B, C);
    }
}

/// C = A * B through the QUAD MULTIPLY-ACCUMULATE micro-kernel -- VNNI
/// `vpdpbusd` on x86, `SDOT`/`UDOT` on NEON (#451 phase 5).
///
/// WHY THIS IS A SEPARATE ENTRY POINT and not a silent upgrade inside `mult`.
/// Both kernels are correct for an `(i8, i8)` pair, so making `mult` choose
/// between them would mean the same call measures a different thing on different
/// machines -- and the benchmark README already warns that nothing in a timing
/// distinguishes `vpdpbusd` from its decomposition. The whole reason to build
/// this kernel was to find out what the instruction is worth, and that needs
/// both arms compilable in ONE binary so they can be compared within a machine.
/// The programme has paid for a missing within-machine control once already: the
/// instruction's share looked like 1.42x from a single Zen 4 run and came out at
/// ~1.2x once an i7 could run both arms decomposed. So `mult` is unchanged, this
/// is explicit, and the default flips later with data behind it.
///
/// The operand pair must be one the hardware op accepts -- `(i8,i8)`, `(u8,i8)`
/// or `(u8,u8)`, accumulating in the matching 32-bit type. `(i8,u8)` is rejected
/// rather than reordered: unlike a dot product a GEMM is NOT symmetric in its
/// operands, so there is no argument swap that fixes it for the caller, and
/// `(u8,i8)` -- unsigned activations against signed weights -- is the shape
/// quantized inference actually produces.
///
/// Falls back to the accumulator-aware generic loop when the matrices are not
/// dense and contiguous, so this is always correct, never merely fast. One case
/// is worth naming: COLUMN-MAJOR C is computed as C^T = B^T A^T, which swaps the
/// operand order, and `(u8,i8)` swapped is `(i8,u8)` -- not a pairing the
/// instruction has. That combination therefore takes the generic loop. The
/// symmetric pairs are unaffected.
template <typename Accumulator, Matrix MA, Matrix MB, Matrix MC>
void mult_quad(const MA& A, const MB& B, MC& C) {
    assert(A.num_cols() == B.num_rows());
    assert(A.num_rows() == C.num_rows());
    assert(B.num_cols() == C.num_cols());

    using TA = typename MA::value_type;
    using TB = typename MB::value_type;
#ifdef MTL5_NATIVE_FAST_GEMM
    // The layout gate comes FIRST, and `orientation` is part of it. `Matrix` does
    // not require that alias -- a sparse matrix and a transposed view both lack
    // it -- and `interface::is_row_major_v` has no default, so naming it outside
    // a discarded branch turns "falls back to the generic loop" into a hard
    // compile error for exactly the types the fallback exists for.
    if constexpr (interface::SimdDenseMatrix<MC> &&
                  interface::ContiguousMatrixData<MA> &&
                  interface::ContiguousMatrixData<MB> &&
                  std::is_same_v<typename MC::value_type, Accumulator> &&
                  requires { typename MA::orientation;
                             typename MB::orientation;
                             typename MC::orientation; }) {
    // Row-major C keeps the operand order, so the pair is (TA, TB); col-major C
    // computes C^T = B^T A^T and therefore needs (TB, TA) to be a pairing too.
    constexpr bool row_major_c = interface::is_row_major_v<MC>;
    constexpr bool kernel_ok = row_major_c ? detail::is_quad_gemm<Accumulator, TA, TB>()
                                           : detail::is_quad_gemm<Accumulator, TB, TA>();
    if constexpr (kernel_ok) {
        const std::size_t M = A.num_rows();
        const std::size_t N = B.num_cols();
        const std::size_t K = A.num_cols();
        const std::ptrdiff_t a_rs = interface::is_row_major_v<MA> ? static_cast<std::ptrdiff_t>(A.num_cols()) : 1;
        const std::ptrdiff_t a_cs = interface::is_row_major_v<MA> ? 1 : static_cast<std::ptrdiff_t>(A.num_rows());
        const std::ptrdiff_t b_rs = interface::is_row_major_v<MB> ? static_cast<std::ptrdiff_t>(B.num_cols()) : 1;
        const std::ptrdiff_t b_cs = interface::is_row_major_v<MB> ? 1 : static_cast<std::ptrdiff_t>(B.num_rows());
        const unsigned nthreads = detail::gemm_default_threads();
        constexpr auto kern = detail::gemm_kernel::quad;   // asked for out loud
        if constexpr (row_major_c) {
            detail::gemm_blocked<Accumulator, TA, TB, kern>(
                M, N, K, math::one<Accumulator>(),
                A.data(), a_rs, a_cs, B.data(), b_rs, b_cs,
                math::zero<Accumulator>(), C.data(), N, nthreads);
        } else {
            detail::gemm_blocked<Accumulator, TB, TA, kern>(
                N, M, K, math::one<Accumulator>(),
                B.data(), b_cs, b_rs, A.data(), a_cs, a_rs,
                math::zero<Accumulator>(), C.data(), M, nthreads);
        }
        return;
    }
    }
#endif
    static_assert(simd::QuadPair<TA, TB>,
                  "mult_quad takes an 8-bit operand pair the hardware op accepts -- "
                  "(i8,i8), (u8,i8) or (u8,u8). (i8,u8) is absent on purpose: a GEMM "
                  "is not symmetric in its operands, so swapping is the caller's "
                  "decision, not this function's");
    detail::mult_generic_par<Accumulator>(A, B, C);
}

} // namespace mtl
