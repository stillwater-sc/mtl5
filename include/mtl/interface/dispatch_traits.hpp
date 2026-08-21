#pragma once
// MTL5 -- Compile-time traits for BLAS/LAPACK dispatch decisions
// Used by operation files to select hardware-accelerated paths when available.

#include <type_traits>
#include <complex>
#include <mtl/tag/orientation.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/simd/batch.hpp>

namespace mtl::interface {

/// True for scalar types supported by standard BLAS/LAPACK (float, double).
template <typename T>
inline constexpr bool is_blas_scalar_v =
    std::is_same_v<T, float> || std::is_same_v<T, double>;

/// True for complex scalar types backed by LAPACK (complex<float>, complex<double>).
/// These route Hermitian eigensolves to cheev/zheev.
template <typename T>
inline constexpr bool is_lapack_complex_v = false;
template <typename T>
inline constexpr bool is_lapack_complex_v<std::complex<T>> = is_blas_scalar_v<T>;

/// Whether the requested accumulator policy permits a hardware-fixed BLAS /
/// native-fast path. External BLAS/LAPACK (and the blocked native GEMM)
/// accumulate inner products in a fixed precision and cannot honor a
/// caller-selected accumulator or result type, so ANY non-default accumulator
/// (`Accumulator != void`) MUST fall to the native (generic) kernel -- even for
/// float/double operands (e.g. a double accumulator over float storage). The
/// default `void` selects today's accelerated dispatch. Operations gate their
/// BLAS path on this predicate.
template <typename Accumulator>
inline constexpr bool accumulator_allows_blas_v = std::is_void_v<Accumulator>;

/// Concept satisfied by dense matrix types eligible for BLAS/LAPACK dispatch.
/// Requires: float/double value_type, contiguous data() pointer, num_rows/num_cols.
template <typename M>
concept BlasDenseMatrix =
    is_blas_scalar_v<typename M::value_type> &&
    requires(const M& m) {
        { m.data() } -> std::convertible_to<const typename M::value_type*>;
        { m.num_rows() };
        { m.num_cols() };
    };

/// Concept satisfied by dense matrix types eligible for complex-Hermitian
/// LAPACK dispatch (cheev/zheev). Same shape as BlasDenseMatrix but for a
/// complex<float/double> value_type. The caller is responsible for the matrix
/// actually being Hermitian; only the storage/scalar shape is checked here.
template <typename M>
concept BlasHermitianMatrix =
    is_lapack_complex_v<typename M::value_type> &&
    requires(const M& m) {
        { m.data() } -> std::convertible_to<const typename M::value_type*>;
        { m.num_rows() };
        { m.num_cols() };
    };

/// True when a vector type's storage is CONTIGUOUS -- `data()[i]` IS element i.
///
/// Every fast path below hands `data()` to a kernel that walks it with unit
/// stride, so this is a precondition of dispatching there at all. Asking only
/// for `data()` and `size()` did not establish it: `vec::strided_vector_ref`
/// supplies both while storing element i at `data()[i * stride()]`, so it was
/// accepted by `BlasDenseVector` and every operation gated on it read the wrong
/// elements and returned a confident wrong answer. Measured on a stride-2 view:
/// `dot_real` gave 14 where the answer is 44. A column of a row-major matrix is
/// exactly such a view, which is the case that makes this worth catching.
///
/// A type qualifies when it either has no stride notion at all -- `data()` and
/// `size()` are then the whole layout, which is what these concepts always
/// assumed -- or pins its stride to 1 at COMPILE time, as `vec::dense_vector`
/// does with `static constexpr size_type stride() { return 1; }`.
///
/// A runtime stride cannot qualify: `strided_vector_ref::stride()` is an
/// instance value, so no concept can admit only its unit-stride objects. Those
/// types take the generic element-wise loop instead, which indexes through
/// `operator()` and is correct for any stride. That costs a stride-1
/// `strided_vector_ref` its fast path -- the right side to err on, and
/// recoverable later with a runtime `stride() == 1` check at the call sites if
/// it ever measures.
template <typename V>
concept ContiguousVector =
    !requires(const V& v) { v.stride(); } ||     // no stride notion
    requires { requires V::stride() == 1; };     // ... or unit stride, statically

/// Concept satisfied by dense vector types eligible for BLAS dispatch.
/// Requires: float/double value_type, contiguous unit-stride storage, size().
template <typename V>
concept BlasDenseVector =
    is_blas_scalar_v<typename V::value_type> &&
    ContiguousVector<V> &&
    requires(const V& v) {
        { v.data() } -> std::convertible_to<const typename V::value_type*>;
        { v.size() };
    };

/// Concept satisfied by dense vector types eligible for MTL5's OWN SIMD kernels.
///
/// Strictly wider than `BlasDenseVector`: same storage shape (contiguous
/// `data()`, `size()`), but the value type only has to be a lane `mtl::simd`
/// can hold -- float, double, int32_t, uint32_t -- rather than a type an
/// external BLAS has a symbol for. The two are distinct on purpose: there is no
/// `sdot` for int32, so the integer lanes are reachable ONLY through the native
/// path, and gating them on the BLAS predicate is what kept them on the generic
/// scalar loop (#451).
template <typename V>
concept SimdDenseVector =
    simd::is_lane_v<typename V::value_type> &&
    ContiguousVector<V> &&
    requires(const V& v) {
        { v.data() } -> std::convertible_to<const typename V::value_type*>;
        { v.size() };
    };

/// Concept satisfied by dense vector types whose element type is not a lane but
/// can be WIDENED into one by a dot-product accumulate -- 16-bit integers, as of
/// #451 phase 2.
///
/// Separate from `SimdDenseVector` because these types are operands, not lanes:
/// `batch<std::int16_t>` does not exist, and the only kernel that accepts them
/// is `reduce_dot_widen`, which never materializes one.
template <typename V>
concept SimdNarrowVector =
    simd::is_widenable_v<typename V::value_type> &&
    ContiguousVector<V> &&
    requires(const V& v) {
        { v.data() } -> std::convertible_to<const typename V::value_type*>;
        { v.size() };
    };

/// Concept satisfied by dense vector types of an 8-bit element that the quad
/// multiply-accumulate can widen into a 32-bit lane (#451 phase 3).
///
/// Distinct from `SimdNarrowVector` because the operations differ in more than
/// width: the pairwise op takes two operands of the SAME type, while the quad op
/// accepts three PAIRINGS, one of them mixed-signedness -- and that mixed one is
/// the native instruction. So the pairing, not the element type alone, decides
/// what is dispatchable; see `mtl::quad_int_dot`.
template <typename V>
concept SimdQuadVector =
    simd::is_quad_widenable_v<typename V::value_type> &&
    ContiguousVector<V> &&
    requires(const V& v) {
        { v.data() } -> std::convertible_to<const typename V::value_type*>;
        { v.size() };
    };

/// Concept satisfied by dense matrix types eligible for MTL5's own SIMD
/// kernels -- `BlasDenseMatrix` widened to every `mtl::simd` lane type, for the
/// same reason as `SimdDenseVector`.
template <typename M>
concept SimdDenseMatrix =
    simd::is_lane_v<typename M::value_type> &&
    requires(const M& m) {
        { m.data() } -> std::convertible_to<const typename M::value_type*>;
        { m.num_rows() };
        { m.num_cols() };
    };

/// Check if a matrix type uses row-major orientation.
template <typename M>
inline constexpr bool is_row_major_v =
    std::is_same_v<typename M::orientation, tag::row_major>;

/// Check if a matrix type is a sparse compressed2D (for sparse solver dispatch).
template <typename M>
inline constexpr bool is_compressed2D_v = false;

template <typename Value, typename Params>
inline constexpr bool is_compressed2D_v<mat::compressed2D<Value, Params>> = true;

/// True for sparse matrices with BLAS-compatible scalar types (float/double).
/// These can be dispatched to SuiteSparse external solvers.
template <typename M>
inline constexpr bool is_suitesparse_eligible_v =
    is_compressed2D_v<M> && is_blas_scalar_v<typename M::value_type>;

} // namespace mtl::interface
