#pragma once
// MTL5 -- block_matrix_view: a read-only Matrix over block-quantized storage
// (#629).
//
// ML weights, and other compressed numerical data, are increasingly stored as
// blocks of 16-256 narrow integers or minifloats sharing a scale. Before this
// type a consumer had two choices: dequantize into dense2D<float>, at 4-8x the
// memory and bandwidth, or write its own view and kernels and give up Matrix
// interop. This is the format-agnostic part of the second route: a Matrix
// whose storage is the encoded bytes, parameterized by a BlockScheme
// (concepts/block_scheme.hpp) that knows how to decode them.
//
//   * NON-OWNING. The bytes usually live in a file or a memory map; the caller
//     keeps them alive for the view's lifetime.
//   * ROW-MAJOR, BLOCKED ALONG ROWS -- the layout GGML uses for weights and the
//     one y = W x reads in order. Each row is cols / block_size consecutive
//     blocks. Column-blocked storage is not supported.
//   * READ-ONLY. Writing one element of a block format is a read-modify-write
//     of its whole block, and usually a change of the shared scale; that is an
//     encode, not an assignment.
//
// Elements read as float, decoded on access. That makes the view a Matrix, so
// generic code works on it, but the per-element path is for interop and tests.
// The bulk operations below decode a row at a time:
//
//   convert(view, dense2D)   the reference decode into a dense matrix;
//   mult(view, x, y)         y = A x: a scheme-supplied fused kernel when there
//                            is one (block_kernels<S>), otherwise decode each
//                            row and dot it with x;
//   mult(view, x, y, scratch) the same, with the fused kernel's working memory
//                            supplied by the caller;
//   view * x                 the same, into a new dense_vector.

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mtl/concepts/block_scheme.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/detail/wrapping_arithmetic.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/simd/algorithm.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::mat {

/// Read-only, row-major view of a `rows x cols` matrix stored as blocks of the
/// scheme `S`, `cols / S::block_size` blocks per row.
template <BlockScheme S>
class block_matrix_view {
public:
    using scheme_type     = S;
    using value_type      = float;
    using size_type       = std::size_t;
    using const_reference = float;
    using reference       = float;   // read-only: elements are decoded values

    static constexpr size_type block_size  = S::block_size;
    static constexpr size_type block_bytes = S::block_bytes;

    /// View `bytes` as a `rows x cols` matrix. Throws std::invalid_argument
    /// unless `cols` is a multiple of the block size and `bytes` holds exactly
    /// `rows * (cols / block_size)` blocks.
    ///
    /// The byte count is computed with overflow checks. The shape usually comes
    /// from a file header, and a product that wrapped would let a huge `rows`
    /// validate against a small span and every later access read past it. A
    /// shape whose byte or element count does not fit in size_t is rejected.
    block_matrix_view(size_type rows, size_type cols, std::span<const std::byte> bytes)
        : rows_(rows), cols_(cols), blocks_per_row_(cols / block_size), data_(bytes.data()) {
        if (cols % block_size != 0)
            throw std::invalid_argument("block_matrix_view: cols (" + std::to_string(cols) +
                                        ") is not a multiple of the block size (" +
                                        std::to_string(block_size) + ")");
        constexpr size_type max = std::numeric_limits<size_type>::max();
        if (blocks_per_row_ > max / block_bytes ||
            (rows != 0 && blocks_per_row_ * block_bytes > max / rows) ||
            (rows != 0 && cols > max / rows))
            throw std::invalid_argument("block_matrix_view: " + std::to_string(rows) + " x " +
                                        std::to_string(cols) +
                                        " overflows size_t in elements or bytes");
        if (bytes.size() != rows * blocks_per_row_ * block_bytes)
            throw std::invalid_argument("block_matrix_view: " + std::to_string(bytes.size()) +
                                        " bytes do not hold " + std::to_string(rows) + " x " +
                                        std::to_string(cols) + " in blocks of " +
                                        std::to_string(block_bytes) + " bytes");
    }

    size_type num_rows() const noexcept { return rows_; }
    size_type num_cols() const noexcept { return cols_; }
    size_type size()     const noexcept { return rows_ * cols_; }
    size_type blocks_per_row() const noexcept { return blocks_per_row_; }

    /// Element (r, c), decoded. One element if the scheme can do that, the
    /// enclosing block otherwise.
    float operator()(size_type r, size_type c) const {
        assert(r < rows_ && c < cols_);
        const std::byte* blk = block_data(r, c / block_size);
        if constexpr (ElementDecodableScheme<S>) {
            return S::decode_element(blk, c % block_size);
        } else {
            std::array<float, block_size> buf;
            S::decode(blk, buf.data());
            return buf[c % block_size];
        }
    }

    /// The encoded bytes of block `b` of row `r`.
    std::span<const std::byte> block(size_type r, size_type b) const {
        assert(r < rows_ && b < blocks_per_row_);
        return {block_data(r, b), block_bytes};
    }

    /// The encoded bytes of row `r`: `blocks_per_row()` consecutive blocks.
    std::span<const std::byte> row(size_type r) const {
        assert(r < rows_);
        return {block_data(r, 0), blocks_per_row_ * block_bytes};
    }

    /// Decode row `r` into `cols` floats at `dst`.
    void decode_row(size_type r, float* dst) const {
        assert(r < rows_);
        const std::byte* p = block_data(r, 0);
        for (size_type b = 0; b < blocks_per_row_; ++b, p += block_bytes, dst += block_size)
            S::decode(p, dst);
    }

private:
    const std::byte* block_data(size_type r, size_type b) const noexcept {
        return data_ + (r * blocks_per_row_ + b) * block_bytes;
    }

    size_type rows_;
    size_type cols_;
    size_type blocks_per_row_;
    const std::byte* data_;
};

} // namespace mtl::mat

namespace mtl {

/// Customization point for a scheme's fused mat-vec kernel.
///
/// The primary template is empty, and then mult() decodes row by row. A
/// scheme that has a faster route (integer dots on the encoded blocks, say,
/// with simd::reduce_dot_block_scaled) specializes it:
///
///     template <> struct mtl::block_kernels<my_scheme> {
///         // y = A x. `scratch` holds at least scratch_bytes(A.num_cols())
///         // bytes, aligned to scratch_alignment.
///         static void gemv(const mtl::mat::block_matrix_view<my_scheme>& A,
///                          const float* x, float* y, std::span<std::byte> scratch);
///
///         // Optional: working memory per call (default 0) and its alignment
///         // (default alignof(std::max_align_t), a power of two).
///         static std::size_t scratch_bytes(std::size_t cols);
///         static constexpr std::size_t scratch_alignment = 64;
///     };
///
/// The scratch is for what a fused kernel builds from x before it touches A,
/// typically x quantized per block to the integer format the dots consume. It
/// is the caller's memory, not the kernel's: mult(A, x, y, scratch) passes it
/// through, so a runtime that sizes its buffers up front never allocates in
/// the mat-vec. mult(A, x, y) allocates it for the one call.
///
/// mult() calls the kernel for float input and output vectors with contiguous
/// storage and no explicit Accumulator. The kernel owns its accuracy contract.
/// The decode-then-dot path stays the reference it is measured against.
template <typename S>
struct block_kernels {};

/// Does the scheme supply a fused gemv?
template <typename S>
concept BlockGemvScheme = BlockScheme<S> &&
    requires(const mat::block_matrix_view<S>& A, const float* x, float* y,
             std::span<std::byte> scratch) {
        block_kernels<S>::gemv(A, x, y, scratch);
    };

namespace detail {

// Named concepts and a plain function rather than requires-expressions in a
// lambda: Apple Clang 15 crashes on the latter.
template <typename K>
concept declares_scratch_bytes = requires(std::size_t cols) {
    { K::scratch_bytes(cols) } -> std::convertible_to<std::size_t>;
};

template <typename K>
concept declares_scratch_alignment = requires {
    { K::scratch_alignment } -> std::convertible_to<std::size_t>;
};

template <typename S>
constexpr std::size_t block_scratch_alignment() noexcept {
    if constexpr (BlockGemvScheme<S> && declares_scratch_alignment<block_kernels<S>>) {
        constexpr std::size_t align = block_kernels<S>::scratch_alignment;
        static_assert(align != 0 && (align & (align - 1)) == 0,
                      "block_kernels<S>::scratch_alignment must be a power of two");
        return align;
    } else {
        return alignof(std::max_align_t);
    }
}

} // namespace detail

/// Scratch bytes mult() needs for a block matrix of scheme S with `cols`
/// columns: the fused kernel's scratch_bytes(cols), or 0 when the scheme has
/// no fused kernel or its kernel declares none.
template <BlockScheme S>
std::size_t block_gemv_scratch_bytes(std::size_t cols) {
    if constexpr (BlockGemvScheme<S> && detail::declares_scratch_bytes<block_kernels<S>>)
        return block_kernels<S>::scratch_bytes(cols);
    else
        return 0;
}

/// The alignment that scratch must have: the fused kernel's scratch_alignment,
/// which must be a power of two, or alignof(std::max_align_t).
template <BlockScheme S>
inline constexpr std::size_t block_gemv_scratch_alignment = detail::block_scratch_alignment<S>();

namespace detail {

/// y[rb, re) = A x for a block matrix: decode one row into `buf`, then dot it
/// with x. The row is decoded once whatever x is; the dot is SIMD when x is
/// contiguous float, and the generic fused loop otherwise.
template <typename Accumulator, typename S, typename VIn, typename VOut>
void block_mult_rows(const mat::block_matrix_view<S>& A, const VIn& x, VOut& y,
                     std::size_t rb, std::size_t re, std::vector<float>& buf) {
    using Result = typename VOut::value_type;
    const std::size_t n = A.num_cols();
    buf.resize(n);
    for (std::size_t r = rb; r < re; ++r) {
        A.decode_row(r, buf.data());
        if constexpr (!std::is_void_v<Accumulator>) {
            using Value = std::common_type_t<float, typename VIn::value_type>;
            using AT = math::accumulator_traits<Accumulator, Value>;
            Accumulator acc{};
            AT::clear(acc);
            for (std::size_t c = 0; c < n; ++c)
                AT::add_product(acc, static_cast<Value>(buf[c]),
                                static_cast<Value>(x(static_cast<typename VIn::size_type>(c))));
            y(static_cast<typename VOut::size_type>(r)) = AT::template value<Result>(acc);
        } else if constexpr (interface::SimdDenseVector<VIn> &&
                             std::is_same_v<typename VIn::value_type, float>) {
            y(static_cast<typename VOut::size_type>(r)) =
                static_cast<Result>(simd::reduce_dot<float>(buf.data(), x.data(), n));
        } else {
            auto acc = math::zero<Result>();
            for (std::size_t c = 0; c < n; ++c)
                acc = generic_fma<Result>(acc, buf[c],
                                          x(static_cast<typename VIn::size_type>(c)));
            y(static_cast<typename VOut::size_type>(r)) = acc;
        }
    }
}

/// Does mult(A, x, y) on these types run the scheme's fused kernel?
template <typename Accumulator, typename S, typename VIn, typename VOut>
inline constexpr bool block_fused_gemv_v =
    std::is_void_v<Accumulator> && BlockGemvScheme<S> &&
    interface::SimdDenseVector<VIn> && interface::SimdDenseVector<VOut> &&
    std::is_same_v<typename VIn::value_type, float> &&
    std::is_same_v<typename VOut::value_type, float>;

/// y = A x by decode-then-dot, rows partitioned across the thread pool.
template <typename Accumulator, typename S, typename VIn, typename VOut>
void block_mult_decode(const mat::block_matrix_view<S>& A, const VIn& x, VOut& y) {
    const std::size_t m = A.num_rows();
    // A row costs its decode plus cols multiply-adds; grain on the latter,
    // as the dense mat-vec does.
    const std::size_t grain = interface::row_grain<float>(A.num_cols());
    thread_pool& pool = thread_pool::instance();
    if (pool.size() > 1 && m >= grain * 2) {
        pool.parallel_for(m, grain, [&](std::size_t b, std::size_t e) {
            std::vector<float> buf;
            block_mult_rows<Accumulator>(A, x, y, b, e, buf);
        });
    } else {
        std::vector<float> buf;
        block_mult_rows<Accumulator>(A, x, y, 0, m, buf);
    }
}

} // namespace detail

/// y = A x for a block matrix, into a pre-allocated y, with the fused kernel's
/// scratch supplied by the caller.
///
/// With the default Accumulator, float contiguous x and y, and a scheme that
/// specializes block_kernels, the scheme's fused kernel runs on `scratch`,
/// which must hold block_gemv_scratch_bytes<S>(A.num_cols()) bytes aligned to
/// block_gemv_scratch_alignment<S>; otherwise this throws
/// std::invalid_argument. On every other path `scratch` is unused: each row is
/// decoded once and dotted with x, with an explicit Accumulator if one is
/// given. Rows are partitioned across the thread pool like the generic dense
/// mat-vec. Each y(r) is computed from row r alone, so the result does not
/// depend on the thread count.
template <typename Accumulator = void, BlockScheme S, Vector VIn, Vector VOut>
void mult(const mat::block_matrix_view<S>& A, const VIn& x, VOut& y,
          std::span<std::byte> scratch) {
    assert(A.num_cols() == static_cast<std::size_t>(x.size()));
    assert(A.num_rows() == static_cast<std::size_t>(y.size()));
    if constexpr (detail::block_fused_gemv_v<Accumulator, S, VIn, VOut>) {
        const std::size_t need = block_gemv_scratch_bytes<S>(A.num_cols());
        constexpr std::size_t align = block_gemv_scratch_alignment<S>;
        if (scratch.size() < need)
            throw std::invalid_argument("mult: scratch of " + std::to_string(scratch.size()) +
                                        " bytes, the block kernel needs " + std::to_string(need));
        if (reinterpret_cast<std::uintptr_t>(scratch.data()) % align != 0)
            throw std::invalid_argument("mult: scratch is not aligned to " + std::to_string(align) +
                                        " bytes");
        block_kernels<S>::gemv(A, x.data(), y.data(), scratch);
    } else {
        detail::block_mult_decode<Accumulator>(A, x, y);
    }
}

/// y = A x for a block matrix, into a pre-allocated y. As the overload above,
/// with any scratch the fused kernel needs allocated for this call.
template <typename Accumulator = void, BlockScheme S, Vector VIn, Vector VOut>
void mult(const mat::block_matrix_view<S>& A, const VIn& x, VOut& y) {
    if constexpr (detail::block_fused_gemv_v<Accumulator, S, VIn, VOut>) {
        const std::size_t need = block_gemv_scratch_bytes<S>(A.num_cols());
        constexpr std::size_t align = block_gemv_scratch_alignment<S>;
        if (need == 0) {
            mult(A, x, y, std::span<std::byte>{});
            return;
        }
        if (need > std::numeric_limits<std::size_t>::max() - (align - 1))
            throw std::length_error("mult: block kernel scratch of " + std::to_string(need) +
                                    " bytes overflows size_t once aligned");
        std::vector<std::byte> raw(need + align - 1);
        void* p = raw.data();
        std::size_t space = raw.size();
        std::align(align, need, p, space);
        mult(A, x, y, std::span<std::byte>(static_cast<std::byte*>(p), need));
    } else {
        mult<Accumulator>(A, x, y, std::span<std::byte>{});
    }
}

/// Decode a block matrix into a pre-allocated destination, a row at a time.
template <BlockScheme S, Matrix MDst>
void convert(const mat::block_matrix_view<S>& src, MDst& dst) {
    assert(src.num_rows() == dst.num_rows());
    assert(src.num_cols() == dst.num_cols());
    using D = typename MDst::value_type;
    std::vector<float> buf(src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r) {
        src.decode_row(r, buf.data());
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(static_cast<typename MDst::size_type>(r),
                static_cast<typename MDst::size_type>(c)) = static_cast<D>(buf[c]);
    }
}

/// Decode a block matrix into a new dense2D<DstValue>.
template <typename DstValue, BlockScheme S>
mat::dense2D<DstValue> convert(const mat::block_matrix_view<S>& src) {
    mat::dense2D<DstValue> dst(src.num_rows(), src.num_cols());
    convert(src, dst);
    return dst;
}

namespace mat {

/// A * x into a new dense_vector, through mult(). Found by ADL, and more
/// specialized than the generic Matrix * Vector operator, which would decode
/// one block per element.
template <BlockScheme S, Vector V>
    requires(!Matrix<V>)
auto operator*(const block_matrix_view<S>& A, const V& x) {
    assert(A.num_cols() == static_cast<std::size_t>(x.size()));
    using result_t = std::common_type_t<float, typename V::value_type>;
    vec::dense_vector<result_t> y(A.num_rows());
    mtl::mult(A, x, y);
    return y;
}

} // namespace mat

} // namespace mtl

namespace mtl::traits {
template <typename S>
struct category<mat::block_matrix_view<S>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename S>
struct ashape<::mtl::mat::block_matrix_view<S>> { using type = mat<float>; };
} // namespace mtl::ashape

namespace mtl { using mat::block_matrix_view; }
