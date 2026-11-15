#pragma once
// MTL5 -- Can an expression be assigned to a matrix in place? (#824)
//
// dense2D::operator=(expr) writes target(r, c) = expr(r, c) one element at a
// time, so an expression that reads the target anywhere other than at (r, c)
// sees elements already overwritten: A = A + trans(A) read A(1,0) after
// writing it, and gave A(1,0) = 10 for {{1,2,3},{4,5,6},{7,8,9}} where 6 is
// right. Under threads it is a data race. Those assignments go through a
// temporary instead; everything else stays in place.
//
// The walk over the expression tree:
//   - The element-wise nodes (+, -, negation, scaling) read their operands at
//     (r, c) only, so they are safe when their operands are.
//   - The target itself as a leaf is read at (r, c) only: safe. A = A + B and
//     A = 2 * A stay in place.
//   - Any other leaf that stores its elements contiguously (data()) is safe
//     unless that storage overlaps the target's: a dense2D over the target's
//     buffer through an external pointer is not.
//   - A view (it has base(): trans, hermitian, upper, banded, ...) or a lazy
//     product reads elements other than (r, c), so it is unsafe as soon as it
//     reaches the target's storage at all.
//   - A leaf whose storage detail::for_each_storage_range can see -- a
//     compressed2D's or ell_matrix's arrays, a coordinate2D's triplets, every
//     block of a block_diagonal2D, a block_matrix_view's bytes -- reaches the
//     target only if that storage overlaps the target's (#857). It used to be
//     assumed to, so A = S or A += S1 + S2 built a dense rows*cols temporary
//     and evaluated every element twice for sparse operands.
//   - Any other matrix is assumed to reach the target: a generated or
//     user-defined matrix may hold a reference to it that nothing here sees.
// The answer is conservative: a strict-lower view of the target, say, reads
// only (r, c) and would have been safe, but takes the temporary.

#include <cstddef>
#include <type_traits>

#include <mtl/mat/expr/mat_mat_op_expr.hpp>
#include <mtl/mat/expr/mat_mat_times_expr.hpp>
#include <mtl/mat/expr/mat_negate_expr.hpp>
#include <mtl/mat/expr/mat_scal_op_expr.hpp>
#include <mtl/detail/same_object.hpp>

namespace mtl::detail {

/// The storage an assignment writes: the target object and its elements' bytes.
struct alias_target {
    const void* self;
    const unsigned char* lo;
    const unsigned char* hi;
};

template <typename T> struct alias_elementwise_binary : std::false_type {};
template <typename E1, typename E2, template <typename, typename> class SF>
struct alias_elementwise_binary<::mtl::mat::expr::mat_mat_op_expr<E1, E2, SF>> : std::true_type {};

template <typename T> struct alias_elementwise_unary : std::false_type {};
template <typename E>
struct alias_elementwise_unary<::mtl::mat::expr::mat_negate_expr<E>> : std::true_type {};
template <typename S, typename M, template <typename, typename> class SF>
struct alias_elementwise_unary<::mtl::mat::expr::mat_scal_op_expr<S, M, SF>> : std::true_type {};
template <typename M, typename S, template <typename, typename> class SF>
struct alias_elementwise_unary<::mtl::mat::expr::mat_rscal_op_expr<M, S, SF>> : std::true_type {};

template <typename T> struct alias_product : std::false_type {};
template <typename E1, typename E2>
struct alias_product<::mtl::mat::expr::mat_mat_times_expr<E1, E2>> : std::true_type {};

template <typename T>
concept alias_view = requires(const T& t) { t.base(); };

template <typename T>
concept alias_contiguous_leaf = requires(const T& t) {
    t.data();
    t.num_rows();
    t.num_cols();
};

/// A leaf whose element storage for_each_storage_range can list.
template <typename T>
concept alias_known_storage = value_array_storage<T> || entry_array_storage<T> ||
                              block_list_storage<T> || encoded_byte_storage<T>;

/// Whether a known-storage leaf is the target or any of its storage overlaps
/// the target's elements.
template <typename T>
bool alias_storage_overlaps(const T& m, const alias_target& t) {
    if (static_cast<const void*>(&m) == t.self) return true;
    bool hit = false;
    for_each_storage_range(m, [&](const unsigned char* lo, const unsigned char* hi) {
        if (lo < t.hi && t.lo < hi) hit = true;
    });
    return hit;
}

/// Whether the leaf's contiguous storage overlaps the target's. Its span is
/// rows * cols elements from data(): the only leaf with data() and a leading
/// dimension is dense2D, whose leading dimension is always its column count
/// (row-major) or row count (column-major), with no padding, so there is no
/// tail beyond rows * cols for an overlap to hide in (#857).
template <typename T>
bool alias_overlaps(const T& m, const alias_target& t) {
    using elem = std::remove_cvref_t<decltype(*m.data())>;
    const auto* lo = reinterpret_cast<const unsigned char*>(m.data());
    const auto n = static_cast<std::size_t>(m.num_rows()) * static_cast<std::size_t>(m.num_cols());
    const auto* hi = lo + n * sizeof(elem);
    return lo != nullptr && lo < t.hi && t.lo < hi;
}

/// Whether evaluating e reads the target's storage at all.
template <typename E>
bool alias_reaches_target(const E& e, const alias_target& t) {
    if constexpr (alias_elementwise_binary<E>::value || alias_product<E>::value) {
        return alias_reaches_target(e.first(), t) || alias_reaches_target(e.second(), t);
    } else if constexpr (alias_elementwise_unary<E>::value) {
        return alias_reaches_target(e.operand(), t);
    } else if constexpr (alias_view<E>) {
        return alias_reaches_target(e.base(), t);
    } else if constexpr (alias_contiguous_leaf<E>) {
        return static_cast<const void*>(&e) == t.self || alias_overlaps(e, t);
    } else if constexpr (alias_known_storage<E>) {
        return alias_storage_overlaps(e, t);
    } else {
        static_cast<void>(e);
        static_cast<void>(t);
        return true;   // may hold the target (see the top of this file)
    }
}

/// Whether evaluating e element by element into the target can read an
/// element of the target other than the one being written.
template <typename E>
bool alias_reads_target_elsewhere(const E& e, const alias_target& t) {
    if constexpr (alias_elementwise_binary<E>::value) {
        return alias_reads_target_elsewhere(e.first(), t) || alias_reads_target_elsewhere(e.second(), t);
    } else if constexpr (alias_elementwise_unary<E>::value) {
        return alias_reads_target_elsewhere(e.operand(), t);
    } else if constexpr (alias_product<E>::value || alias_view<E>) {
        return alias_reaches_target(e, t);
    } else if constexpr (alias_contiguous_leaf<E>) {
        return static_cast<const void*>(&e) != t.self && alias_overlaps(e, t);
    } else if constexpr (alias_known_storage<E>) {
        // Not the target itself (it is dense), so any overlap reads it elsewhere.
        return alias_storage_overlaps(e, t);
    } else {
        static_cast<void>(e);
        static_cast<void>(t);
        return true;   // may hold the target (see the top of this file)
    }
}

} // namespace mtl::detail
