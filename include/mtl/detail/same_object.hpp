#pragma once
// MTL5 -- same_object and storage_overlaps: the address tests behind mult's
// aliasing guards
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace mtl::detail {

/// True when a and b are the same object (#696). An operation that overwrites
/// its output while still reading its inputs, such as mult, asserts that the
/// output is not one of them. The test compares addresses only, so it does not
/// see a view of the output, such as mult(trans(C), B, C).
template <typename T, typename U>
constexpr bool same_object(const T& a, const U& b) noexcept {
    return static_cast<const void*>(std::addressof(a)) == static_cast<const void*>(std::addressof(b));
}

/// A collection whose elements are laid out from data(): dense vectors and
/// matrices, strided references, and views over a caller's buffer.
template <typename T>
concept contiguous_storage = requires(const T& t) {
    t.data();
    t.size();
};

/// The bytes from t's first element to its last.
template <contiguous_storage T>
std::pair<const unsigned char*, const unsigned char*> storage_bytes(const T& t) {
    using elem = std::remove_cvref_t<decltype(*t.data())>;
    const auto* lo = reinterpret_cast<const unsigned char*>(t.data());
    const auto n = static_cast<std::size_t>(t.size());
    if (lo == nullptr || n == 0) return {nullptr, nullptr};
    std::size_t span = n;                                  // elements from the first to past the last
    if constexpr (requires { t.stride(); }) {
        const auto s = static_cast<std::size_t>(t.stride());
        span = (n - 1) * s + 1;
    }
    return {lo, lo + span * sizeof(elem)};
}

/// The bytes of a std::vector-like array's elements; empty when it is.
template <typename A>
std::pair<const unsigned char*, const unsigned char*> array_bytes(const A& v) {
    using elem = std::remove_cvref_t<decltype(*v.data())>;
    const auto* lo = reinterpret_cast<const unsigned char*>(v.data());
    if (lo == nullptr || v.size() == 0) return {nullptr, nullptr};
    return {lo, lo + static_cast<std::size_t>(v.size()) * sizeof(elem)};
}

/// A compressed matrix's index arrays (compressed2D's ref_minor() and
/// ref_major()): a writable output over them would corrupt the indices the
/// kernel is still reading.
template <typename T>
concept minor_array_storage = requires(const T& t) {
    t.ref_minor().data();
    t.ref_minor().size();
};
template <typename T>
concept major_array_storage = requires(const T& t) {
    t.ref_major().data();
    t.ref_major().size();
};

/// A sparse matrix whose values are one array (compressed2D's and
/// ell_matrix's ref_data()).
template <typename T>
concept value_array_storage = requires(const T& t) {
    t.ref_data().data();
    t.ref_data().size();
};

/// A matrix whose entries are one array of (row, col, value) triplets
/// (coordinate2D's ref_entries()).
template <typename T>
concept entry_array_storage = requires(const T& t) {
    t.ref_entries().data();
    t.ref_entries().size();
};

/// A matrix made of separately stored blocks (block_diagonal2D): its storage
/// is every block's.
template <typename T>
concept block_list_storage = requires(const T& t, std::size_t k) {
    t.num_blocks();
    t.block(k).data();
};

/// A matrix over encoded bytes (block_matrix_view's bytes()).
template <typename T>
concept encoded_byte_storage = requires(const T& t) {
    t.bytes().data();
    t.bytes().size();
};

/// A view over another collection (transposed_view, the triangular and banded
/// views, ...): its storage is its base's.
template <typename T>
concept view_of_storage = requires(const T& t) { t.base(); };

/// Calls f(lo, hi) for each byte range a collection's elements live in,
/// looking through views: a contiguous collection's elements; a compressed
/// or ELL matrix's value array; a coordinate matrix's triplets; each block of
/// a block-diagonal matrix; a block matrix view's encoded bytes; a view's
/// base. Nothing for a type with none of these (an expression).
template <typename T, typename F>
void for_each_storage_range(const T& t, F&& f) {
    auto visit = [&f](std::pair<const unsigned char*, const unsigned char*> r) {
        if (r.first != nullptr) f(r.first, r.second);
    };
    if constexpr (contiguous_storage<T>) {
        visit(storage_bytes(t));
    } else if constexpr (value_array_storage<T>) {
        visit(array_bytes(t.ref_data()));
        if constexpr (minor_array_storage<T>) visit(array_bytes(t.ref_minor()));
        if constexpr (major_array_storage<T>) visit(array_bytes(t.ref_major()));
    } else if constexpr (entry_array_storage<T>) {
        visit(array_bytes(t.ref_entries()));
    } else if constexpr (block_list_storage<T>) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(t.num_blocks()); ++k)
            for_each_storage_range(t.block(k), f);
    } else if constexpr (encoded_byte_storage<T>) {
        visit(array_bytes(t.bytes()));
    } else if constexpr (view_of_storage<T>) {
        for_each_storage_range(t.base(), f);
    } else {
        static_cast<void>(t);
    }
}

/// Whether a and b may share storage (#840, #855): the same object, or element
/// storage whose byte ranges overlap -- a vector over part of another's buffer,
/// as zero-copy views make easy; a vector over a sparse matrix's values or
/// triplets, a block's elements or a block view's bytes; the matrix behind a
/// transposed view. A type with no visible storage (an expression) is seen
/// only as the same object.
template <typename T, typename U>
bool storage_overlaps(const T& a, const U& b) noexcept {
    if (same_object(a, b)) return true;
    const std::less<const unsigned char*> lt;
    bool overlap = false;
    for_each_storage_range(a, [&](const unsigned char* a0, const unsigned char* a1) {
        if (overlap) return;
        for_each_storage_range(b, [&](const unsigned char* b0, const unsigned char* b1) {
            if (!overlap && lt(a0, b1) && lt(b0, a1)) overlap = true;
        });
    });
    return overlap;
}

} // namespace mtl::detail
