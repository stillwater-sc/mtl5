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

/// A sparse matrix whose values are one array (compressed2D's ref_data()).
template <typename T>
concept value_array_storage = requires(const T& t) {
    t.ref_data().data();
    t.ref_data().size();
};

/// A view over another collection (transposed_view, the triangular and banded
/// views, ...): its storage is its base's.
template <typename T>
concept view_of_storage = requires(const T& t) { t.base(); };

/// The bytes a collection's elements live in, looking through views: a
/// contiguous collection's elements, a compressed matrix's value array, a
/// view's base. Empty when the type has none of these (an expression).
template <typename T>
std::pair<const unsigned char*, const unsigned char*> storage_range(const T& t) {
    if constexpr (contiguous_storage<T>) {
        return storage_bytes(t);
    } else if constexpr (value_array_storage<T>) {
        const auto& v = t.ref_data();
        using elem = std::remove_cvref_t<decltype(*v.data())>;
        const auto* lo = reinterpret_cast<const unsigned char*>(v.data());
        if (lo == nullptr || v.size() == 0) return {nullptr, nullptr};
        return {lo, lo + v.size() * sizeof(elem)};
    } else if constexpr (view_of_storage<T>) {
        return storage_range(t.base());
    } else {
        static_cast<void>(t);
        return {nullptr, nullptr};
    }
}

/// Whether a and b may share storage (#840): the same object, or element
/// storage whose byte ranges overlap -- a vector over part of another's buffer,
/// as zero-copy views make easy; a vector over a compressed matrix's values;
/// the matrix behind a transposed view. A type with no visible storage (an
/// expression) is seen only as the same object.
template <typename T, typename U>
bool storage_overlaps(const T& a, const U& b) noexcept {
    if (same_object(a, b)) return true;
    const auto [a0, a1] = storage_range(a);
    const auto [b0, b1] = storage_range(b);
    if (a0 == nullptr || b0 == nullptr) return false;
    const std::less<const unsigned char*> lt;
    return lt(a0, b1) && lt(b0, a1);
}

} // namespace mtl::detail
