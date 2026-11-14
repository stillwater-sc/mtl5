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

/// Whether a and b may share storage (#840). Two collections with contiguous
/// storage overlap when their byte ranges do: a vector over part of another's
/// buffer, as zero-copy views make easy, is caught, not just the same object.
/// Otherwise -- an expression, a sparse vector -- only the same object is seen.
template <typename T, typename U>
bool storage_overlaps(const T& a, const U& b) noexcept {
    if (same_object(a, b)) return true;
    if constexpr (contiguous_storage<T> && contiguous_storage<U>) {
        const auto [a0, a1] = storage_bytes(a);
        const auto [b0, b1] = storage_bytes(b);
        if (a0 == nullptr || b0 == nullptr) return false;
        const std::less<const unsigned char*> lt;
        return lt(a0, b1) && lt(b0, a1);
    } else {
        return false;
    }
}

} // namespace mtl::detail
