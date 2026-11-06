#pragma once
// MTL5 -- same_object: the address test behind mult's no-aliasing assertions
#include <memory>

namespace mtl::detail {

/// True when a and b are the same object (#696). An operation that overwrites
/// its output while still reading its inputs, such as mult, asserts that the
/// output is not one of them. The test compares addresses only, so it does not
/// see a view of the output, such as mult(trans(C), B, C).
template <typename T, typename U>
constexpr bool same_object(const T& a, const U& b) noexcept {
    return static_cast<const void*>(std::addressof(a)) == static_cast<const void*>(std::addressof(b));
}

} // namespace mtl::detail
