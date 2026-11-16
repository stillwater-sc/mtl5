#pragma once
// MTL5 -- Maximum element in a collection
#include <concepts>
#include <stdexcept>
#include <mtl/concepts/collection.hpp>
#include <mtl/detail/exceptions.hpp>

namespace mtl {

/// Returns the maximum element value. A NaN element makes the result NaN,
/// wherever it is: a plain `>` comparison skips a NaN, so the result
/// depended on its position, as infinity_norm's did before #690. An empty
/// collection throws std::invalid_argument; it has no maximum.
///
/// Seeded from the first element (issue 887): the seed was
/// numeric_limits<T>::lowest(), so max({-inf}) returned the finite
/// -1.79769e+308, an empty collection returned that sentinel, and a type
/// without a numeric_limits specialization got a default value.
template <Collection C>
auto max(const C& c) {
    auto it = c.begin();
    const auto end = c.end();
    if (it == end)
        ::mtl::detail::throw_or_abort(std::invalid_argument("mtl::max: empty collection"));
    using T = std::remove_cvref_t<decltype(*it)>;
    T result = *it;
    for (++it; it != end; ++it) {
        const T& a = *it;
        if constexpr (requires { { a != a } -> std::convertible_to<bool>; }) {
            if (result != result) break;             // already NaN: it stays
            if (a != a || a > result) result = a;
        } else {
            if (a > result) result = a;
        }
    }
    return result;
}

} // namespace mtl
