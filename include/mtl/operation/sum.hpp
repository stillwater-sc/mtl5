#pragma once
// MTL5 -- Sum of all elements in a collection
//
// ACCUMULATOR (#585). sum<Accumulator>(c) routes the sum through
// math::accumulator_traits and rounds it once. A sum is not a sum of products,
// and the accumulator contract has no plain `add`; each element enters as
// add_product(acc, c[i], 1). That is exact for every configuration -- x * 1 is
// x in IEEE arithmetic, posits and LNS alike -- so the policy path computes
// exactly acc += c[i], and a plain element-type accumulator gives the void
// bits. The cost is one multiply by one per element, on the non-void path
// only; a dedicated add primitive would remove it if that ever matters.
#include <type_traits>
#include <mtl/concepts/collection.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace mtl {

/// Returns the sum of all elements: sum(c[i])
///
/// `Accumulator` selects how the sum accumulates (see the header note and
/// math/accumulator_traits.hpp); the default `void` is the plain value_type sum.
template <typename Accumulator = void, Collection C>
auto sum(const C& c) {
    using T = typename C::value_type;
    if constexpr (!std::is_void_v<Accumulator>) {
        using AT = math::accumulator_traits<Accumulator, T>;
        Accumulator acc;
        AT::clear(acc);
        for (auto it = c.begin(); it != c.end(); ++it)
            AT::add_product(acc, static_cast<T>(*it), math::one<T>());
        return AT::template value<T>(acc);
    } else {
        auto acc = math::zero<T>();
        for (auto it = c.begin(); it != c.end(); ++it) {
            acc += *it;
        }
        return acc;
    }
}

} // namespace mtl
