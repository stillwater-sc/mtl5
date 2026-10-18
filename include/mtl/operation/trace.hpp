#pragma once
// MTL5 -- Matrix trace: sum of diagonal elements
//
// ACCUMULATOR (#585). trace<Accumulator>(A) routes the diagonal sum through
// math::accumulator_traits and rounds it once, each A(i,i) entering as
// add_product(acc, A(i,i), 1) -- exact for every configuration; see the note
// in operation/sum.hpp.
#include <algorithm>
#include <type_traits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace mtl {

/// trace(A) = sum(A(i,i)) for i = 0..min(nrows,ncols)-1
///
/// `Accumulator` selects how the sum accumulates; the default `void` is the
/// plain value_type sum.
template <typename Accumulator = void, Matrix M>
auto trace(const M& A) {
    using value_type = typename M::value_type;
    auto n = std::min(A.num_rows(), A.num_cols());
    if constexpr (!std::is_void_v<Accumulator>) {
        using AT = math::accumulator_traits<Accumulator, value_type>;
        Accumulator acc;
        AT::clear(acc);
        for (typename M::size_type i = 0; i < n; ++i)
            AT::add_product(acc, static_cast<value_type>(A(i, i)), math::one<value_type>());
        return AT::template value<value_type>(acc);
    } else {
        auto acc = math::zero<value_type>();
        for (typename M::size_type i = 0; i < n; ++i)
            acc += A(i, i);
        return acc;
    }
}

} // namespace mtl
