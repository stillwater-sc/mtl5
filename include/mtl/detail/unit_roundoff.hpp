#pragma once
// MTL5 -- Machine epsilon and unit roundoff of a scalar type
//
// Algorithms that adapt to the working precision read it from
// std::numeric_limits, which every MTL5 element type specializes (#838).
#include <limits>
#include <mtl/math/identity.hpp>

namespace mtl::detail {

/// The machine epsilon of Real: the spacing of Real's values at 1, from
/// std::numeric_limits. Every MTL5 element type has a specialization (#838);
/// a measured fallback for types without one was removed with that rule.
template <typename Real>
    requires std::numeric_limits<Real>::is_specialized
Real machine_epsilon() {
    return std::numeric_limits<Real>::epsilon();
}

/// The unit roundoff u of Real, the bound on the relative error of one
/// rounding: fl(x) = x (1 + d), |d| <= u. Under round-to-nearest this is half
/// the machine epsilon, which is what error analyses (and tables derived from
/// them, such as the Pade degree tables of expm) mean by u: 2^-53 for double.
template <typename Real>
Real unit_roundoff() {
    const Real one = math::one<Real>();
    return machine_epsilon<Real>() / (one + one);
}

} // namespace mtl::detail
