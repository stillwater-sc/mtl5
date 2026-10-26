#pragma once
// MTL5 -- Machine epsilon and unit roundoff of a scalar type
//
// Algorithms that adapt to the working precision need the precision of a type
// that may not have a std::numeric_limits specialization. For such a type
// numeric_limits<T>::epsilon() is a value-initialized T, i.e. zero, and a
// zero epsilon silently turns a tolerance or a precision-dependent choice into
// its exact-arithmetic limit. These helpers measure it instead.
#include <limits>
#include <mtl/math/identity.hpp>

namespace mtl::detail {

/// The machine epsilon of Real: the spacing of Real's values at 1. Taken from
/// std::numeric_limits when Real has a specialization; otherwise measured by
/// halving until 1 + eps/2 rounds to 1. The halving is capped so an
/// exact-arithmetic type cannot loop forever.
template <typename Real>
Real machine_epsilon() {
    if constexpr (std::numeric_limits<Real>::is_specialized) {
        return std::numeric_limits<Real>::epsilon();
    } else {
        const Real one = math::one<Real>();
        const Real two = one + one;
        Real eps = one;
        for (int i = 0; i < 1024 && one + eps / two > one; ++i)
            eps = eps / two;
        return eps;
    }
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
