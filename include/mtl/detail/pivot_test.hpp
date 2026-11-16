#pragma once
// MTL5 -- the pivot test of the dense factorizations (#854)
//
// A factorization stops at a pivot it cannot divide by and reports its 1-based
// index. Testing `pivot == 0` lets a NaN through: NaN == 0 is false, so
// [[NaN, 1], [1, 2]] factored "successfully" with NaN in D and L. !(|p| > 0)
// is true for 0 and for NaN alike, as cholesky_factor's !(diag > 0) has been
// since #556. An infinite pivot still passes: it is a valid, if extreme, value.
#include <cmath>

namespace mtl::detail {

/// True when p is zero or NaN: not a pivot a factorization can divide by.
/// abs is found by ADL, so a custom number type's own is used.
template <typename T>
constexpr bool pivot_unusable(const T& p) {
    using std::abs;
    const auto m = abs(p);
    return !(m > decltype(m)(0));
}

} // namespace mtl::detail
