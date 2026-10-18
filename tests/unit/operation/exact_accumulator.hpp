#pragma once
// The exact accumulator as the tests name it (#586), and the reference helpers
// they measure with. The accumulator itself is mtl::math::exact_accumulator
// (include/mtl/math/exact_accumulator.hpp, #611): configuration 3 without
// Universal, and the REFERENCE as well -- `err` below measures a computed
// result's error exactly, as |exact - computed| rounded once to double.
#include <cmath>
#include <limits>
#include <type_traits>

#include <mtl/math/exact_accumulator.hpp>

namespace exact_test {

template <typename T>
using exact_acc = mtl::math::exact_accumulator<T>;

/// A cleared exact accumulator.
template <typename T>
exact_acc<T> zero() {
    exact_acc<T> a;
    accumulator_clear(a);
    return a;
}

/// |exact - computed|, rounded once to double. A non-finite `computed` has a
/// non-finite error (inf, or NaN, which fails every `<=` it meets).
template <typename T, typename C>
double err(const exact_acc<T>& exact, C computed) {
    if (!std::isfinite(computed))
        return std::isnan(computed) ? std::numeric_limits<double>::quiet_NaN()
                                    : std::numeric_limits<double>::infinity();
    exact_acc<T> d = exact;
    accumulator_add_product(d, static_cast<T>(-computed), T(1));
    return std::fabs(accumulator_value(d, std::type_identity<double>{}));
}

} // namespace exact_test
