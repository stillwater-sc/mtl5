#pragma once
// MTL5 -- Scale collection by a scalar factor
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/collection.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/simd/algorithm.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// In-place scale: c[i] *= alpha
template <Scalar S, MutableCollection C>
void scale(const S& alpha, C& c) {
    // Native SIMD / BLAS path for contiguous real float/double vectors;
    // generic iterator loop for everything else (matrices, strided, complex).
    if constexpr (interface::BlasDenseVector<C>) {
        using T = typename C::value_type;
        const std::size_t n = c.size();
#ifdef MTL5_HAS_BLAS
        if (n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            interface::blas::scal(static_cast<int>(n), static_cast<T>(alpha), c.data(), 1);
            return;
        }
#endif
        // Element-wise: chunk the range across the pool (bit-identical).
        const T a = static_cast<T>(alpha);
        T* cp = c.data();
        detail::thread_pool::instance().parallel_for(n, /*grain=*/std::size_t{65536},
            [&](std::size_t b, std::size_t e) { simd::scal<T>(a, cp + b, e - b); });
    } else {
        for (auto it = c.begin(); it != c.end(); ++it) {
            *it *= alpha;
        }
    }
}

/// Returns a scaled copy of a vector
template <Scalar S, Collection C>
auto scaled(const S& alpha, const C& c) {
    auto result = c;  // copy
    scale(alpha, result);
    return result;
}

} // namespace mtl
