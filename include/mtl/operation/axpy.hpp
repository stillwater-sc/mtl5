#pragma once
// MTL5 -- axpy: y <- alpha*x + y  (BLAS level-1)
// Native SIMD path for contiguous, same-type real float/double vectors
// (mtl::simd::axpy, #86); BLAS dispatch when available; generic scalar
// fallback otherwise.
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/simd/algorithm.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// y[i] += alpha * x[i]
template <Scalar S, Vector VX, Vector VY>
void axpy(const S& alpha, const VX& x, VY& y) {
    assert(x.size() == y.size());
    if constexpr (interface::BlasDenseVector<VX> && interface::BlasDenseVector<VY> &&
                  std::is_same_v<typename VX::value_type, typename VY::value_type>) {
        using T = typename VY::value_type;
        const std::size_t n = y.size();
#ifdef MTL5_HAS_BLAS
        if (n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            interface::blas::axpy(static_cast<int>(n), static_cast<T>(alpha),
                                  x.data(), 1, y.data(), 1);
            return;
        }
#endif
        // Element-wise: chunk the range across the pool (bit-identical, each
        // element in exactly one chunk). Serial by default (MTL5_NUM_THREADS=1).
        const T a = static_cast<T>(alpha);
        const T* xp = x.data();
        T* yp = y.data();
        detail::thread_pool::instance().parallel_for(n, /*grain=*/std::size_t{65536},
            [&](std::size_t b, std::size_t e) { simd::axpy<T>(a, xp + b, yp + b, e - b); });
    } else {
        for (typename VY::size_type i = 0; i < y.size(); ++i) {
            y(i) += alpha * x(i);
        }
    }
}

} // namespace mtl
