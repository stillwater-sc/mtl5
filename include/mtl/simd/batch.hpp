#pragma once
// MTL5 -- portable SIMD abstraction (Phase 0 of the native-BLAS performance epic, #82/#83)
//
// `mtl::simd::batch<T>` is a value-type wrapper over one SIMD register with a
// COMPILE-TIME lane count (`batch<T>::size`). A kernel is written ONCE against
// this type and the lane width becomes the unroll factor -- the algorithm is
// decoupled from the SIMD width (the xtensor/xsimd idea; cf. std::simd, Eigen
// PacketMath). It exposes: aligned/unaligned load & store, broadcast, + - * /,
// a single fma() primitive (one FMA decision point), and horizontal
// reduce_add / reduce_min / reduce_max.
//
// Backends, selected at compile time:
//   * Google Highway (Apache-2.0 / BSD-3-Clause), when MTL5_HAS_HIGHWAY is
//     defined AND the target is not vector-length-agnostic (x86 SSE..AVX-512,
//     NEON, ...). One ISA per build, chosen by the compiler -m flags (static
//     dispatch). Enable with CMake -DMTL5_WITH_HIGHWAY=ON.
//   * Portable scalar fallback (size == 1) otherwise -- correct everywhere and
//     autovectorizable, so MTL5 stays MIT and dependency-free by default.
//
// The API surface is identical across backends, so the dense kernels (#86-#90)
// never need to know which one is active.

#include <cmath>
#include <cstddef>
#include <type_traits>

#if defined(MTL5_HAS_HIGHWAY)
#  include <hwy/highway.h>
#  if HWY_HAVE_SCALABLE
#    define MTL5_SIMD_USE_HIGHWAY 0   // scalable (SVE/RVV): no constexpr lane count yet -> scalar
#  else
#    define MTL5_SIMD_USE_HIGHWAY 1
#  endif
#else
#  define MTL5_SIMD_USE_HIGHWAY 0
#endif

namespace mtl::simd {

#if MTL5_SIMD_USE_HIGHWAY

namespace hn = hwy::HWY_NAMESPACE;

/// SIMD register of `T`, backed by Google Highway (static dispatch).
template <typename T>
class batch {
    static_assert(std::is_floating_point_v<T>,
                  "mtl::simd::batch currently supports floating-point lanes "
                  "(float/double) -- the dense-BLAS use case. The ops (/, fma, "
                  "reductions) assume floating semantics; integer lanes are a "
                  "future extension.");
    using D = hn::ScalableTag<T>;
    using V = hn::VFromD<D>;
    static constexpr D d_{};
    V v_;
    explicit batch(V v) : v_(v) {}

public:
    using value_type = T;
    /// Lane count for the compiled target (compile-time constant).
    static constexpr std::size_t size = HWY_MAX_LANES_D(D);

    batch() : v_(hn::Zero(d_)) {}
    explicit batch(T x) : v_(hn::Set(d_, x)) {}

    static batch load_aligned(const T* p)   { return batch(hn::Load(d_, p)); }
    static batch load_unaligned(const T* p) { return batch(hn::LoadU(d_, p)); }
    void store_aligned(T* p)   const { hn::Store(v_, d_, p); }
    void store_unaligned(T* p) const { hn::StoreU(v_, d_, p); }

    /// Load `size` values of a NARROWER type `Src` and widen each lane to `T`
    /// (e.g. load float, accumulate in double). The float descriptor is rebound
    /// to the same lane count as `T`'s, so exactly `size` source values are read.
    template <typename Src>
    static batch load_widen(const Src* p) {
        static_assert(sizeof(Src) < sizeof(T),
                      "load_widen widens; use load_unaligned for equal-width types");
        const hn::Rebind<Src, D> ds;
        return batch(hn::PromoteTo(d_, hn::LoadU(ds, p)));
    }

    friend batch operator+(batch a, batch b) { return batch(hn::Add(a.v_, b.v_)); }
    friend batch operator-(batch a, batch b) { return batch(hn::Sub(a.v_, b.v_)); }
    friend batch operator*(batch a, batch b) { return batch(hn::Mul(a.v_, b.v_)); }
    friend batch operator/(batch a, batch b) { return batch(hn::Div(a.v_, b.v_)); }

    /// Fused multiply-add: a*b + c (the single FMA decision point).
    friend batch fma(batch a, batch b, batch c) { return batch(hn::MulAdd(a.v_, b.v_, c.v_)); }

    friend T reduce_add(batch a) { return hn::ReduceSum(d_, a.v_); }
    friend T reduce_min(batch a) { return hn::ReduceMin(d_, a.v_); }
    friend T reduce_max(batch a) { return hn::ReduceMax(d_, a.v_); }
};

#else  // ---- portable scalar fallback (size == 1) ----

template <typename T>
class batch {
    static_assert(std::is_floating_point_v<T>,
                  "mtl::simd::batch currently supports floating-point lanes "
                  "(float/double) -- the dense-BLAS use case. The ops (/, fma, "
                  "reductions) assume floating semantics; integer lanes are a "
                  "future extension.");
    T v_{};

public:
    using value_type = T;
    static constexpr std::size_t size = 1;

    batch() = default;
    explicit batch(T x) : v_(x) {}

    static batch load_aligned(const T* p)   { return batch(p[0]); }
    static batch load_unaligned(const T* p) { return batch(p[0]); }
    void store_aligned(T* p)   const { p[0] = v_; }
    void store_unaligned(T* p) const { p[0] = v_; }

    /// Load one narrower value and widen to T (scalar fallback of the SIMD
    /// widening load; see the Highway variant).
    template <typename Src>
    static batch load_widen(const Src* p) {
        static_assert(sizeof(Src) < sizeof(T),
                      "load_widen widens; use load_unaligned for equal-width types");
        return batch(static_cast<T>(p[0]));
    }

    friend batch operator+(batch a, batch b) { return batch(a.v_ + b.v_); }
    friend batch operator-(batch a, batch b) { return batch(a.v_ - b.v_); }
    friend batch operator*(batch a, batch b) { return batch(a.v_ * b.v_); }
    friend batch operator/(batch a, batch b) { return batch(a.v_ / b.v_); }

    /// Fused multiply-add: a*b + c (the single FMA decision point).
    friend batch fma(batch a, batch b, batch c) {
        using std::fma;
        return batch(fma(a.v_, b.v_, c.v_));
    }

    friend T reduce_add(batch a) { return a.v_; }
    friend T reduce_min(batch a) { return a.v_; }
    friend T reduce_max(batch a) { return a.v_; }
};

#endif

/// Largest multiple of W not exceeding n -- the SIMD body length; iterate the
/// remainder [vectorizable_length(n) .. n) scalar:
///   for (i = 0; i < vectorizable_length(n, W); i += W) { ... }   // SIMD
///   for (; i < n; ++i) { ... }                                   // tail
constexpr std::size_t vectorizable_length(std::size_t n, std::size_t w) noexcept {
    return w == 0 ? 0 : n - (n % w);
}

/// Lane count of batch<T> for the current build (1 on the scalar fallback).
template <typename T>
inline constexpr std::size_t width = batch<T>::size;

} // namespace mtl::simd
