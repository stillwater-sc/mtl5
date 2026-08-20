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
//
// LANE TYPES AND THE INTEGER OVERFLOW CONTRACT (#451 phase 0)
//
// Supported lanes are float, double, int32_t and uint32_t -- see is_lane_v for
// why the integer entry stops at 32 bits. Integer arithmetic here is
// **two's-complement wrapping**: + - * and fma each return the exact
// mathematical result reduced mod 2^32. That is not a fallback, it is the
// native semantics of every target ISA (Highway specifies `operator*` as
// "truncating to the lower half for integer inputs"), and it buys a property
// floating point cannot offer:
//
//   addition mod 2^32 is associative and commutative, so an integer reduction
//   is BIT-IDENTICAL across lane counts, unroll factors, backends and thread
//   partitions.
//
// A float reduce_dot has to qualify every equality claim with an accumulation
// order; the int32 one does not. The cost is that the caller owns overflow:
// int32 x int32 accumulated in int32 wraps within a couple of terms at full
// range. Choosing an accumulator wide enough for the data is #451 phases 2-3,
// and tests/unit/simd/test_int_accumulation.cpp already pins the envelope.
//
// Division is NOT provided for integer lanes -- deliberately; see operator/.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// wrap_add / wrap_sub / wrap_mul: the two's-complement helpers the scalar paths
// here use. They live in mtl::detail rather than in this header because the
// generic element-wise loops in operation/ need them for the same reason these
// kernels do -- an integer op that misses the fast path must still land on the
// documented answer instead of on signed-overflow UB.
#include <mtl/detail/wrapping_arithmetic.hpp>

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

/// Lane types `batch<T>` supports.
///
/// Real floating point (the dense-BLAS case), plus **32-bit integers** as of
/// phase 0 of the integer-lane epic (#451). The integer entry is deliberately
/// narrow:
///
///   * 8- and 16-bit lanes are excluded because their VALUE is the widening
///     accumulate (`vpdpbusd` / `vpdpwssd` / SDOT), not the plain lane -- and
///     x86 has no 8-bit vector multiply at all. They arrive with the
///     accumulator contract, in phases 2-3.
///   * 64-bit lanes are excluded because `Mul` is emulated on x86 before
///     AVX512DQ, so a `batch<int64_t>` would advertise a vector multiply that
///     is a scalar loop in disguise.
///
/// 32-bit is exactly the width where the multiply is a single native
/// instruction on every target ISA (`vpmulld` / NEON `mul.4s`), which is what
/// makes it the honest plumbing case.
///
/// The floating half lists `float` and `double` rather than asking
/// `std::is_floating_point_v`, which also admits `long double`. Highway has no
/// `long double` vector type, so `batch<long double>` is a hard compile error
/// there -- and since this trait gates `SimdDenseVector`, saying otherwise sent
/// `dot(dense_vector<long double>, ...)` into `reduce_dot<long double>` and
/// broke a build that previously worked, because the old `BlasDenseVector` gate
/// kept it on the generic loop. The static_assert below already claimed "float,
/// double, int32_t and uint32_t"; this is the trait finally agreeing with it.
template <typename T>
inline constexpr bool is_lane_v =
    std::is_same_v<T, float>        || std::is_same_v<T, double> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>;

/// True for the integer lane types -- the ones whose arithmetic WRAPS.
template <typename T>
inline constexpr bool is_integer_lane_v = is_lane_v<T> && std::is_integral_v<T>;

#if MTL5_SIMD_USE_HIGHWAY

namespace hn = hwy::HWY_NAMESPACE;

/// SIMD register of `T`, backed by Google Highway (static dispatch).
template <typename T>
class batch {
    static_assert(is_lane_v<T>,
                  "mtl::simd::batch supports float, double, int32_t and uint32_t "
                  "lanes. 8/16-bit integers arrive with the widening accumulate "
                  "(#451 phases 2-3); 64-bit integer multiply is emulated on x86 "
                  "before AVX512DQ.");
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
    ///
    /// Floating lanes only. Widening an integer lane is the whole substance of
    /// #451 phases 2-3 -- it has to pick a signedness convention and an overflow
    /// contract for the accumulate -- so it is excluded here rather than
    /// inherited by accident from `sizeof(Src) < sizeof(T)`.
    template <typename Src>
    static batch load_widen(const Src* p) {
        static_assert(std::is_floating_point_v<T> && std::is_floating_point_v<Src>,
                      "load_widen is floating-point only; widening integer lanes "
                      "needs an accumulator contract (#451 phases 2-3)");
        static_assert(sizeof(Src) < sizeof(T),
                      "load_widen widens; use load_unaligned for equal-width types");
        const hn::Rebind<Src, D> ds;
        return batch(hn::PromoteTo(d_, hn::LoadU(ds, p)));
    }

    friend batch operator+(batch a, batch b) { return batch(hn::Add(a.v_, b.v_)); }
    friend batch operator-(batch a, batch b) { return batch(hn::Sub(a.v_, b.v_)); }
    friend batch operator*(batch a, batch b) { return batch(hn::Mul(a.v_, b.v_)); }

    /// Lane-wise division -- FLOATING LANES ONLY, and the omission is deliberate.
    ///
    /// Highway does offer an integer `Div`, but it is implementation-defined
    /// where `b[i] == 0` and where `a[i] == INT_MIN && b[i] == -1`, and it
    /// truncates. #450 took exactly this position one level up: `Field` now
    /// excludes the integers so `lu_factor(dense2D<int>)` fails to compile
    /// rather than returning a truncated factorization. Offering `batch<int32_t>
    /// / batch<int32_t>` would reopen that door underneath the concept. Generic
    /// code that divides therefore fails to compile on integer lanes, which is
    /// the intended answer.
    friend batch operator/(batch a, batch b)
        requires std::is_floating_point_v<T>
    { return batch(hn::Div(a.v_, b.v_)); }

    /// Fused multiply-add: a*b + c (the single FMA decision point).
    ///
    /// On integer lanes there is nothing to fuse and nothing to round: `MulAdd`
    /// is a multiply and an add, and the result is the exact product-sum reduced
    /// mod 2^32. So the "fused" in the name is a floating-point concern only,
    /// and the integer answer is exact by construction.
    friend batch fma(batch a, batch b, batch c) { return batch(hn::MulAdd(a.v_, b.v_, c.v_)); }

    friend T reduce_add(batch a) { return hn::ReduceSum(d_, a.v_); }
    friend T reduce_min(batch a) { return hn::ReduceMin(d_, a.v_); }
    friend T reduce_max(batch a) { return hn::ReduceMax(d_, a.v_); }
};

#else  // ---- portable scalar fallback (size == 1) ----

template <typename T>
class batch {
    static_assert(is_lane_v<T>,
                  "mtl::simd::batch supports float, double, int32_t and uint32_t "
                  "lanes. 8/16-bit integers arrive with the widening accumulate "
                  "(#451 phases 2-3); 64-bit integer multiply is emulated on x86 "
                  "before AVX512DQ.");
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
        static_assert(std::is_floating_point_v<T> && std::is_floating_point_v<Src>,
                      "load_widen is floating-point only; widening integer lanes "
                      "needs an accumulator contract (#451 phases 2-3)");
        static_assert(sizeof(Src) < sizeof(T),
                      "load_widen widens; use load_unaligned for equal-width types");
        return batch(static_cast<T>(p[0]));
    }

    friend batch operator+(batch a, batch b) { return batch(mtl::detail::wrap_add(a.v_, b.v_)); }
    friend batch operator-(batch a, batch b) { return batch(mtl::detail::wrap_sub(a.v_, b.v_)); }
    friend batch operator*(batch a, batch b) { return batch(mtl::detail::wrap_mul(a.v_, b.v_)); }

    /// Lane-wise division -- floating lanes only; see the Highway variant for why.
    friend batch operator/(batch a, batch b)
        requires std::is_floating_point_v<T>
    { return batch(a.v_ / b.v_); }

    /// Fused multiply-add: a*b + c (the single FMA decision point).
    ///
    /// `std::fma` is a floating-point function, so the integer case is the plain
    /// wrapping multiply-add -- which is exact mod 2^32 and therefore matches the
    /// Highway backend's `MulAdd` lane for lane.
    friend batch fma(batch a, batch b, batch c) {
        if constexpr (std::is_integral_v<T>) {
            return batch(mtl::detail::wrap_add(mtl::detail::wrap_mul(a.v_, b.v_), c.v_));
        } else {
            using std::fma;
            return batch(fma(a.v_, b.v_, c.v_));
        }
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
