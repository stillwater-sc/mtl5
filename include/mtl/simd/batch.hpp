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
#include <cstring>
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

/// Narrow integer types that can be WIDENED into a lane by a dot-product
/// accumulate (#451 phase 2). These are storage/operand types, not lanes:
/// `batch<std::int16_t>` does not exist and is not wanted, because a 16-bit
/// lane's value is the widening accumulate, not the plain multiply.
template <typename T>
inline constexpr bool is_widenable_v =
    std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t>;

/// The accumulator a narrow type widens into: 16 bits doubles to 32, keeping
/// signedness. Products of two 16-bit values always fit the 32-bit accumulator
/// EXACTLY (|-2^15 * -2^15| = 2^30 < 2^31); it is the running sum that can
/// overflow, and does so quickly -- see reduce_dot_widen in simd/algorithm.hpp
/// for the headroom this actually buys.
template <typename Narrow>
struct widen_accumulator;
template <> struct widen_accumulator<std::int16_t>  { using type = std::int32_t;  };
template <> struct widen_accumulator<std::uint16_t> { using type = std::uint32_t; };

template <typename Narrow>
using widen_accumulator_t = typename widen_accumulator<Narrow>::type;

/// Narrow integer types whose dot product widens by FOUR into a 32-bit lane,
/// via the hardware's quad multiply-accumulate (#451 phase 3) -- VNNI's
/// `vpdpbusd` on x86, `SDOT`/`UDOT` on NEON. Four products are summed into one
/// accumulator lane by a single instruction.
///
/// This is the case the epic calls "the one that matters", and the reason is
/// headroom rather than lane count: an 8-bit product needs 15 bits, leaving ~16
/// in an int32 accumulator -- of the order of 10^5 terms, against the handful
/// that 16-bit operands afford (see reduce_dot_widen in simd/algorithm.hpp).
template <typename T>
inline constexpr bool is_quad_widenable_v =
    std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>;

/// The accumulator for a quad-widening operand PAIR.
///
/// Three pairings, and the asymmetry is the hardware's, not a design choice:
/// VNNI implements `unsigned x signed` natively (`_mm_dpbusd_epi32`) because
/// that is the shape of quantized inference -- unsigned activations against
/// signed weights. Symmetric `i8 x i8` is native only from AVX10.2
/// (`_mm_dpbssd_epi32`); before that Highway emulates it with two `dpbusd`
/// calls plus a shift and subtract, so it costs about three times the native
/// form on every x86 shipping today. NEON has all three natively.
///
/// `(int8, uint8)` is deliberately absent rather than silently reordered: a dot
/// product is symmetric, so a caller holding that pairing should swap the
/// arguments and get the native instruction, and being told so at compile time
/// is better than being given the emulated path without knowing.
template <typename A, typename B> struct quad_accumulator;
template <> struct quad_accumulator<std::int8_t,  std::int8_t>  { using type = std::int32_t;  };
template <> struct quad_accumulator<std::uint8_t, std::int8_t>  { using type = std::int32_t;  };
template <> struct quad_accumulator<std::uint8_t, std::uint8_t> { using type = std::uint32_t; };

template <typename A, typename B>
using quad_accumulator_t = typename quad_accumulator<A, B>::type;

/// Is `(A, B)` a quad-widening pair the hardware op accepts?
template <typename A, typename B>
concept QuadPair = requires { typename quad_accumulator<A, B>::type; };

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
    /// Integer lanes are permitted too, and the rule is SAME SIGNEDNESS. Phase 0
    /// excluded them because widening an integer needed a signedness convention
    /// and an overflow contract that did not exist yet; phases 2-3 settled both,
    /// and the integer GEMM needs this load. Sign extension for signed sources,
    /// zero extension for unsigned, and no mixing -- a mixed pair is a question
    /// about the caller's intent, not about the load.
    ///
    /// Note this is the WIDEN-ON-LOAD path, not the hardware dot product: it
    /// promotes narrow operands into TC-wide lanes and then does an ordinary
    /// multiply-add. It captures the memory-traffic win (one byte per element
    /// instead of eight) which is where most of the integer speedup lives --
    /// see docs/performance -- but not the specialised instruction, which needs
    /// a quad-interleaved pack layout and the broadcast-quad micro-kernel in
    /// detail/gemm_quad_microkernel.hpp (#451 phase 5). For a GEMM that one is
    /// measurably faster even where the instruction is emulated, so this path is
    /// the fallback for pairs the quad op does not accept, not the preferred one.
    template <typename Src>
    static batch load_widen(const Src* p) {
        static_assert((std::is_floating_point_v<T> && std::is_floating_point_v<Src>) ||
                      (std::is_integral_v<T> && std::is_integral_v<Src> &&
                       std::is_signed_v<T> == std::is_signed_v<Src>),
                      "load_widen takes float->float or integer->integer of the "
                      "SAME signedness; mixing them is a question about intent");
        static_assert(sizeof(Src) < sizeof(T),
                      "load_widen widens; use load_unaligned for equal-width types");
        const hn::Rebind<Src, D> ds;
        return batch(hn::PromoteTo(d_, hn::LoadU(ds, p)));
    }

    /// Narrow values consumed by one `widen_dot_accumulate` call: a full narrow
    /// vector, which holds twice as many 16-bit lanes as this batch holds 32-bit
    /// ones. Compile-time, like `size`.
    ///
    /// Only meaningful on a 32-bit integer accumulator -- it is the loop stride
    /// for `reduce_dot_widen`, and `widen_dot_accumulate` static_asserts the
    /// lane/operand pairing anyway, so the constant existing on `batch<double>`
    /// is inert rather than a second gate to keep in sync.
    static constexpr std::size_t widen_step = 2 * size;

    /// Widening dot accumulate: multiply `widen_step` pairs of NARROW values and
    /// add the products into two wide accumulators.
    ///
    /// The lane assignment is deliberately UNSPECIFIED. Highway's
    /// `ReorderWidenMulAccumulate` promises only that
    /// `reduce_add(sum0 + sum1)` is the sum of all the products -- which lane a
    /// given product lands in is a permutation the ISA chooses (`vpmaddwd` pairs
    /// adjacent lanes on x86; NEON splits low/high halves). That is exactly
    /// enough for a reduction and nothing more, so this primitive is only usable
    /// inside one.
    ///
    /// For a FLOATING accumulator such a permutation would be a problem: the sum
    /// would depend on which products got grouped, and the result would vary by
    /// ISA. On integer lanes it costs nothing, because addition mod 2^32 is
    /// associative and commutative -- the permutation is unobservable. The
    /// contract established in phase 0 is what makes this instruction usable
    /// without qualification.
    ///
    /// `sum1` MUST start zeroed; Highway leaves it untouched on targets that do
    /// not need it, so a nonzero initial value would silently be counted or not
    /// depending on the ISA.
    template <typename Narrow>
    static void widen_dot_accumulate(const Narrow* a, const Narrow* b,
                                     batch& sum0, batch& sum1) {
        static_assert(is_widenable_v<Narrow>, "widen_dot_accumulate takes 16-bit operands");
        static_assert(std::is_same_v<T, widen_accumulator_t<Narrow>>,
                      "the accumulator must be the widened counterpart of Narrow "
                      "(int16 -> int32, uint16 -> uint32)");
        const hn::Repartition<Narrow, D> dn;
        auto s1 = sum1.v_;
        sum0.v_ = hn::ReorderWidenMulAccumulate(d_, hn::LoadU(dn, a), hn::LoadU(dn, b),
                                                sum0.v_, s1);
        sum1.v_ = s1;
    }

    /// Narrow values consumed by one `quad_dot_accumulate` call: a full narrow
    /// vector, holding four times as many 8-bit lanes as this batch holds
    /// 32-bit ones.
    static constexpr std::size_t quad_step = 4 * size;

    /// Quad-widening dot accumulate: four 8-bit products summed into each
    /// 32-bit accumulator lane by one instruction (#451 phase 3).
    ///
    /// Note what is NOT here: a second accumulator. The pairwise op
    /// (`widen_dot_accumulate`) needs one because it may permute products across
    /// lanes and only guarantees the total. `SumOfMulQuadAccumulate` specifies
    /// exactly which four products land in lane i, so there is nothing to
    /// reorder and one accumulator suffices. The wrapping contract still does
    /// the work of making the horizontal sum order-independent.
    ///
    /// The two operand types may differ: `(uint8, int8)` is VNNI's native shape
    /// and the reason the instruction exists. See `quad_accumulator`.
    template <typename NA, typename NB>
    static void quad_dot_accumulate(const NA* a, const NB* b, batch& sum) {
        static_assert(is_quad_widenable_v<NA> && is_quad_widenable_v<NB>,
                      "quad_dot_accumulate takes 8-bit operands");
        static_assert(std::is_same_v<T, quad_accumulator_t<NA, NB>>,
                      "accumulator must match the operand pairing: i8*i8 and "
                      "u8*i8 -> int32, u8*u8 -> uint32");
        const hn::Repartition<NA, D> dna;
        const hn::Repartition<NB, D> dnb;
        sum.v_ = hn::SumOfMulQuadAccumulate(d_, hn::LoadU(dna, a), hn::LoadU(dnb, b), sum.v_);
    }

    /// Quad multiply-accumulate with a BROADCAST left operand -- the GEMM shape
    /// of the instruction, and the reason the micro-kernel needs a primitive of
    /// its own (#451 phase 5).
    ///
    /// `quad_dot_accumulate` walks two vectors in step, which is a dot product.
    /// A GEMM does not: one C row needs ONE A element (four of them here, for
    /// four k values) multiplied against a whole row of B. So the left operand
    /// is the same four narrow values in every lane, and lane j accumulates
    ///
    ///     sum[j] += sum_{q<4} a4[q] * b[4*j + q]
    ///
    /// which is exactly the rank-4 update the packed panels are laid out for:
    /// `a4` is A(i, p..p+3) and `b[4*j+q]` is B(p+q, j). Four k steps per
    /// instruction, against the widening load's one.
    ///
    /// The broadcast is a 32-bit splat of the four bytes AS THEY LIE IN MEMORY
    /// (memcpy, then `Set`, then a reinterpret back to narrow lanes) -- one
    /// `vpbroadcastd` / `LD1R`, not four inserts. Note this ties the quad's lane
    /// order to the byte order of the machine: on a little-endian target lane 0
    /// receives `a4[0]`, which is what pairs it with the `b` load. Every target
    /// that HAS this instruction is little-endian, and
    /// tests/unit/simd/test_quad_dot.cpp checks the pairing against a scalar
    /// reference, so a big-endian port would fail loudly rather than quietly
    /// transpose the quad.
    template <typename NA, typename NB>
    static void quad_dot_broadcast_accumulate(const NA* a4, const NB* b, batch& sum) {
        static_assert(is_quad_widenable_v<NA> && is_quad_widenable_v<NB>,
                      "quad_dot_broadcast_accumulate takes 8-bit operands");
        static_assert(std::is_same_v<T, quad_accumulator_t<NA, NB>>,
                      "accumulator must match the operand pairing: i8*i8 and "
                      "u8*i8 -> int32, u8*u8 -> uint32");
        static_assert(sizeof(T) == 4 * sizeof(NA), "the broadcast unit is four narrow values");
        const hn::Repartition<NA, D> dna;
        const hn::Repartition<NB, D> dnb;
        T quad;                                  // a4[0..3], in memory order
        std::memcpy(&quad, a4, sizeof(T));
        sum.v_ = hn::SumOfMulQuadAccumulate(d_, hn::BitCast(dna, hn::Set(d_, quad)),
                                            hn::LoadU(dnb, b), sum.v_);
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
        static_assert((std::is_floating_point_v<T> && std::is_floating_point_v<Src>) ||
                      (std::is_integral_v<T> && std::is_integral_v<Src> &&
                       std::is_signed_v<T> == std::is_signed_v<Src>),
                      "load_widen takes float->float or integer->integer of the "
                      "SAME signedness; mixing them is a question about intent");
        static_assert(sizeof(Src) < sizeof(T),
                      "load_widen widens; use load_unaligned for equal-width types");
        return batch(static_cast<T>(p[0]));
    }

    /// Narrow values consumed per widen_dot_accumulate call. Two rather than
    /// one, so the fallback drives BOTH accumulators and the loop in
    /// reduce_dot_widen is the same shape on either backend -- which also means
    /// the fallback exercises the sum1 path that Highway may or may not use.
    static constexpr std::size_t widen_step = 2;

    /// Widening dot accumulate -- scalar fallback of the SIMD primitive; see the
    /// Highway variant for the contract. Splitting the two products across the
    /// two accumulators reproduces the "unspecified lane assignment, defined
    /// total" behaviour rather than papering over it.
    template <typename Narrow>
    static void widen_dot_accumulate(const Narrow* a, const Narrow* b,
                                     batch& sum0, batch& sum1) {
        static_assert(is_widenable_v<Narrow>, "widen_dot_accumulate takes 16-bit operands");
        static_assert(std::is_same_v<T, widen_accumulator_t<Narrow>>,
                      "the accumulator must be the widened counterpart of Narrow "
                      "(int16 -> int32, uint16 -> uint32)");
        sum0.v_ = mtl::detail::wrap_add(
            sum0.v_, mtl::detail::wrap_mul(static_cast<T>(a[0]), static_cast<T>(b[0])));
        sum1.v_ = mtl::detail::wrap_add(
            sum1.v_, mtl::detail::wrap_mul(static_cast<T>(a[1]), static_cast<T>(b[1])));
    }

    /// Narrow values consumed per quad_dot_accumulate call -- four, matching the
    /// instruction's four-products-per-lane shape rather than the lane count.
    static constexpr std::size_t quad_step = 4;

    /// Quad-widening dot accumulate -- scalar fallback; see the Highway variant.
    /// Four products into the single accumulator, which is exactly what the
    /// instruction does to one lane.
    template <typename NA, typename NB>
    static void quad_dot_accumulate(const NA* a, const NB* b, batch& sum) {
        static_assert(is_quad_widenable_v<NA> && is_quad_widenable_v<NB>,
                      "quad_dot_accumulate takes 8-bit operands");
        static_assert(std::is_same_v<T, quad_accumulator_t<NA, NB>>,
                      "accumulator must match the operand pairing: i8*i8 and "
                      "u8*i8 -> int32, u8*u8 -> uint32");
        for (std::size_t k = 0; k < 4; ++k)
            sum.v_ = mtl::detail::wrap_add(
                sum.v_, mtl::detail::wrap_mul(static_cast<T>(a[k]), static_cast<T>(b[k])));
    }

    /// Quad multiply-accumulate with a broadcast left operand -- scalar
    /// fallback; see the Highway variant for the contract. With one lane the
    /// broadcast is a no-op, so this is the same four products as
    /// quad_dot_accumulate -- which is the definition the SIMD path has to
    /// reproduce lane by lane, and the reference the tests compare against.
    template <typename NA, typename NB>
    static void quad_dot_broadcast_accumulate(const NA* a4, const NB* b, batch& sum) {
        static_assert(is_quad_widenable_v<NA> && is_quad_widenable_v<NB>,
                      "quad_dot_broadcast_accumulate takes 8-bit operands");
        static_assert(std::is_same_v<T, quad_accumulator_t<NA, NB>>,
                      "accumulator must match the operand pairing: i8*i8 and "
                      "u8*i8 -> int32, u8*u8 -> uint32");
        for (std::size_t q = 0; q < 4; ++q)
            sum.v_ = mtl::detail::wrap_add(
                sum.v_, mtl::detail::wrap_mul(static_cast<T>(a4[q]), static_cast<T>(b[q])));
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

/// Name of the SIMD backend actually compiled -- "scalar", or Highway's target
/// ("AVX3_DL", "AVX2", "SSE4", "NEON", ...).
///
/// This exists for PROVENANCE. Whether the int8 kernel got `vpdpbusd` or a
/// decomposition into `vpmaddwd` is not visible in a result, only in a
/// throughput number that nobody can check afterwards -- and the difference is
/// several times the work. `AVX3_DL` is the target that has the instruction;
/// anything else on x86 does not, however much AVX-512 the machine supports.
/// #451 phase 4 records this in every benchmark sidecar for that reason.
inline const char* backend_name() noexcept {
#if MTL5_SIMD_USE_HIGHWAY
    return hwy::TargetName(HWY_TARGET);
#else
    return "scalar";
#endif
}

// -- native quad multiply-accumulate, PER OPERAND PAIRING ---------------------
//
// WHY THIS IS NOT ONE BOOLEAN. The hardware support is per pairing, and the two
// ISAs that have it disagree about WHICH pairing they implement -- so a single
// flag cannot describe either machine honestly, and reports the wrong thing
// about at least one arm on both:
//
//                     x86 AVX3_DL   x86 AVX10.2   NEON+DotProd   NEON+I8MM
//     u8 x i8           NATIVE        native       emulated       NATIVE
//     i8 x i8          emulated       native        NATIVE        native
//     u8 x u8          emulated       native        NATIVE        native
//
// x86 implements the mixed form first (`vpdpbusd` -- unsigned activations
// against signed weights, the shape quantized inference produces) and gets the
// symmetric ones only with AVX10.2's `vpdpbssd`/`vpdpbuud`. ARM implements the
// symmetric ones first (`SDOT`/`UDOT`, Armv8.2 FEAT_DotProd) and gets the mixed
// one only with I8MM's `USDOT` (Armv8.6). Everything else is emulated at about
// three times the work -- two native calls plus a shift and a subtract.
//
// The consequence for measurement is the whole reason this exists: the arm that
// is fastest on every x86 (`u8 x i8`) is the SLOW one on a Cortex-A78, and a
// cross-machine comparison that pairs arms by NAME rather than by whether they
// were native gets the sign of the effect wrong.
//
// These mirror HIGHWAY'S OWN CONDITIONS -- `HWY_TARGET <= HWY_AVX3_DL` is
// literally the gate in x86_128-inl.h, and the ARM clauses are those in
// arm_neon-inl.h -- rather than reconstructing them from feature macros. The
// previous single flag did reconstruct them, from a seven-macro AVX-512
// conjunction, which was correct but had to be re-derived by hand every time
// Highway moved.

#if !MTL5_SIMD_USE_HIGHWAY
#  define MTL5_QUAD_NATIVE_U8I8 0
#  define MTL5_QUAD_NATIVE_I8I8 0
#  define MTL5_QUAD_NATIVE_U8U8 0

#elif HWY_ARCH_X86
// `HWY_X86_HAVE_AVX10_2_OPS` also carries the COMPILER-version requirement for
// the AVX10.2 intrinsics, which `__AVX10_2__` alone does not: a compiler too old
// to emit `vpdpbssd` makes Highway fall back to the emulation even on a target
// that advertises the ISA. Prefer it, and fall back to the ISA macro only if a
// future Highway stops defining it.
#  if defined(HWY_X86_HAVE_AVX10_2_OPS)
#    define MTL5_X86_QUAD_AVX10_2 HWY_X86_HAVE_AVX10_2_OPS
#  elif defined(__AVX10_2__)
#    define MTL5_X86_QUAD_AVX10_2 1
#  else
#    define MTL5_X86_QUAD_AVX10_2 0
#  endif
#  if HWY_TARGET <= HWY_AVX3_DL
#    define MTL5_QUAD_NATIVE_U8I8 1        // _mm_dpbusd_epi32
#  else
#    define MTL5_QUAD_NATIVE_U8I8 0
#  endif
#  define MTL5_QUAD_NATIVE_I8I8 MTL5_X86_QUAD_AVX10_2   // _mm_dpbssd_epi32
#  define MTL5_QUAD_NATIVE_U8U8 MTL5_X86_QUAD_AVX10_2   // _mm_dpbuud_epi32

#elif HWY_ARCH_ARM
// SDOT/UDOT: FEAT_DotProd, or the NEON_BF16 target which implies it.
#  if defined(__ARM_FEATURE_DOTPROD) || HWY_TARGET == HWY_NEON_BF16
#    define MTL5_QUAD_NATIVE_I8I8 1        // vdotq_s32
#    define MTL5_QUAD_NATIVE_U8U8 1        // vdotq_u32
#  else
#    define MTL5_QUAD_NATIVE_I8I8 0
#    define MTL5_QUAD_NATIVE_U8U8 0
#  endif
// USDOT: FEAT_I8MM. The Apple clause is Highway's, kept verbatim -- those cores
// have the instruction without the build advertising the feature macro.
#  if defined(__ARM_FEATURE_MATMUL_INT8) ||                                    \
      (HWY_TARGET == HWY_NEON_BF16 && HWY_OS_APPLE && HWY_ARCH_ARM_A64 &&      \
       defined(HWY_HAVE_RUNTIME_DISPATCH) && HWY_HAVE_RUNTIME_DISPATCH)
#    define MTL5_QUAD_NATIVE_U8I8 1        // vusdotq_s32
#  else
#    define MTL5_QUAD_NATIVE_U8I8 0
#  endif

#else
#  define MTL5_QUAD_NATIVE_U8I8 0
#  define MTL5_QUAD_NATIVE_I8I8 0
#  define MTL5_QUAD_NATIVE_U8U8 0
#endif

/// Does this build have a NATIVE quad multiply-accumulate for the pairing
/// `(NA, NB)`, or will Highway emulate it? See the table above.
///
/// False for any pairing the op does not accept at all -- `(i8, u8)` is absent
/// on purpose, so it reads as "not native", which is true and is also what a
/// caller should act on.
template <typename NA, typename NB>
inline constexpr bool has_native_quad_dot_v = false;

template <> inline constexpr bool
    has_native_quad_dot_v<std::uint8_t, std::int8_t>  = (MTL5_QUAD_NATIVE_U8I8 != 0);
template <> inline constexpr bool
    has_native_quad_dot_v<std::int8_t,  std::int8_t>  = (MTL5_QUAD_NATIVE_I8I8 != 0);
template <> inline constexpr bool
    has_native_quad_dot_v<std::uint8_t, std::uint8_t> = (MTL5_QUAD_NATIVE_U8U8 != 0);

#undef MTL5_QUAD_NATIVE_U8I8
#undef MTL5_QUAD_NATIVE_I8I8
#undef MTL5_QUAD_NATIVE_U8U8
#ifdef MTL5_X86_QUAD_AVX10_2
#  undef MTL5_X86_QUAD_AVX10_2
#endif

/// How much of the quad multiply-accumulate this build gets natively.
///
/// Three states rather than two, because `partial` is the COMMON case: every
/// machine measured so far that has the instruction at all has it for some
/// pairings and not others. A benchmark that reports "native" or "decomposed"
/// against a partial build is mislabelling at least one of its arms.
enum class quad_dot_support {
    none,       ///< every pairing emulated -- e.g. SSE4, AVX2, plain NEON
    partial,    ///< some native, some emulated -- AVX3_DL, and NEON+DotProd
    all,        ///< every pairing native -- AVX10.2, or NEON with I8MM
};

inline constexpr quad_dot_support quad_dot_native_support = [] {
    constexpr bool a = has_native_quad_dot_v<std::uint8_t, std::int8_t>;
    constexpr bool b = has_native_quad_dot_v<std::int8_t,  std::int8_t>;
    constexpr bool c = has_native_quad_dot_v<std::uint8_t, std::uint8_t>;
    if constexpr (a && b && c) return quad_dot_support::all;
    else if constexpr (a || b || c) return quad_dot_support::partial;
    else return quad_dot_support::none;
}();

/// Does this build have a native quad multiply-accumulate for ANY pairing?
///
/// Retained with its original meaning, which is what every existing caller and
/// every committed sidecar assumed: it is the "did this build get the
/// instruction at all" question. It is deliberately NOT "all pairings" -- that
/// would read false on Zen 4 and on a Cortex-A78 alike and erase the distinction
/// the sidecars exist to record. Use `has_native_quad_dot_v<NA, NB>` to say
/// anything about a specific arm, and `quad_dot_native_support` to report.
inline constexpr bool has_native_quad_dot = (quad_dot_native_support != quad_dot_support::none);

/// "NATIVE", "PARTIAL" or "DECOMPOSED" -- the token a benchmark sidecar records.
inline const char* quad_dot_support_name() noexcept {
    switch (quad_dot_native_support) {
        case quad_dot_support::all:     return "NATIVE";
        case quad_dot_support::partial: return "PARTIAL";
        default:                        return "DECOMPOSED";
    }
}

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

/// The SCALAR equivalent of `fma(batch,batch,batch)` -- what one lane of it does.
///
/// A kernel's vector body and its scalar tail must agree, or an element's
/// rounding depends on its INDEX: the first W*floor(n/W) elements are computed
/// one way and the last n mod W another (#512).
///
/// IT IS THE PRIMITIVE ITSELF, not a reimplementation of it. Broadcasting to a
/// batch, applying the same `fma`, and taking a lane back is exact -- every lane
/// holds the identical value, so `reduce_max` recovers it -- and it CANNOT drift
/// from the body, because it is the body's operation.
///
/// The first version of this did try to mirror the branch, guarded on
/// `HWY_NATIVE_FMA`, on the reasoning that Highway's `MulAdd` is a hardware FMA
/// only where the target has one. CI falsified it: on the clang Highway lane the
/// batch `fma` fused while the guard read the macro as absent, so the mirror
/// took the decomposed branch and disagreed with the very operation it was
/// mirroring. A predicate that has to stay in sync with a third-party header's
/// per-target macros is the wrong mechanism when the operation itself is one
/// call away.
///
/// The tail runs at most W-1 times per kernel, so the broadcast and horizontal
/// reduce are not worth avoiding for a correctness property.
///
/// Integer lanes need no special case: `fma` on an integer batch is the wrapping
/// multiply-add, exact mod 2^N on both backends, which is the #451/#460 contract.
template <typename T>
inline T scalar_fma(T a, T b, T c) {
    return reduce_max(fma(batch<T>(a), batch<T>(b), batch<T>(c)));
}

} // namespace mtl::simd
