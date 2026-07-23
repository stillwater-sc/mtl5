#pragma once
// MTL5 -- accumulator policy for inner-product accumulation (issue #158)
//
// A single, cross-cutting customization point for how an inner product (a dot
// product, the columns of an LU, a GEMM element) is accumulated. It lets the
// THREE precisions of a mixed-precision tensor op be chosen independently:
//
//   * Value  -- the element/storage precision of the operands (bandwidth in)
//   * Acc    -- the accumulator precision the products are summed in (registers)
//   * Result -- the precision the rounded result is delivered/stored in (out)
//
// The trait abstracts a sum-of-products reduction behind clear/assign/
// add_product/value, so a kernel writes the same loop regardless of HOW the
// terms are combined. Three reduction configurations are expressible:
//
//   1. Plain accumulate (acc += product) -- the DEFAULT primary template.
//      The product `m*v` is formed in the accumulator precision `Acc`, rounded,
//      then added: two rounding events per term. Zero overhead and byte-
//      identical to hand-written arithmetic when `Acc == Value`; a wider `Acc`
//      than `Value` (e.g. fp32 accumulate over bf16 elements) gains accuracy
//      out of the box. Use any arithmetic type as `Acc` (e.g. `float`,`double`).
//
//   2. Fused multiply-add (acc = fma(m, v, acc)) -- `fma_accumulator<T>`.
//      The product is never rounded: `m*v + acc` is formed as if in infinite
//      precision and rounded ONCE to `T` per term. One rounding event per term
//      instead of two. Requires hardware/library FMA (`std::fma`).
//
//   3. Super-accumulator (exact sum of products, single final round-out).
//      A caller supplies a custom `Acc` (a compensated/Kahan accumulator, or a
//      Universal `quire` exact dot product) by specializing
//      `accumulator_traits<Acc, Value>`. MTL5 stays free of any external number
//      library, so the quire super-accumulator itself lives in the peer repo
//      that pairs MTL5 with Universal; this header only defines the contract it
//      plugs into. `value<Result>` then rounds the exact accumulator out once.
//
// Used by the sparse factorizations and the dense BLAS-level operations.

#include <cmath>      // std::fma
#include <concepts>   // std::floating_point

namespace mtl::math {

/// Accumulator policy for accumulating products of `Value`s into an `Acc`.
///
/// Configuration 1 (plain `acc += product`): the primary template. The product
/// is formed in `Acc`, so `Acc` wider than `Value` accumulates more accurately.
template <typename Acc, typename Value>
struct accumulator_traits {
    /// Reset the accumulator to zero.
    static void clear(Acc& a) { a = Acc{}; }

    /// Set the accumulator to a single value.
    static void assign(Acc& a, const Value& v) { a = v; }

    /// Round the accumulator out to `Result` (default: the element type `Value`),
    /// once, at the point the accumulated entry is consumed -- giving
    /// single-rounding ("exact dot product") semantics when `Acc` is exact, and
    /// fusing the accumulate->output conversion when `Result` differs from `Acc`.
    template <typename Result = Value>
    static Result value(const Acc& a) { return static_cast<Result>(a); }

    /// a += m * v : the canonical accumulate-a-product primitive (a dot product /
    /// quire is a sum of products). Callers pass a negated multiplier for a
    /// subtraction (e.g. the LU elimination update). The product is formed in the
    /// accumulator precision `Acc`, so a wider `Acc` than `Value` (e.g. fp32
    /// accumulate over bf16 elements) gains accuracy out of the box; this is a
    /// no-op cast and byte-identical when `Acc == Value`.
    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += static_cast<Acc>(m) * static_cast<Acc>(v);
    }
};

/// Configuration 2 -- fused multiply-add accumulator.
///
/// A distinguished accumulator type that selects the FMA reduction: pass it as
/// the `Acc` template argument to any accumulator-aware operation to fuse each
/// `m*v` into the running sum through `std::fma`, so the intermediate product is
/// never rounded (one rounding per term instead of two). `T` is the accumulation
/// precision held in registers; it need not equal the element precision `Value`.
///
/// `T` is constrained to `std::floating_point`: `std::fma` is an IEEE/hardware
/// primitive defined for the built-in floating types. A custom number type that
/// wants fused accumulation supplies its own accumulator specialization
/// (configuration 3), rather than reusing this one.
template <std::floating_point T = double>
struct fma_accumulator {
    T sum{};
};

/// Accumulator policy for the FMA reduction (configuration 2).
///
/// `add_product` computes `sum = m*v + sum` with a single rounding to `T`, so the
/// product `m*v` incurs no separate rounding event -- one rounding per term
/// instead of two. When `T` is wider than `Value` the operand widening is exact.
/// Eliminating the product rounding generally improves accuracy over the plain
/// path at the same `T`; over a long reduction the two accumulation-error streams
/// can occasionally align differently, so this is a per-term guarantee, not a
/// strict ordering on every possible sum.
template <std::floating_point T, typename Value>
struct accumulator_traits<fma_accumulator<T>, Value> {
    using Acc = fma_accumulator<T>;

    static void clear(Acc& a) { a.sum = T{}; }

    static void assign(Acc& a, const Value& v) { a.sum = static_cast<T>(v); }

    template <typename Result = Value>
    static Result value(const Acc& a) { return static_cast<Result>(a.sum); }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        a.sum = std::fma(static_cast<T>(m), static_cast<T>(v), a.sum);
    }
};

} // namespace mtl::math
