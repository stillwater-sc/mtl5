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
//      instead of two. The fused step is `using std::fma; fma(...)`, so `T` may
//      be a built-in float or any custom arithmetic type with an ADL-found `fma`.
//
//   3. Super-accumulator (exact sum of products, single final round-out).
//      A caller supplies a custom `Acc` (a compensated/Kahan accumulator, or a
//      Universal `quire` exact dot product) in one of two ways:
//
//        a. PREFERRED -- the ADL hook. The accumulator's own namespace provides
//           the free functions listed at `adl_accumulator` below, and the
//           constrained specialization here forwards to them. Nobody includes
//           anybody: MTL5 names the functions, the number library defines them
//           next to its accumulator type, and an operation such as
//           `dot<quire<...>, Result>(x, y)` finds them by argument-dependent
//           lookup. One definition in the number library serves every
//           downstream repository.
//        b. Specializing `accumulator_traits<Acc, Value>` directly. This still
//           works and still wins over the hook (a specialization naming the
//           accumulator type is more specialized than the constrained one), so
//           an existing bridge header keeps compiling until it is deleted.
//
//      MTL5 stays free of any external number library either way; this header
//      only defines the contract. `value<Result>` then rounds the exact
//      accumulator out once.
//
// Used by the sparse factorizations and the dense BLAS-level operations.
//
// WORKSPACE COST (#342). Several accumulator-aware kernels allocate a
// std::vector<Acc> whose length scales with the problem, not with a register.
// Measured peak live accumulators on a 2-D Laplacian, so this is the real
// scaling of these kernels rather than an estimate:
//
//     kernel             n=400   n=1600   n=6400   peak/n
//     sparse_lu            400     1600     6400   1.00
//     supernodal_ldlt      489     3073     9558   1.22 - 1.92
//
// i.e. Theta(n), and up to ~2n for the supernodal dense panel, whose exact
// multiplier tracks supernode structure and so is matrix-dependent.
// supernodal_lu, the transposed SpMV in operation/mult.hpp and the R
// back-substitution of qr_numeric::solve<Acc> (#535) have the same shape.
//
// That is free for configurations 1 and 2 -- both are the size of the value
// type -- but a configuration-3 super-accumulator is sized to the value type's
// complete dynamic range plus carry-guard bits, so it is one to two orders of
// magnitude larger per element. At n = 1e6 a workspace that costs 8 MB with a
// double costs on the order of 64 MB with a posit32 quire and ~536 MB with a
// binary64 Kulisch accumulator, before the supernodal multiplier.
//
// POLICY: the kernels are NOT restructured for this. A caller choosing an
// exact accumulator is choosing a workspace proportional to sizeof(Acc), and
// the capacity/guard bits of that accumulator are the lever -- a quire sized
// for a shorter reduction is correspondingly smaller. Revisit with measurements
// once a quire specialization exists to measure; restructuring these inner
// loops for a type that cannot yet be instantiated would be speculative, and
// the alternatives are not free either (blocking trades passes for memory;
// splitting the accumulation weakens the single-rounding guarantee that is the
// entire point of configuration 3).
//
// tests/unit/sparse/test_accumulator_workspace.cpp pins the scaling above, so a
// change that made a workspace superlinear fails rather than silently shipping.

#include <mtl/math/identity.hpp>
#include <cmath>        // std::fma
#include <concepts>
#include <type_traits>

namespace mtl::math {

/// Accumulator policy for accumulating products of `Value`s into an `Acc`.
///
/// Configuration 1 (plain `acc += product`): the primary template. The product
/// is formed in `Acc`, so `Acc` wider than `Value` accumulates more accurately.
template <typename Acc, typename Value>
struct accumulator_traits {
    /// Reset the accumulator to zero.
    ///
    /// `math::zero<Acc>()`, i.e. `Acc{0}`, NOT a value-initialized `Acc{}`. The two
    /// agree for the built-in types but not in general: a logarithmic number
    /// system encodes 2^0 in its all-zero bit pattern, so Universal's `lns{}` is
    /// 1.0. With `Acc{}` here, every sum of products over an `lns` accumulator
    /// started at 1 -- which, once householder() routed its default-path sum of
    /// squares through this function (#537), broke QR/LQ for lns outright.
    static void clear(Acc& a) { a = math::zero<Acc>(); }

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

    /// a += b : combine two partial accumulators of the same reduction, e.g. the
    /// per-thread or per-process partials of a parallel dot product. For this
    /// configuration it is an ordinary add, one rounding; for an exact
    /// super-accumulator it is exact, which is what makes a parallel reduction
    /// independent of how the terms were partitioned. Offered only when `Acc`
    /// has a `+=`, so `mergeable_accumulator` is an accurate test.
    static void merge(Acc& a, const Acc& b)
        requires requires(Acc& x, const Acc& y) { x += y; }
    {
        a += b;
    }
};

/// Configuration 3 via the ADL hook: does `Acc`'s own namespace provide the
/// super-accumulator functions for products of `Value`s?
///
/// The functions, found by argument-dependent lookup on `Acc`:
///
///   accumulator_clear(acc)                          acc = 0
///   accumulator_assign(acc, v)                      acc = v, exactly
///   accumulator_add_product(acc, m, v)              acc += m * v
///   accumulator_value(acc, std::type_identity<R>{}) -> R, the single round-out
///   accumulator_merge(acc, other)                   acc += other  (optional)
///
/// `accumulator_value` takes a `std::type_identity<Result>` tag rather than an
/// explicit template argument so that the call is an ordinary ADL call on every
/// compiler. `accumulator_merge` is optional: an accumulator without it still
/// satisfies the hook, and only `merge()` is unavailable.
///
/// Constrain the functions on the accumulator type. They are also looked up in
/// the namespaces associated with `Value`, so an unconstrained
/// `template <class A> void accumulator_clear(A&)` in a number library would
/// claim `accumulator_traits<double, its_number>` too.
template <typename Acc, typename Value>
concept adl_accumulator = requires(Acc& a, const Acc& ca, const Value& v) {
    accumulator_clear(a);
    accumulator_assign(a, v);
    accumulator_add_product(a, v, v);
    { accumulator_value(ca, std::type_identity<Value>{}) } -> std::convertible_to<Value>;
};

/// Accumulator policy for an accumulator that provides the ADL hook
/// (configuration 3, see `adl_accumulator`). Each operation forwards to the
/// accumulator's own free function, so the number library that defines the
/// accumulator defines its policy once, without including MTL5.
template <typename Acc, typename Value>
    requires adl_accumulator<Acc, Value>
struct accumulator_traits<Acc, Value> {
    static void clear(Acc& a) { accumulator_clear(a); }

    static void assign(Acc& a, const Value& v) { accumulator_assign(a, v); }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        return accumulator_value(a, std::type_identity<Result>{});
    }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        accumulator_add_product(a, m, v);
    }

    static void merge(Acc& a, const Acc& b)
        requires requires(Acc& x, const Acc& y) { accumulator_merge(x, y); }
    {
        accumulator_merge(a, b);
    }
};

/// Configuration 2 -- fused multiply-add accumulator.
///
/// A distinguished accumulator type that selects the FMA reduction: pass it as
/// the `Acc` template argument to any accumulator-aware operation to fuse each
/// `m*v` into the running sum, so the intermediate product is never rounded (one
/// rounding per term instead of two). `T` is the accumulation precision held in
/// registers; it need not equal the element precision `Value`.
///
/// `T` is intentionally unconstrained so this stays usable by custom arithmetic
/// types: the fused step is `using std::fma; fma(...)`, which selects `std::fma`
/// for the built-in floating types and, via ADL, a type-specific fused multiply-
/// add for a custom number type (e.g. a posit `fma`). A type that has no fused
/// operation simply does not satisfy the call and is not a valid `T`.
template <typename T = double>
struct fma_accumulator {
    T sum = math::zero<T>();   // not T{}: see accumulator_traits::clear
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
template <typename T, typename Value>
struct accumulator_traits<fma_accumulator<T>, Value> {
    using Acc = fma_accumulator<T>;

    static void clear(Acc& a) { a.sum = math::zero<T>(); }

    static void assign(Acc& a, const Value& v) { a.sum = static_cast<T>(v); }

    template <typename Result = Value>
    static Result value(const Acc& a) { return static_cast<Result>(a.sum); }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        using std::fma;   // ADL: std::fma for built-ins, a custom fma for user types
        a.sum = fma(static_cast<T>(m), static_cast<T>(v), a.sum);
    }

    /// Combine two partial sums: one ordinary add, rounded to `T`.
    static void merge(Acc& a, const Acc& b) { a.sum += b.sum; }
};

/// Can two partial accumulators of a reduction be combined?
///
/// True for configurations 1 and 2, and for an ADL accumulator that provides
/// `accumulator_merge`; false for a specialization without a `merge`. The
/// reductions in dot.hpp and norms.hpp split across the thread pool only when
/// this holds (#582), and otherwise run serially as before.
template <typename Acc, typename Value>
concept mergeable_accumulator = requires(Acc& a, const Acc& b) {
    accumulator_traits<Acc, Value>::merge(a, b);
};

/// The arithmetic type an accumulator should be rounded out to before a scalar
/// post-processing step such as `sqrt`.
///
/// `value<Result>` rounds the accumulator out to a DELIVERY type. Passing the
/// accumulator itself is a no-op for configuration 1 and so appears to work,
/// but for `fma_accumulator<T>` it yields an `fma_accumulator<T>` and for a
/// super-accumulator a quire -- neither of which has a `sqrt` (#324).
///
/// Rounding straight to the magnitude type compiles everywhere, but throws away
/// the precision the accumulator was chosen for: the sum would be narrowed to
/// the element magnitude BEFORE the square root. So prefer the accumulator's own
/// arithmetic precision where one is nameable:
///
///   * configuration 1 (arithmetic `Acc`)   -> `Acc` itself
///   * configuration 2 (`fma_accumulator<T>`) -> `T`
///   * configuration 3 (custom/quire)       -> `Mag`, the documented delivery
///     type. A super-accumulator is exact, so this is the single final
///     round-out its contract already promises, and external specializations
///     need not provide anything new.
template <typename Acc, typename Mag>
struct accumulator_round_type {
    using type = std::conditional_t<std::is_arithmetic_v<Acc>, Acc, Mag>;
};

template <typename T, typename Mag>
struct accumulator_round_type<fma_accumulator<T>, Mag> {
    using type = T;
};

template <typename Acc, typename Mag>
using accumulator_round_type_t = typename accumulator_round_type<Acc, Mag>::type;

} // namespace mtl::math
