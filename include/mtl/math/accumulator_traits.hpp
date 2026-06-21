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
// MTL5 stays free of any external number library: a caller supplies a custom
// `Acc` (a compensated/Kahan accumulator, a Universal `quire` super-accumulator,
// or simply a wider IEEE type) by specializing `accumulator_traits<Acc, Value>`.
//
// The default specialization makes `Acc == Value` (plain arithmetic, zero
// overhead, identical results) -- the behavior unless a caller opts in.
//
// Used by the sparse factorizations and the dense BLAS-level operations.

namespace mtl::math {

/// Accumulator policy for accumulating products of `Value`s into an `Acc`.
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
    /// subtraction (e.g. the LU elimination update).
    static void add_product(Acc& a, const Value& m, const Value& v) { a += m * v; }
};

} // namespace mtl::math
