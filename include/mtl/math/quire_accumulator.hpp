#pragma once
// MTL5 -- accumulator_traits bridge to Universal's posit + quire (optional).
//
// Specializes mtl::math::accumulator_traits<Acc, Value> so that MTL5's
// mixed-precision dot()/mult()/norms() (see math/accumulator_traits.hpp,
// issue #158) can accumulate inner products in an exact quire and round
// once, instead of accumulating in posit arithmetic directly.
//
// MTL5 has no hard dependency on Universal: this header only compiles the
// specialization when the caller has already included Universal's posit/
// quire headers and defined MTL5_HAS_UNIVERSAL, mirroring the existing
// MTL5_HAS_BLAS pattern (see interface/dispatch_traits.hpp).
//
// Verified against the actual Universal source (github.com/stillwater-sc/
// universal, current `posit` module, not the legacy `posit1` module):
//   - quire_mul(posit<nbits,es,bt>, posit<nbits,es,bt>) returns a
//     blocktriple (include/sw/universal/number/posit/fdp.hpp)
//   - quire<NumberType, capacity, LimbType> is parameterized on the VALUE
//     type directly (capacity defaults from quire_traits<NumberType>), not
//     on nbits/es separately (include/sw/universal/number/quire/quire_impl.hpp)
//   - conversion back to a value is q.convert_to<TargetType>()
//
// Usage:
//   #define MTL5_HAS_UNIVERSAL
//   #include <universal/number/posit/posit.hpp>
//   #include <mtl/math/quire_accumulator.hpp>
//   ...
//   using Posit = sw::universal::posit<32,2>;
//   using Quire = sw::universal::quire<Posit>;   // capacity auto-derived
//   auto rho = mtl::dot<Quire>(r, z);   // accumulate rho in the quire, round once to Posit

#ifdef MTL5_HAS_UNIVERSAL

#include <mtl/math/accumulator_traits.hpp>

namespace mtl::math {

/// accumulator_traits specialization: Acc is a Universal quire parameterized
/// on posit<nbits,es,bt>, matching quire's real signature
/// quire<NumberType, capacity, LimbType>. add_product uses quire_mul's
/// exact (unrounded) blocktriple product; value() rounds out once at the
/// end via convert_to<Result>() (the single-rounding semantics the generic
/// template's docstring describes).
///
/// NOTE: posit's third template parameter `bt` (limb type) defaults to
/// std::uint8_t in Universal's posit_impl.hpp. This specialization pattern
/// binds to that default -- if your project overrides `bt`, extend the
/// pattern to include it explicitly.
template <unsigned nbits, unsigned es, unsigned capacity, typename LimbType>
struct accumulator_traits<sw::universal::quire<sw::universal::posit<nbits, es>, capacity, LimbType>,
                           sw::universal::posit<nbits, es>> {
    using Value = sw::universal::posit<nbits, es>;
    using Acc   = sw::universal::quire<Value, capacity, LimbType>;

    static void clear(Acc& a) { a.clear(); }

    static void assign(Acc& a, const Value& v) {
        a.clear();
        a += sw::universal::quire_mul(v, Value(1));
    }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        return a.template convert_to<Result>();
    }

    /// The core operation: a += m * v, accumulated exactly in the quire via
    /// Universal's quire_mul (returns an unrounded blocktriple product) --
    /// no rounding until value() is called.
    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += sw::universal::quire_mul(m, v);
    }
};

} // namespace mtl::math

#endif // MTL5_HAS_UNIVERSAL
