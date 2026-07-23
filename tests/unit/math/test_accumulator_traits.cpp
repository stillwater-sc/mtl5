// MTL5 -- mtl::math::accumulator_traits (issue #158).
// The shared accumulator policy: clear/assign/add_product, plus the generalized
// value<Result> round-out that delivers the accumulator in a result type distinct
// from both the element type and the accumulator type (the Element -> Accumulate
// -> Result model behind mixed-precision tensor ops).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include <mtl/math/accumulator_traits.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>   // for the inheriting sparse alias

namespace {
// Extended accumulator: hold the running sum in double, accumulate products in
// double, and deliver in any result type via value<Result>.
struct wide_acc { double v = 0.0; };

// A minimal custom arithmetic type with an ADL-found fma, standing in for a
// posit or other Universal number type. fma_accumulator<T> must route through
// THIS fma (not std::fma), which is why its T is not constrained to
// std::floating_point.
int g_adl_fma_calls = 0;
struct adl_real {
    double x = 0.0;
    adl_real() = default;
    adl_real(double d) : x(d) {}                     // static_cast<adl_real>(float)
    explicit operator double() const { return x; }   // value<double>()
    explicit operator float()  const { return static_cast<float>(x); }
};
adl_real fma(adl_real a, adl_real b, adl_real c) {
    ++g_adl_fma_calls;
    return adl_real{std::fma(a.x, b.x, c.x)};
}
} // namespace

namespace mtl::math {
template <>
struct accumulator_traits<wide_acc, float> {
    static void clear(wide_acc& a) { a.v = 0.0; }
    static void assign(wide_acc& a, const float& x) { a.v = static_cast<double>(x); }
    template <typename Result = float>
    static Result value(const wide_acc& a) { return static_cast<Result>(a.v); }
    static void add_product(wide_acc& a, const float& m, const float& x) {
        a.v += static_cast<double>(m) * static_cast<double>(x);   // product in double
    }
};
} // namespace mtl::math

using mtl::math::accumulator_traits;

TEST_CASE("accumulator_traits default: plain arithmetic", "[math][accumulator]") {
    using AT = accumulator_traits<double, double>;
    double a;
    AT::clear(a);            REQUIRE(a == 0.0);
    AT::assign(a, 1.5);      REQUIRE(a == 1.5);
    AT::add_product(a, 2.0, 3.0);  REQUIRE(a == 7.5);   // 1.5 + 6
    REQUIRE(AT::value(a) == 7.5);
}

TEST_CASE("accumulator_traits value<Result> rounds to a distinct result type",
          "[math][accumulator]") {
    // Accumulate in double (Acc), elements are float (Value), deliver as float
    // (Result default = Value) or as double (explicit Result) -- the fused
    // accumulate->output conversion.
    using AT = accumulator_traits<double, float>;
    double a; AT::clear(a);
    AT::assign(a, 1.0f);
    AT::add_product(a, 0.1f, 1.0f);   // a += 0.1f

    float  rf = AT::value(a);             // Result defaults to Value (float)
    double rd = AT::value<double>(a);     // explicit wider Result
    REQUIRE(rf == static_cast<float>(a));
    REQUIRE(rd == a);
}

TEST_CASE("accumulator_traits: extended accumulator beats element-precision sum",
          "[math][accumulator]") {
    // Sum many float products. A double accumulator (with double products)
    // rounded once to float is more accurate than summing in float throughout.
    const std::size_t n = 4096;
    std::vector<float> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) { x[i] = 1.0f + 1e-3f * static_cast<float>(i % 7);
                                          y[i] = 1.0f - 1e-3f * static_cast<float>(i % 5); }

    // exact-ish reference in double
    double ref = 0.0;
    for (std::size_t i = 0; i < n; ++i) ref += static_cast<double>(x[i]) * static_cast<double>(y[i]);

    float naive = 0.0f;
    for (std::size_t i = 0; i < n; ++i) naive += x[i] * y[i];   // float accumulate

    using ATW = accumulator_traits<wide_acc, float>;
    wide_acc w; ATW::clear(w);
    for (std::size_t i = 0; i < n; ++i) ATW::add_product(w, x[i], y[i]);
    float wide = ATW::value(w);                                  // round once to float

    double err_naive = std::abs(static_cast<double>(naive) - ref);
    double err_wide  = std::abs(static_cast<double>(wide)  - ref);
    INFO("naive err = " << err_naive << ", wide err = " << err_wide);
    REQUIRE(err_wide <= err_naive);
}

TEST_CASE("sparse accumulator_traits inherits the canonical mtl::math trait",
          "[math][accumulator][sparse]") {
    // The sparse alias forwards to mtl::math, including value<Result>.
    using SAT = mtl::sparse::factorization::accumulator_traits<double, float>;
    double a; SAT::clear(a); SAT::assign(a, 2.0f);
    REQUIRE(SAT::value(a) == 2.0f);
    REQUIRE(SAT::value<double>(a) == 2.0);
}

TEST_CASE("accumulator_traits configuration 2: FMA accumulator basics",
          "[math][accumulator][fma]") {
    // fma_accumulator<T> selects the fused reduction: sum = fma(m, v, sum).
    using Acc = mtl::math::fma_accumulator<double>;
    using AT = accumulator_traits<Acc, double>;
    Acc a;
    AT::clear(a);                 REQUIRE(a.sum == 0.0);
    AT::assign(a, 1.5);           REQUIRE(a.sum == 1.5);
    AT::add_product(a, 2.0, 3.0); REQUIRE(a.sum == 7.5);   // fma(2,3,1.5) = 7.5
    REQUIRE(AT::value(a) == 7.5);

    // value<Result> genuinely rounds to Result: pick a double accumulator value
    // that is not representable in float and check it rounds to the adjacent float.
    Acc r; AT::clear(r);
    AT::assign(r, 1.0);
    AT::add_product(r, 1.0, 0x1p-30);          // 1 + 2^-30, exact in double...
    REQUIRE(r.sum == 1.0 + 0x1p-30);
    REQUIRE(AT::value<float>(r) == 1.0f);      // ...but rounds to 1.0f in float
    REQUIRE(AT::value<double>(r) == 1.0 + 0x1p-30);   // exact when Result == Acc
}

TEST_CASE("accumulator_traits configuration 2: FMA avoids the product rounding event",
          "[math][accumulator][fma]") {
    // The two-product identity: for representable a, b the exact product a*b
    // splits as round(a*b) + err, where err is the (representable) rounding error
    // of the product. A separately-rounded product loses err (it rounds a*b to
    // round(a*b) before the add); the fused std::fma keeps it (it forms a*b + c
    // with a single rounding). Configuration 2 guarantees the fused behavior.
    //
    // With T = float (ulp 2^-23 on [1,2)):
    //   a = b = 1 + 2^-13
    //   a*b = 1 + 2^-12 + 2^-26   (exact)
    //   round(a*b) = 1 + 2^-12    (the 2^-26 tail is below half-ulp, rounds away)
    //   err = a*b - round(a*b) = 2^-26   (representable)
    const float a = 1.0f + 0x1p-13f;
    const float b = 1.0f + 0x1p-13f;
    const float rounded_prod = a * b;               // one rounding: 1 + 2^-12
    const float c = -rounded_prod;                  // cancel the rounded product
    const float exact_err = 0x1p-26f;               // a*b - round(a*b)

    REQUIRE(rounded_prod == 1.0f + 0x1p-12f);        // premise: the tail rounds away

    // A separately-rounded product (two roundings, computed here in two distinct
    // statements so no compiler may contract them into an FMA): the tail is lost.
    // This is the accuracy configuration 2 exists to avoid -- note that whether
    // configuration 1's own `a += m*v` is contracted into an FMA is left to the
    // compiler's FP-contraction setting, which is exactly why an *explicit* fused
    // accumulator is offered as a distinct, guaranteed policy.
    const float separately_rounded = rounded_prod + c;
    REQUIRE(separately_rounded == 0.0f);            // product error rounded away

    // Configuration 2: FMA accumulator in float -- guaranteed single rounding.
    using Acc2 = mtl::math::fma_accumulator<float>;
    using AT2 = accumulator_traits<Acc2, float>;
    Acc2 p2; AT2::clear(p2); AT2::assign(p2, c);    // start at c
    AT2::add_product(p2, a, b);                     // fma(a, b, c), one rounding
    const float fused = AT2::value(p2);

    REQUIRE(fused == exact_err);                    // product error kept via FMA
}

TEST_CASE("accumulator_traits configuration 2: fma_accumulator routes through an ADL fma",
          "[math][accumulator][fma]") {
    // fma_accumulator<T> is intentionally unconstrained so custom arithmetic
    // types (posits, etc.) can plug in a type-specific fused multiply-add. The
    // `using std::fma; fma(...)` step must select the ADL-found custom fma, not
    // std::fma. adl_real records each call to prove the dispatch.
    using Acc = mtl::math::fma_accumulator<adl_real>;
    using AT = accumulator_traits<Acc, float>;
    g_adl_fma_calls = 0;

    Acc a;
    AT::clear(a);
    AT::assign(a, 1.5f);
    AT::add_product(a, 2.0f, 3.0f);                  // must call adl_real's fma
    AT::add_product(a, 1.0f, 1.0f);

    REQUIRE(g_adl_fma_calls == 2);                   // the custom fma was used
    REQUIRE(AT::value<double>(a) == 8.5);            // 1.5 + 2*3 + 1*1
}
