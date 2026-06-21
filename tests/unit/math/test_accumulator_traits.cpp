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
