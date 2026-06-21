// MTL5 -- mixed-precision accumulator/result policy for dot (issue #159).
// dot<Accumulator,Result>(a,b) sums the products in Accumulator precision (which
// may differ from the element type) and rounds out to Result -- the dot-product
// instance of the Element -> Accumulate -> Result model behind the BLAS epic #157.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

TEST_CASE("dot default behavior is unchanged", "[operation][dot][accumulator]") {
    vec::dense_vector<double> a = {1.0, 2.0, 3.0}, b = {4.0, 5.0, 6.0};
    REQUIRE_THAT(dot(a, b), WithinRel(32.0, 1e-12));      // 4+10+18
    REQUIRE_THAT(dot_real(a, b), WithinRel(32.0, 1e-12));
}

TEST_CASE("dot accumulator/result types are honored", "[operation][dot][accumulator]") {
    vec::dense_vector<float> a = {1.0f, 2.0f, 3.0f}, b = {4.0f, 5.0f, 6.0f};
    // fp32 elements, fp64 accumulate -> result defaults to the accumulator type.
    auto s = dot<double>(a, b);
    static_assert(std::is_same_v<decltype(s), double>);
    REQUIRE_THAT(s, WithinRel(32.0, 1e-12));
    // Explicit result type rounds the accumulator out to float.
    auto sf = dot<double, float>(a, b);
    static_assert(std::is_same_v<decltype(sf), float>);
    REQUIRE_THAT(sf, WithinRel(32.0f, 1e-6f));
}

TEST_CASE("dot fp64 accumulator beats fp32 accumulation on a cancellation-prone sum",
          "[operation][dot][accumulator]") {
    // A long sum with a small true value relative to the partial sums: float
    // accumulation loses digits a double accumulator recovers.
    const std::size_t n = 100000;
    vec::dense_vector<float> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        a(static_cast<int>(i)) = 1.0f;
        b(static_cast<int>(i)) = (i % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;  // ~cancels
    }
    // exact-ish reference in double
    double ref = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        ref += static_cast<double>(a(static_cast<int>(i))) * static_cast<double>(b(static_cast<int>(i)));

    float  naive = dot<float>(a, b);     // fp32 accumulate (via the policy path)
    double wide  = dot<double>(a, b);    // fp64 accumulate

    double err_naive = std::abs(static_cast<double>(naive) - ref);
    double err_wide  = std::abs(wide - ref);
    INFO("ref = " << ref << ", naive(f32) err = " << err_naive << ", wide(f64) err = " << err_wide);
    REQUIRE(err_wide <= err_naive);
    REQUIRE(err_wide < 1e-6);
}
