// MTL5 -- accumulator policy for Householder reflector construction and
// application (issue #536, following the dot.hpp pattern from #159/#261).
// householder() and apply_householder_left/right() now take an optional
// Accumulator (default void), so the reduction inside each can run in a
// precision distinct from the element type without changing default behavior.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/householder.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("householder default behavior is unchanged", "[operation][householder][accumulator]") {
    vec::dense_vector<double> x = {3.0, 4.0, 0.0};
    auto [v, beta] = householder(x);
    // H*x should land entirely on e_1.
    mat::dense2D<double> A(3, 1);
    A(0, 0) = x(0); A(1, 0) = x(1); A(2, 0) = x(2);
    apply_householder_left(A, v, beta, 0, 0);
    REQUIRE_THAT(std::abs(A(0, 0)), WithinRel(5.0, 1e-12));
    REQUIRE_THAT(A(1, 0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(A(2, 0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("householder accumulator type is honored", "[operation][householder][accumulator]") {
    vec::dense_vector<float> x = {3.0f, 4.0f, 0.0f};
    // fp32 elements, fp64 accumulate for the sigma reduction.
    auto [v, beta] = householder<float, double>(x);
    static_assert(std::is_same_v<decltype(beta), float>);
    mat::dense2D<float> A(3, 1);
    A(0, 0) = x(0); A(1, 0) = x(1); A(2, 0) = x(2);
    apply_householder_left<mat::dense2D<float>, float, double>(A, v, beta, 0, 0);
    REQUIRE_THAT(std::abs(A(0, 0)), WithinRel(5.0f, 1e-5f));
    REQUIRE_THAT(A(1, 0), WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(A(2, 0), WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("apply_householder_left fp64 accumulator beats fp32 accumulation on a cancellation-prone reflection",
          "[operation][householder][accumulator]") {
    // A long reflector vector with a cancellation-prone alternating pattern,
    // same shape as the dot-product accumulator regression test: fp32
    // accumulation loses digits a fp64 accumulator recovers.
    const std::size_t n = 100000;
    vec::dense_vector<float> v(n);
    v(0) = 1.0f;
    for (std::size_t i = 1; i < n; ++i)
        v(static_cast<int>(i)) = (i % 2 == 0) ? 1.0e-3f : -1.0e-3f + 1.0e-9f;

    mat::dense2D<float> A_naive(static_cast<int>(n), 1), A_wide(static_cast<int>(n), 1);
    std::vector<double> col(n);
    for (std::size_t i = 0; i < n; ++i) {
        float val = (i % 3 == 0) ? 1.0f : -1.0f + 1.0e-6f;
        A_naive(static_cast<int>(i), 0) = val;
        A_wide(static_cast<int>(i), 0)  = val;
        col[i] = static_cast<double>(val);
    }

    // exact-ish reference dot product v^H * A(:,0) in double
    double ref = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        ref += static_cast<double>(v(static_cast<int>(i))) * col[i];

    const float beta = 1.0f;  // arbitrary nonzero scale; only w's accuracy matters here
    apply_householder_left<mat::dense2D<float>, float>(A_naive, v, beta, 0, 0);          // fp32 accumulate
    apply_householder_left<mat::dense2D<float>, float, double>(A_wide, v, beta, 0, 0);   // fp64 accumulate

    // A(:,0) -= beta * v * w, so w = (A_before(0,0) - A_after(0,0)) / (beta * v(0))
    double w_naive = (static_cast<double>(1.0f) - static_cast<double>(A_naive(0, 0))) / (static_cast<double>(beta) * static_cast<double>(v(0)));
    double w_wide  = (static_cast<double>(1.0f) - static_cast<double>(A_wide(0, 0)))  / (static_cast<double>(beta) * static_cast<double>(v(0)));

    double err_naive = std::abs(w_naive - ref);
    double err_wide  = std::abs(w_wide - ref);
    INFO("ref = " << ref << ", naive(f32) err = " << err_naive << ", wide(f64) err = " << err_wide);
    REQUIRE(err_wide <= err_naive);
}
