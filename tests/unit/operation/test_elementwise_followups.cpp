// Element-wise, reduction and conversion follow-ups of issue 887 (QA sweep
// #872):
//
//   - min/max seeded from the first element: infinities, NaN anywhere, and an
//     empty collection (it threw nothing and returned a finite sentinel);
//   - fused_assign/fused_plus_assign with an expression that reads the target
//     elsewhere (A = trans(A)), and with a target that must be resized;
//   - signum on unsigned types (it did not compile);
//   - condition_number of a singular matrix for a type without infinity;
//   - unscaled project_onto to an integer type: NaN and out-of-range values
//     saturate, from double and from a non-std floating type;
//   - normwise_backward_error's argument checks.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/backward_error.hpp>
#include <mtl/operation/convert.hpp>
#include <mtl/operation/fuse.hpp>
#include <mtl/operation/max.hpp>
#include <mtl/operation/min.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/projection.hpp>
#include <mtl/operation/signum.hpp>
#include <mtl/operation/spectral_properties.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl5_test/limits_of.hpp>

using namespace mtl;

namespace {

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double qnan = std::numeric_limits<double>::quiet_NaN();

/// A real without infinity, like a posit: a double underneath, with
/// numeric_limits saying has_infinity = false (so infinity() is 0).
struct no_inf {
    double v = 0.0;
    no_inf() = default;
    constexpr no_inf(double d) : v(d) {}
    explicit constexpr operator double() const { return v; }
    template <typename I>
        requires std::is_integral_v<I>
    explicit constexpr operator I() const { return static_cast<I>(v); }   // as a posit converts
    friend no_inf operator+(no_inf a, no_inf b) { return a.v + b.v; }
    friend no_inf operator-(no_inf a, no_inf b) { return a.v - b.v; }
    friend no_inf operator*(no_inf a, no_inf b) { return a.v * b.v; }
    friend no_inf operator/(no_inf a, no_inf b) { return a.v / b.v; }
    friend no_inf operator-(no_inf a) { return -a.v; }
    no_inf& operator+=(no_inf o) { v += o.v; return *this; }
    no_inf& operator-=(no_inf o) { v -= o.v; return *this; }
    no_inf& operator*=(no_inf o) { v *= o.v; return *this; }
    no_inf& operator/=(no_inf o) { v /= o.v; return *this; }
    friend bool operator==(no_inf a, no_inf b) { return a.v == b.v; }
    friend auto operator<=>(no_inf a, no_inf b) { return a.v <=> b.v; }
    friend no_inf sqrt(no_inf a) { return std::sqrt(a.v); }
    friend no_inf abs(no_inf a) { return std::abs(a.v); }
};

} // namespace

template <> class std::numeric_limits<no_inf> : public mtl5_test::limits_of<no_inf, double> {
public:
    static constexpr bool has_infinity = false;
    static constexpr no_inf infinity() noexcept { return no_inf(0.0); }
};

// -- min / max -------------------------------------------------------------

TEST_CASE("min and max: infinities, NaN and an empty collection (issue 887)", "[operation][minmax][887]") {
    // Seeded with numeric_limits<T>::max() / lowest(), min({+inf}) gave
    // 1.79769e+308 and max({-inf}) gave -1.79769e+308.
    CHECK(mtl::min(vec::dense_vector<double>{inf}) == inf);
    CHECK(mtl::max(vec::dense_vector<double>{-inf}) == -inf);
    CHECK(mtl::min(vec::dense_vector<double>{inf, -inf, 3.0}) == -inf);
    CHECK(mtl::max(vec::dense_vector<double>{inf, -inf, 3.0}) == inf);
    CHECK(mtl::min(vec::dense_vector<double>{4.0, 2.0, 7.0}) == 2.0);
    CHECK(mtl::max(vec::dense_vector<double>{4.0, 2.0, 7.0}) == 7.0);
    // A NaN anywhere makes the result NaN: a plain comparison skipped it, so
    // {1, NaN, 2} gave 1 or 2 depending on where the NaN sat.
    for (const auto& v : {vec::dense_vector<double>{qnan, 1.0, 2.0}, vec::dense_vector<double>{1.0, qnan, 2.0},
                          vec::dense_vector<double>{1.0, 2.0, qnan}}) {
        CHECK(std::isnan(mtl::min(v)));
        CHECK(std::isnan(mtl::max(v)));
    }
    // An empty collection has no minimum: it returned the sentinel silently.
    CHECK_THROWS_AS(mtl::min(vec::dense_vector<double>(0)), std::invalid_argument);
    CHECK_THROWS_AS(mtl::max(vec::dense_vector<double>(0)), std::invalid_argument);
    // Integers, and a matrix as a collection.
    CHECK(mtl::min(vec::dense_vector<int>{3, -7, 5}) == -7);
    CHECK(mtl::max(vec::dense_vector<std::uint8_t>{3, 200, 5}) == 200);
}

// -- fused_assign / fused_plus_assign -----------------------------------------

TEST_CASE("fused_assign and fused_plus_assign with an aliased operand (issue 887)", "[expr][fuse][887]") {
    using M = mat::dense2D<double>;
    {
        M A = {{1, 2}, {3, 4}};
        fused_assign(A, 1.0 * trans(A));   // was [[1,3],[3,4]]
        CHECK(A(0, 0) == 1.0); CHECK(A(0, 1) == 3.0);
        CHECK(A(1, 0) == 2.0); CHECK(A(1, 1) == 4.0);
    }
    {
        M C = {{1, 2}, {3, 4}};
        fused_plus_assign(C, trans(C) * 1.0);   // was [[2,5],[8,8]]
        CHECK(C(0, 0) == 2.0); CHECK(C(0, 1) == 5.0);
        CHECK(C(1, 0) == 5.0); CHECK(C(1, 1) == 8.0);
    }
    {
        // The target as an element-wise leaf is read only where it is
        // written: still exact in place.
        M A = {{1, 2}, {3, 4}};
        const M B = {{10, 20}, {30, 40}};
        fused_assign(A, A + B);
        CHECK(A(1, 0) == 33.0);
        fused_plus_assign(A, 2.0 * A);
        CHECK(A(1, 1) == 132.0);
    }
    {
        // A target of another shape is resized after the expression is read.
        M A = {{1, 2, 3}, {4, 5, 6}};      // 2 x 3
        fused_assign(A, 1.0 * trans(A));   // 3 x 2
        REQUIRE(A.num_rows() == 3);
        REQUIRE(A.num_cols() == 2);
        CHECK(A(0, 1) == 4.0); CHECK(A(2, 0) == 3.0); CHECK(A(2, 1) == 6.0);
    }
}

// -- signum, condition_number, convert ------------------------------------------

TEST_CASE("signum on unsigned element types (issue 887)", "[operation][signum][887]") {
    // T{-1} narrowed for an unsigned T, so this did not compile.
    const vec::dense_vector<unsigned> v = {0u, 3u, 7u};
    const auto s = signum(v);
    CHECK(s(0) == 0u); CHECK(s(1) == 1u); CHECK(s(2) == 1u);
    const vec::dense_vector<int> w = {-4, 0, 9};
    const auto t = signum(w);
    CHECK(t(0) == -1); CHECK(t(1) == 0); CHECK(t(2) == 1);
}

TEST_CASE("condition_number of a singular matrix (issue 887)", "[operation][spectral][887]") {
    const mat::dense2D<double> Z = {{1, 2}, {2, 4}};   // rank 1
    CHECK(condition_number(Z) == inf);
    // Without infinity, numeric_limits::infinity() is 0, the opposite of
    // infinite: the largest finite value now.
    const mat::dense2D<no_inf> N = {{no_inf(1), no_inf(2)}, {no_inf(2), no_inf(4)}};
    CHECK(double(condition_number(N)) == double(std::numeric_limits<no_inf>::max()));
}

TEST_CASE("convert copies every element, indexed by size_type (issue 887)", "[operation][convert][887]") {
    const vec::dense_vector<double> v = {1.5, -2.25, 3.0};
    vec::dense_vector<float> w(3);
    convert(v, w);
    CHECK(w(0) == 1.5f); CHECK(w(1) == -2.25f); CHECK(w(2) == 3.0f);
}

// -- project_onto ------------------------------------------------------------------

TEST_CASE("unscaled project_onto to an integer type saturates (issue 887)", "[operation][projection][887]") {
    // A bare static_cast of NaN or an out-of-range float is undefined.
    const vec::dense_vector<double> v = {qnan, 1e10, -1e10, 100.0, -100.0, inf, -inf};
    const auto p = project_onto<std::int8_t>(v);
    CHECK(p(0) == 0);
    CHECK(p(1) == 127);
    CHECK(p(2) == -128);
    CHECK(p(3) == 100);
    CHECK(p(4) == -100);
    CHECK(p(5) == 127);
    CHECK(p(6) == -128);
    mat::dense2D<double> A = {{1e10, qnan}, {-3.0, -1e10}};
    const auto q = project_onto<std::int16_t>(A);
    CHECK(q(0, 0) == 32767); CHECK(q(0, 1) == 0); CHECK(q(1, 0) == -3); CHECK(q(1, 1) == -32768);
}

TEST_CASE("saturating_cast clamps a non-std floating source (issue 887)", "[operation][projection][887]") {
    // Only std::is_floating_point sources were clamped, so a posit or
    // bfloat16 source was cast unclamped.
    CHECK(detail::saturating_cast<std::int8_t>(no_inf(1e10)) == 127);
    CHECK(detail::saturating_cast<std::int8_t>(no_inf(-1e10)) == -128);
    CHECK(detail::saturating_cast<std::int8_t>(no_inf(qnan)) == 0);
    CHECK(detail::saturating_cast<std::int8_t>(no_inf(42.0)) == 42);
}

// -- normwise_backward_error ---------------------------------------------------------

TEST_CASE("normwise_backward_error rejects mismatched shapes", "[operation][backward_error][887]") {
    const mat::dense2D<double> A = {{2, 0}, {0, 2}};
    const mat::dense2D<double> R = {{1, 2, 3}, {4, 5, 6}};
    const vec::dense_vector<double> x = {1, 1}, b = {2, 2}, x3 = {1, 1, 1};
    CHECK(normwise_backward_error(A, x, b) == 0.0);
    CHECK_THROWS_AS(normwise_backward_error(R, x3, b), std::invalid_argument);   // not square
    CHECK_THROWS_AS(normwise_backward_error(A, x3, b), std::invalid_argument);   // x too long
    CHECK_THROWS_AS(normwise_backward_error(A, x, x3), std::invalid_argument);   // b too long
}
