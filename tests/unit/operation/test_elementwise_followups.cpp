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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

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

/// A posit-like real of range +-64: every conversion saturates to +-64, so
/// sat64(127) is 64, and 64 is the largest value.
struct sat64 {
    double v = 0.0;
    sat64() = default;
    constexpr sat64(double d) : v(d != d ? d : (d > 64 ? 64 : (d < -64 ? -64 : d))) {}
    template <typename I>
        requires std::is_integral_v<I>
    constexpr sat64(I i) : sat64(static_cast<double>(i)) {}
    explicit constexpr operator long double() const { return v; }
    template <typename I>
        requires std::is_integral_v<I>
    explicit constexpr operator I() const { return static_cast<I>(v); }
    friend constexpr bool operator==(sat64 a, sat64 b) { return a.v == b.v; }
    friend constexpr auto operator<=>(sat64 a, sat64 b) { return a.v <=> b.v; }
};

/// A collection whose iterator returns a proxy that reads the element
/// through a pointer (the Collection concept allows it).
struct proxy_collection {
    using value_type = double;
    using size_type = std::size_t;
    std::vector<double> data;
    struct proxy {
        const double* p;
        operator double() const { return *p; }
    };
    struct iterator {
        const double* p;
        proxy operator*() const { return {p}; }
        iterator& operator++() { ++p; return *this; }
        bool operator!=(const iterator& o) const { return p != o.p; }
        bool operator==(const iterator& o) const { return p == o.p; }
    };
    size_type size() const { return data.size(); }
    iterator begin() const { return {data.data()}; }
    iterator end() const { return {data.data() + data.size()}; }
};

} // namespace

template <> class std::numeric_limits<sat64> : public mtl5_test::limits_of<sat64, double> {
public:
    static constexpr bool has_infinity = false;
    static constexpr sat64 max() noexcept { return sat64(64.0); }
    static constexpr sat64 lowest() noexcept { return sat64(-64.0); }
};

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
    // diag(1, 0): an exactly zero singular value in every SVD, LAPACK's too
    // (a rank-1 [[1,2],[2,4]] gives sigma_min ~ 1e-16 there, and a finite
    // condition number).
    const mat::dense2D<double> Z = {{1, 0}, {0, 0}};
    CHECK(condition_number(Z) == inf);
    // Without infinity, numeric_limits::infinity() is 0, the opposite of
    // infinite: the largest finite value now.
    const mat::dense2D<no_inf> N = {{no_inf(1), no_inf(0)}, {no_inf(0), no_inf(0)}};
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

// A non-integer number type with no conversion to long double (only explicit
// ones to the integral types): saturating_cast's branch that cannot judge
// Value's range in long double, and so clamps on both sides (issue 909: the
// sweep found it unreached).
namespace {
struct int_only {
    long long v;   // in hundredths: int_only(3) holds 300
    constexpr int_only() : v(0) {}
    constexpr int_only(long long i) : v(i * 100) {}
    static constexpr int_only raw(long long r) { int_only x; x.v = r; return x; }
    template <std::integral I>
    explicit constexpr operator I() const { return static_cast<I>(v / 100); }
    friend constexpr bool operator==(int_only a, int_only b) { return a.v == b.v; }
    friend constexpr bool operator!=(int_only a, int_only b) { return a.v != b.v; }
    friend constexpr bool operator<=(int_only a, int_only b) { return a.v <= b.v; }
    friend constexpr bool operator>=(int_only a, int_only b) { return a.v >= b.v; }
};
template <typename T>
concept converts_to_long_double = requires(const T& t) { static_cast<long double>(t); };
} // namespace
template <> struct std::numeric_limits<int_only> {
    static constexpr bool is_specialized = true;
    static constexpr bool is_integer = false;
    static constexpr int_only max() noexcept { return int_only::raw(std::numeric_limits<long long>::max()); }
    static constexpr int_only lowest() noexcept { return int_only::raw(std::numeric_limits<long long>::lowest()); }
    static constexpr int_only min() noexcept { return int_only::raw(1); }
};

TEST_CASE("saturating_cast from a type without a long double conversion (issue 909)", "[operation][projection][909]") {
    static_assert(!converts_to_long_double<int_only>);
    CHECK(detail::saturating_cast<std::int8_t>(int_only(1000)) == 127);
    CHECK(detail::saturating_cast<std::int8_t>(int_only(-1000)) == -128);
    CHECK(detail::saturating_cast<std::int8_t>(int_only(42)) == 42);
    CHECK(detail::saturating_cast<std::int8_t>(int_only(-7)) == -7);
}

// fused_assign and fused_plus_assign on a target whose element storage can't
// be seen (no data()/size()), and the vector overload (issue 909): the matrix
// forms take the temporary, assumed read, and the results are exact.
namespace {
struct grid {   // a matrix target with no visible storage
    using value_type = double;
    using size_type = std::size_t;
    std::size_t r = 0, c = 0;
    std::vector<double> cells;
    grid(std::size_t rr, std::size_t cc) : r(rr), c(cc), cells(rr * cc, 0.0) {}
    std::size_t num_rows() const { return r; }
    std::size_t num_cols() const { return c; }
    void change_dim(std::size_t rr, std::size_t cc) { r = rr; c = cc; cells.assign(rr * cc, 0.0); }
    double& operator()(std::size_t i, std::size_t j) { return cells[i * c + j]; }
    double operator()(std::size_t i, std::size_t j) const { return cells[i * c + j]; }
};
} // namespace

TEST_CASE("fused_assign on a target without visible storage, and on vectors (issue 909)", "[expr][fuse][909]") {
    static_assert(!detail::contiguous_storage<grid>);
    using M = mat::dense2D<double>;
    const M A = {{1, 2}, {3, 4}};
    const M B = {{10, 20}, {30, 40}};
    grid G(2, 2);
    fused_assign(G, A + B);
    CHECK(G(0, 0) == 11.0); CHECK(G(0, 1) == 22.0); CHECK(G(1, 0) == 33.0); CHECK(G(1, 1) == 44.0);
    fused_plus_assign(G, 2.0 * A);
    CHECK(G(0, 0) == 13.0); CHECK(G(1, 1) == 52.0);
    grid H(1, 1);                  // another shape: resized after the expression is read
    fused_assign(H, 1.0 * trans(B));
    REQUIRE(H.num_rows() == 2);
    CHECK(H(0, 1) == 30.0); CHECK(H(1, 0) == 20.0);

    const vec::dense_vector<double> x = {1.0, 2.0, 3.0};
    vec::dense_vector<double> y = {5.0, 6.0, 7.0};
    fused_assign(y, x + y);        // element i reads only element i: in place
    CHECK(y(0) == 6.0); CHECK(y(1) == 8.0); CHECK(y(2) == 10.0);
    vec::dense_vector<double> z(1);
    fused_assign(z, x * 2.0);      // resized to the expression's size
    REQUIRE(z.size() == 3);
    CHECK(z(2) == 6.0);
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

TEST_CASE("min and max return a value, not a proxy into the collection", "[operation][minmax][887]") {
    // With the accumulator typed as *it, result was a proxy: the returned
    // maximum changed when the element changed.
    proxy_collection c{{1.0, 2.0}};
    const auto mx = mtl::max(c);   // auto: what max returns, not a conversion of it
    const auto mn = mtl::min(c);
    STATIC_REQUIRE(std::is_same_v<std::remove_cv_t<decltype(mx)>, double>);
    c.data[1] = 100.0;
    c.data[0] = -100.0;
    CHECK(static_cast<double>(mx) == 2.0);
    CHECK(static_cast<double>(mn) == 1.0);
}

TEST_CASE("saturating_cast from a saturating source does not clamp its own range limit", "[operation][projection][887]") {
    // sat64(127) is 64, so the bound was 64 and sat64(64) came back as 127.
    CHECK(detail::saturating_cast<std::int8_t>(sat64(64.0)) == 64);
    CHECK(detail::saturating_cast<std::int8_t>(sat64(-64.0)) == -64);
    CHECK(detail::saturating_cast<std::int8_t>(sat64(12.0)) == 12);
    CHECK(detail::saturating_cast<std::int32_t>(sat64(64.0)) == 64);
}
