// MTL5 -- accumulator policy for trsm (#261, Part C, BLAS L3).
// Each row's Sigma A(i,j)*B(j,c) reduction over already-solved columns is
// formed via clear/add_product/value, same shape as trsv's row loop extended
// over B's columns. The subtraction from alpha*B(i,c) and division by the
// diagonal happen AFTER the reduction is rounded out, outside the
// accumulator -- not seeded into it. Traversal order (forward for lower,
// reverse for upper) is a genuine data dependency (B(j,c) must already be
// SOLVED), unrelated to accumulator choice.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/trsm.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

struct counting_acc {
    double v = 0.0;
    static inline int clears = 0, assigns = 0, products = 0, values = 0;
    static void reset() { clears = assigns = products = values = 0; }
};

} // namespace

namespace mtl::math {
template <typename Value>
struct accumulator_traits<::counting_acc, Value> {
    static void clear(::counting_acc& a) { a.v = 0.0; ++::counting_acc::clears; }
    static void assign(::counting_acc& a, const Value& x) {
        a.v = static_cast<double>(x); ++::counting_acc::assigns;
    }
    template <typename Result = Value>
    static Result value(const ::counting_acc& a) {
        ++::counting_acc::values; return static_cast<Result>(a.v);
    }
    static void add_product(::counting_acc& a, const Value& m, const Value& x) {
        a.v += static_cast<double>(m) * static_cast<double>(x);
        ++::counting_acc::products;
    }
};
} // namespace mtl::math

TEST_CASE("trsm default behavior is unchanged -- lower and upper",
          "[operation][trsm][accumulator]") {
    mat::dense2D<double> L(3, 3);
    L(0,0) = 2; L(0,1) = 0; L(0,2) = 0;
    L(1,0) = 1; L(1,1) = 3; L(1,2) = 0;
    L(2,0) = 4; L(2,1) = 2; L(2,2) = 5;
    mat::dense2D<double> B(3, 1);
    B(0,0) = 2.0; B(1,0) = 7.0; B(2,0) = 26.0;
    trsm(1.0, L, B, /*upper=*/false);
    // Verify L*X == original B by recomputing.
    double r0 = L(0,0)*B(0,0);
    double r1 = L(1,0)*B(0,0) + L(1,1)*B(1,0);
    double r2 = L(2,0)*B(0,0) + L(2,1)*B(1,0) + L(2,2)*B(2,0);
    REQUIRE_THAT(r0, WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(r1, WithinAbs(7.0, 1e-10));
    REQUIRE_THAT(r2, WithinAbs(26.0, 1e-10));

    mat::dense2D<double> U(3, 3);
    U(0,0) = 3; U(0,1) = 1; U(0,2) = 2;
    U(1,0) = 0; U(1,1) = 4; U(1,2) = 1;
    U(2,0) = 0; U(2,1) = 0; U(2,2) = 2;
    mat::dense2D<double> Bu(3, 1);
    Bu(0,0) = 10.0; Bu(1,0) = 9.0; Bu(2,0) = 4.0;
    trsm(1.0, U, Bu, /*upper=*/true);
    double ru0 = U(0,0)*Bu(0,0) + U(0,1)*Bu(1,0) + U(0,2)*Bu(2,0);
    double ru1 = U(1,1)*Bu(1,0) + U(1,2)*Bu(2,0);
    double ru2 = U(2,2)*Bu(2,0);
    REQUIRE_THAT(ru0, WithinAbs(10.0, 1e-10));
    REQUIRE_THAT(ru1, WithinAbs(9.0, 1e-10));
    REQUIRE_THAT(ru2, WithinAbs(4.0, 1e-10));
}

TEST_CASE("trsm handles multiple columns of B independently",
          "[operation][trsm][accumulator]") {
    mat::dense2D<double> L(2, 2);
    L(0,0) = 2; L(0,1) = 0;
    L(1,0) = 1; L(1,1) = 3;
    mat::dense2D<double> B(2, 2);
    B(0,0) = 4.0; B(0,1) = 6.0;
    B(1,0) = 5.0; B(1,1) = 10.0;
    trsm(1.0, L, B, /*upper=*/false);
    // col0: x0 = 4/2=2, x1=(5-1*2)/3=1
    // col1: x0 = 6/2=3, x1=(10-1*3)/3=7/3
    REQUIRE_THAT(B(0,0), WithinRel(2.0, 1e-12));
    REQUIRE_THAT(B(1,0), WithinRel(1.0, 1e-12));
    REQUIRE_THAT(B(0,1), WithinRel(3.0, 1e-12));
    REQUIRE_THAT(B(1,1), WithinRel(7.0/3.0, 1e-12));
}

TEST_CASE("trsm drives clear, not assign -- reduction excludes the diagonal solve",
          "[operation][trsm][accumulator]") {
    mat::dense2D<double> L(2, 2);
    L(0,0) = 2; L(0,1) = 0;
    L(1,0) = 1; L(1,1) = 3;
    mat::dense2D<double> B(2, 1);
    B(0,0) = 4.0; B(1,0) = 5.0;

    counting_acc::reset();
    trsm<counting_acc>(1.0, L, B, /*upper=*/false);

    const int m = 2, n = 1;
    REQUIRE(counting_acc::clears   == m * n);
    // row0: 0 products (no already-solved cols); row1: 1 product
    REQUIRE(counting_acc::products == (0 + 1) * n);
    REQUIRE(counting_acc::values   == m * n);
    REQUIRE(counting_acc::assigns  == 0);
    REQUIRE_THAT(B(0,0), WithinRel(2.0, 1e-12));
    REQUIRE_THAT(B(1,0), WithinRel(1.0, 1e-12));
}

TEST_CASE("trsm fp64 accumulator beats fp32 on a near-cancelling row",
          "[operation][trsm][accumulator]") {
    const std::size_t n = 2000;
    mat::dense2D<float> L(n, n);
    for (std::size_t i = 0; i < n; ++i) L(i, i) = 1.0f;
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            L(i, j) = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;

    mat::dense2D<float> X_true(n, 1);
    for (std::size_t j = 0; j < n; ++j) X_true(j, 0) = 1.0f;
    mat::dense2D<float> B(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j <= i; ++j)
            s += static_cast<double>(L(i, j)) * static_cast<double>(X_true(j, 0));
        B(i, 0) = static_cast<float>(s);
    }

    mat::dense2D<float> B_naive = B, B_wide = B;
    trsm(1.0f, L, B_naive, /*upper=*/false);
    trsm<double>(1.0, L, B_wide, /*upper=*/false);

    double e_naive = std::abs(static_cast<double>(B_naive(n - 1, 0)) - 1.0);
    double e_wide  = std::abs(static_cast<double>(B_wide(n - 1, 0))  - 1.0);
    INFO("naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("trsm accumulator/result types are honored", "[operation][trsm][accumulator]") {
    mat::dense2D<float> U(2, 2);
    U(0,0) = 2; U(0,1) = 1;
    U(1,0) = 0; U(1,1) = 2;
    mat::dense2D<float> B(2, 1);
    B(0,0) = 4.0f; B(1,0) = 4.0f;
    trsm<double>(1.0, U, B, /*upper=*/true);
    REQUIRE_THAT(B(0,0), WithinRel(1.0f, 1e-6f));
    REQUIRE_THAT(B(1,0), WithinRel(2.0f, 1e-6f));
}
