// MTL5 -- accumulator policy for trsv / lower_trisolve / upper_trisolve
// (#261, Part C, BLAS L2).
// Each row's Sigma L(i,j)*x(j) (or U(i,j)*x(j)) reduction is formed via
// clear/add_product/value, same shape as symv/trmv's row loop. The
// subtraction from b(i) and division by the diagonal happen AFTER the
// reduction is rounded out, outside the accumulator -- not seeded into it.
// Traversal order (forward for lower, reverse for upper) is a genuine data
// dependency (x(j) must already be SOLVED), unrelated to accumulator choice.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/operation/trsv.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/operators.hpp>

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

TEST_CASE("trsv default behavior is unchanged -- lower and upper",
          "[operation][trsv][accumulator]") {
    mat::dense2D<double> L(3, 3);
    L(0,0) = 2; L(0,1) = 0; L(0,2) = 0;
    L(1,0) = 1; L(1,1) = 3; L(1,2) = 0;
    L(2,0) = 4; L(2,1) = 2; L(2,2) = 5;
    vec::dense_vector<double> b = {2.0, 7.0, 26.0};
    vec::dense_vector<double> x(3);
    trsv(L, x, b, false);
    auto r = L * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), WithinAbs(b(i), 1e-10));

    mat::dense2D<double> U(3, 3);
    U(0,0) = 3; U(0,1) = 1; U(0,2) = 2;
    U(1,0) = 0; U(1,1) = 4; U(1,2) = 1;
    U(2,0) = 0; U(2,1) = 0; U(2,2) = 2;
    vec::dense_vector<double> bu = {10.0, 9.0, 4.0};
    vec::dense_vector<double> xu(3);
    trsv(U, xu, bu, true);
    auto ru = U * xu;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(ru(i), WithinAbs(bu(i), 1e-10));
}

TEST_CASE("lower_trisolve drives clear, not assign -- reduction excludes the diagonal solve",
          "[operation][trsv][accumulator]") {
    mat::dense2D<double> L(2, 2);
    L(0,0) = 2; L(0,1) = 0;
    L(1,0) = 1; L(1,1) = 3;
    vec::dense_vector<double> b = {4.0, 5.0};
    vec::dense_vector<double> x(2);

    counting_acc::reset();
    lower_trisolve<counting_acc>(L, x, b);

    const int n = 2;
    REQUIRE(counting_acc::clears   == n);
    REQUIRE(counting_acc::products == 0 + 1);
    REQUIRE(counting_acc::values   == n);
    REQUIRE(counting_acc::assigns  == 0);
    REQUIRE_THAT(x(0), WithinRel(2.0, 1e-12));
    REQUIRE_THAT(x(1), WithinRel(1.0, 1e-12));
}

TEST_CASE("trsv fp64 accumulator beats fp32 on a near-cancelling row",
          "[operation][trsv][accumulator]") {
    const std::size_t n = 2000;
    mat::dense2D<float> L(n, n);
    for (std::size_t i = 0; i < n; ++i) L(i, i) = 1.0f;
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            L(i, j) = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;

    vec::dense_vector<float> x_true(n);
    for (std::size_t j = 0; j < n; ++j) x_true(j) = 1.0f;
    vec::dense_vector<float> b(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j <= i; ++j)
            s += static_cast<double>(L(i, j)) * static_cast<double>(x_true(j));
        b(i) = static_cast<float>(s);
    }

    vec::dense_vector<float> x_naive(n), x_wide(n);
    lower_trisolve(L, x_naive, b);
    lower_trisolve<double>(L, x_wide, b);

    double e_naive = std::abs(static_cast<double>(x_naive(n - 1)) - 1.0);
    double e_wide  = std::abs(static_cast<double>(x_wide(n - 1))  - 1.0);
    INFO("naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("trsv accumulator/result types are honored", "[operation][trsv][accumulator]") {
    mat::dense2D<float> U(2, 2);
    U(0,0) = 2; U(0,1) = 1;
    U(1,0) = 0; U(1,1) = 2;
    vec::dense_vector<float> b = {4.0f, 4.0f};
    vec::dense_vector<float> x_wide(2);
    trsv<double>(U, x_wide, b, true);
    REQUIRE_THAT(x_wide(0), WithinRel(1.0f, 1e-6f));
    REQUIRE_THAT(x_wide(1), WithinRel(2.0f, 1e-6f));
}
