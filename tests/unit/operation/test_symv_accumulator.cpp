// MTL5 -- accumulator policy for symv (#261, Part C, BLAS L2).
// symv sums each row's A(i,j)*x(j) products, seeded with `clear` -- a zero-seeded
// reduction, same shape as dot's -- then combines with the caller's alpha/beta
// once, outside the reduction. Default Accumulator = void keeps the BLAS/generic
// dispatch byte for byte; a non-default accumulator forces the native path.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/symv.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

namespace {

/// Counts contract operations, so a test can assert WHICH primitives symv
/// drives, mirroring the axpy/scale counting_acc pattern (#511).
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

TEST_CASE("symv default behavior is unchanged", "[operation][symv][accumulator]") {
    mat::dense2D<double> A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=2; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=2;
    vec::dense_vector<double> x = {1.0, 1.0, 1.0}, y = {1.0, 1.0, 1.0};
    symv(2.0, A, x, 0.5, y);
    // y = 2*A*x + 0.5*y ; A*x = {3,4,3}
    REQUIRE_THAT(y(0), WithinRel(6.5, 1e-12));
    REQUIRE_THAT(y(1), WithinRel(8.5, 1e-12));
    REQUIRE_THAT(y(2), WithinRel(6.5, 1e-12));
}

TEST_CASE("symv drives clear, not assign -- it is a zero-seeded row reduction",
          "[operation][symv][accumulator]") {
    // Mirrors scale's clear-seeded case, not axpy's assign-seeded one: symv sums
    // A(i,j)*x(j) from zero, then combines with alpha/beta once, outside the loop.
    mat::dense2D<double> A(2, 2);
    A(0,0)=1; A(0,1)=1; A(1,0)=1; A(1,1)=1;
    vec::dense_vector<double> x = {2.0, 3.0}, y = {10.0, 10.0};

    counting_acc::reset();
    symv<counting_acc>(1.0, A, x, 0.0, y);

    const int n = 2;
    REQUIRE(counting_acc::clears   == n);          // one clear per row
    REQUIRE(counting_acc::products == n * n);       // one add_product per entry
    REQUIRE(counting_acc::values   == n);          // one round-out per row
    REQUIRE(counting_acc::assigns  == 0);          // NOT seeded from y(i)
    REQUIRE(y(0) == 5.0);   // 1*2 + 1*3
    REQUIRE(y(1) == 5.0);
}

TEST_CASE("symv fp64 accumulator beats fp32 on a near-cancelling row",
          "[operation][symv][accumulator]") {
    // Symmetric by construction; off-diagonal magnitude chosen so each row sum
    // nearly cancels in fp32, same premise as dot's cancellation-prone case.
    const std::size_t n = 2000;
    mat::dense2D<float> A(n, n);
    vec::dense_vector<float> x(n);
    for (std::size_t j = 0; j < n; ++j) x(static_cast<int>(j)) = 1.0f;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j) {
            float v = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;
            A(i, j) = v;
            A(j, i) = v;   // keep A symmetric
        }

    double ref = 0.0;
    for (std::size_t j = 0; j < n; ++j) ref += static_cast<double>(A(0, j));

    vec::dense_vector<float> y_naive(n, 0.0f), y_wide(n, 0.0f);
    symv(1.0f, A, x, 0.0f, y_naive);           // fp32 accumulate
    symv<double>(1.0, A, x, 0.0, y_wide);      // fp64 accumulate, fp32 result

    double e_naive = std::abs(static_cast<double>(y_naive(0)) - ref);
    double e_wide  = std::abs(static_cast<double>(y_wide(0))  - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("symv accumulator/result types are honored", "[operation][symv][accumulator]") {
    mat::dense2D<float> A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=2; A(1,1)=1;
    vec::dense_vector<float> x = {1.0f, 1.0f};
    vec::dense_vector<float> y_wide(2, 0.0f);
    symv<double>(1.0f, A, x, 0.0f, y_wide);
    REQUIRE_THAT(y_wide(0), WithinRel(3.0f, 1e-6f));
    REQUIRE_THAT(y_wide(1), WithinRel(3.0f, 1e-6f));
}
