// MTL5 -- accumulator policy for symm (#261, Part C, BLAS L3).
// symm sums each (row, col) reduction A(i,k)*B(k,j) over k, seeded with
// `clear` -- a zero-seeded reduction, same shape as symv's row loop extended
// to a second free index (B's columns) instead of a single vector -- then
// combines with the caller's alpha/beta once, outside the reduction. Default
// Accumulator = void keeps the BLAS/generic dispatch byte for byte; a
// non-default accumulator forces the native path.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

namespace {

/// Counts contract operations, mirrors symv's counting_acc (#515 pattern).
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

TEST_CASE("symm default behavior is unchanged", "[operation][symm][accumulator]") {
    mat::dense2D<double> A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=2; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=2;
    mat::dense2D<double> B(3, 1);
    B(0,0)=1; B(1,0)=1; B(2,0)=1;
    mat::dense2D<double> C(3, 1);
    C(0,0)=1; C(1,0)=1; C(2,0)=1;
    symm(2.0, A, B, 0.5, C);
    // C = 2*A*B + 0.5*C ; A*B = {3,4,3}
    REQUIRE_THAT(C(0,0), WithinRel(6.5, 1e-12));
    REQUIRE_THAT(C(1,0), WithinRel(8.5, 1e-12));
    REQUIRE_THAT(C(2,0), WithinRel(6.5, 1e-12));
}

TEST_CASE("symm handles multiple columns of B independently",
          "[operation][symm][accumulator]") {
    mat::dense2D<double> A(2, 2);
    A(0,0)=1; A(0,1)=1;
    A(1,0)=1; A(1,1)=1;
    mat::dense2D<double> B(2, 2);
    B(0,0)=2; B(0,1)=1;
    B(1,0)=3; B(1,1)=1;
    mat::dense2D<double> C(2, 2);
    C(0,0)=0; C(0,1)=0;
    C(1,0)=0; C(1,1)=0;
    symm(1.0, A, B, 0.0, C);
    // col0: row0: 1*2+1*3=5, row1: 1*2+1*3=5
    // col1: row0: 1*1+1*1=2, row1: 1*1+1*1=2
    REQUIRE_THAT(C(0,0), WithinRel(5.0, 1e-12));
    REQUIRE_THAT(C(1,0), WithinRel(5.0, 1e-12));
    REQUIRE_THAT(C(0,1), WithinRel(2.0, 1e-12));
    REQUIRE_THAT(C(1,1), WithinRel(2.0, 1e-12));
}

TEST_CASE("symm drives clear, not assign -- it is a zero-seeded reduction",
          "[operation][symm][accumulator]") {
    mat::dense2D<double> A(2, 2);
    A(0,0)=1; A(0,1)=1; A(1,0)=1; A(1,1)=1;
    mat::dense2D<double> B(2, 1);
    B(0,0)=2; B(1,0)=3;
    mat::dense2D<double> C(2, 1);
    C(0,0)=10; C(1,0)=10;

    counting_acc::reset();
    symm<counting_acc>(1.0, A, B, 0.0, C);

    const std::size_t m = 2, n = 1, k = 2;
    REQUIRE(counting_acc::clears   == m * n);
    REQUIRE(counting_acc::products == m * n * k);
    REQUIRE(counting_acc::values   == m * n);
    REQUIRE(counting_acc::assigns  == 0);   // NOT seeded from C(i,j)
    REQUIRE(C(0,0) == 5.0);   // 1*2 + 1*3
    REQUIRE(C(1,0) == 5.0);
}

TEST_CASE("symm fp64 accumulator beats fp32 on a near-cancelling row",
          "[operation][symm][accumulator]") {
    const std::size_t n = 2000;
    mat::dense2D<float> A(n, n);
    mat::dense2D<float> B(n, 1);
    for (std::size_t j = 0; j < n; ++j) B(j, 0) = 1.0f;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j) {
            float v = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;
            A(i, j) = v;
            A(j, i) = v;   // keep A symmetric
        }

    double ref = 0.0;
    for (std::size_t j = 0; j < n; ++j) ref += static_cast<double>(A(0, j));

    mat::dense2D<float> C_naive(n, 1), C_wide(n, 1);
    for (std::size_t i = 0; i < n; ++i) { C_naive(i,0) = 0.0f; C_wide(i,0) = 0.0f; }
    symm(1.0f, A, B, 0.0f, C_naive);          // fp32 accumulate
    symm<double>(1.0, A, B, 0.0, C_wide);     // fp64 accumulate, fp32 result

    double e_naive = std::abs(static_cast<double>(C_naive(0,0)) - ref);
    double e_wide  = std::abs(static_cast<double>(C_wide(0,0))  - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("symm accumulator/result types are honored", "[operation][symm][accumulator]") {
    mat::dense2D<float> A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=2; A(1,1)=1;
    mat::dense2D<float> B(2, 1);
    B(0,0)=1.0f; B(1,0)=1.0f;
    mat::dense2D<float> C_wide(2, 1);
    C_wide(0,0)=0.0f; C_wide(1,0)=0.0f;
    symm<double>(1.0f, A, B, 0.0f, C_wide);
    REQUIRE_THAT(C_wide(0,0), WithinRel(3.0f, 1e-6f));
    REQUIRE_THAT(C_wide(1,0), WithinRel(3.0f, 1e-6f));
}
