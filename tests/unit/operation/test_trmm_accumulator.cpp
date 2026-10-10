// MTL5 -- accumulator policy for trmm (#261, Part C, BLAS L3).
// trmm computes each (row, col) reduction over the triangular part of A, with
// the diagonal term (1*B(i,c) or A(i,i)*B(i,c)) fed through add_product as the
// first term of a clear-seeded reduction -- same shape as trmv's row loop
// extended over B's columns, not special-cased as a seed. Traversal order
// (forward for upper, reverse for lower) is an in-place-overwrite hazard,
// unrelated to accumulator choice.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/trmm.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

namespace {

/// Counts contract operations, mirrors trmv's counting_acc (#516 pattern).
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

TEST_CASE("trmm default behavior is unchanged -- upper, explicit diagonal",
          "[operation][trmm][accumulator]") {
    mat::dense2D<double> A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=1;
    A(1,0)=9; A(1,1)=2; A(1,2)=1;  // lower entries must be ignored
    A(2,0)=9; A(2,1)=9; A(2,2)=2;
    mat::dense2D<double> B(3, 1);
    B(0,0)=1; B(1,0)=1; B(2,0)=1;
    trmm(1.0, A, B, /*upper=*/true, /*unit_diag=*/false);
    // row0: 2*1 + 1*1 + 1*1 = 4
    // row1: 2*1 + 1*1       = 3
    // row2: 2*1             = 2
    REQUIRE_THAT(B(0,0), WithinRel(4.0, 1e-12));
    REQUIRE_THAT(B(1,0), WithinRel(3.0, 1e-12));
    REQUIRE_THAT(B(2,0), WithinRel(2.0, 1e-12));
}

TEST_CASE("trmm default behavior is unchanged -- lower, unit diagonal",
          "[operation][trmm][accumulator]") {
    mat::dense2D<double> A(3, 3);
    A(0,0)=9; A(0,1)=9; A(0,2)=9;  // upper entries must be ignored
    A(1,0)=1; A(1,1)=9; A(1,2)=9;
    A(2,0)=1; A(2,1)=1; A(2,2)=9;
    mat::dense2D<double> B(3, 1);
    B(0,0)=1; B(1,0)=1; B(2,0)=1;
    trmm(1.0, A, B, /*upper=*/false, /*unit_diag=*/true);
    REQUIRE_THAT(B(0,0), WithinRel(1.0, 1e-12));
    REQUIRE_THAT(B(1,0), WithinRel(2.0, 1e-12));
    REQUIRE_THAT(B(2,0), WithinRel(3.0, 1e-12));
}

TEST_CASE("trmm handles multiple columns of B independently",
          "[operation][trmm][accumulator]") {
    mat::dense2D<double> A(2, 2);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=9; A(1,1)=3;
    mat::dense2D<double> B(2, 2);
    B(0,0)=1; B(0,1)=2;
    B(1,0)=1; B(1,1)=1;
    trmm(1.0, A, B, /*upper=*/true, /*unit_diag=*/false);
    // col0: row0: 2*1+1*1=3, row1: 3*1=3
    // col1: row0: 2*2+1*1=5, row1: 3*1=3
    REQUIRE_THAT(B(0,0), WithinRel(3.0, 1e-12));
    REQUIRE_THAT(B(1,0), WithinRel(3.0, 1e-12));
    REQUIRE_THAT(B(0,1), WithinRel(5.0, 1e-12));
    REQUIRE_THAT(B(1,1), WithinRel(3.0, 1e-12));
}

TEST_CASE("trmm drives clear, not assign -- diagonal is fed via add_product, not seeded",
          "[operation][trmm][accumulator]") {
    mat::dense2D<double> A(2, 2);
    A(0,0)=1; A(0,1)=1; A(1,0)=1; A(1,1)=1;
    mat::dense2D<double> B(2, 1);
    B(0,0)=2; B(1,0)=3;

    counting_acc::reset();
    trmm<counting_acc>(1.0, A, B, /*upper=*/true, /*unit_diag=*/false);

    const int m = 2, n = 1;
    REQUIRE(counting_acc::clears   == m * n);  // one clear per (row, col)
    // row0: diagonal + 1 off-diag = 2 add_products; row1: diagonal only = 1
    REQUIRE(counting_acc::products == (m + (m - 1)) * n);
    REQUIRE(counting_acc::values   == m * n);
    REQUIRE(counting_acc::assigns  == 0);  // diagonal is NOT an assign-seed
    REQUIRE(B(0,0) == 5.0);  // 1*2 + 1*3
    REQUIRE(B(1,0) == 3.0);  // 1*3
}

TEST_CASE("trmm fp64 accumulator beats fp32 on a near-cancelling row",
          "[operation][trmm][accumulator]") {
    const std::size_t n = 2000;
    mat::dense2D<float> A(n, n);
    mat::dense2D<float> B_naive(n, 1), B_wide(n, 1);
    for (std::size_t j = 0; j < n; ++j) { B_naive(j,0) = 1.0f; B_wide(j,0) = 1.0f; }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j)
            A(i, j) = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;

    double ref = 0.0;
    for (std::size_t j = 0; j < n; ++j) ref += static_cast<double>(A(0, j));

    trmm(1.0f, A, B_naive, /*upper=*/true, /*unit_diag=*/false);          // fp32 accumulate
    trmm<double>(1.0, A, B_wide, /*upper=*/true, /*unit_diag=*/false);    // fp64 accumulate, fp32 result

    double e_naive = std::abs(static_cast<double>(B_naive(0,0)) - ref);
    double e_wide  = std::abs(static_cast<double>(B_wide(0,0))  - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}
