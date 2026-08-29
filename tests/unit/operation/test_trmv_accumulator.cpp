// MTL5 -- accumulator policy for trmv (#261, Part C, BLAS L2).
// trmv computes each row's reduction over the triangular part of A, with the
// diagonal term (1*x(i) or A(i,i)*x(i)) fed through add_product as the first
// term of a clear-seeded reduction -- same shape as symv's row loop, not
// special-cased as a seed. Traversal order (forward for upper, reverse for
// lower) is an in-place-overwrite hazard, unrelated to accumulator choice.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

namespace {

/// Counts contract operations, mirrors symv's counting_acc (#511 pattern).
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

TEST_CASE("trmv default behavior is unchanged -- upper, explicit diagonal",
          "[operation][trmv][accumulator]") {
    mat::dense2D<double> A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=1;
    A(1,0)=9; A(1,1)=2; A(1,2)=1;  // lower entries must be ignored
    A(2,0)=9; A(2,1)=9; A(2,2)=2;
    vec::dense_vector<double> x = {1.0, 1.0, 1.0};
    trmv(A, x, /*upper=*/true, /*unit_diag=*/false);
    // row0: 2*1 + 1*1 + 1*1 = 4
    // row1: 2*1 + 1*1       = 3
    // row2: 2*1             = 2
    REQUIRE_THAT(x(0), WithinRel(4.0, 1e-12));
    REQUIRE_THAT(x(1), WithinRel(3.0, 1e-12));
    REQUIRE_THAT(x(2), WithinRel(2.0, 1e-12));
}

TEST_CASE("trmv default behavior is unchanged -- lower, unit diagonal",
          "[operation][trmv][accumulator]") {
    mat::dense2D<double> A(3, 3);
    A(0,0)=9; A(0,1)=9; A(0,2)=9;  // upper entries must be ignored
    A(1,0)=1; A(1,1)=9; A(1,2)=9;
    A(2,0)=1; A(2,1)=1; A(2,2)=9;
    vec::dense_vector<double> x = {1.0, 1.0, 1.0};
    trmv(A, x, /*upper=*/false, /*unit_diag=*/true);
    // row0: 1*x0                = 1
    // row1: 1*x0 + 1*x1         = 1 + 1 = 2
    // row2: 1*x0 + 1*x1 + 1*x2  = 1 + 1 + 1 = 3
    REQUIRE_THAT(x(0), WithinRel(1.0, 1e-12));
    REQUIRE_THAT(x(1), WithinRel(2.0, 1e-12));
    REQUIRE_THAT(x(2), WithinRel(3.0, 1e-12));
}

TEST_CASE("trmv drives clear, not assign -- diagonal is fed via add_product, not seeded",
          "[operation][trmv][accumulator]") {
    // Confirms the diagonal term is the first add_product call, not an assign
    // -- matching the header comment's stated design (no seed special-case).
    mat::dense2D<double> A(2, 2);
    A(0,0)=1; A(0,1)=1; A(1,0)=1; A(1,1)=1;
    vec::dense_vector<double> x = {2.0, 3.0};

    counting_acc::reset();
    trmv<counting_acc>(A, x, /*upper=*/true, /*unit_diag=*/false);

    const int n = 2;
    REQUIRE(counting_acc::clears   == n);  // one clear per row
    // row0: diagonal + 1 off-diag = 2 add_products; row1: diagonal only = 1
    REQUIRE(counting_acc::products == n + (n - 1));
    REQUIRE(counting_acc::values   == n);  // one round-out per row
    REQUIRE(counting_acc::assigns  == 0);  // diagonal is NOT an assign-seed
    REQUIRE(x(0) == 5.0);  // 1*2 + 1*3
    REQUIRE(x(1) == 3.0);  // 1*3
}

TEST_CASE("trmv fp64 accumulator beats fp32 on a near-cancelling row",
          "[operation][trmv][accumulator]") {
    const std::size_t n = 2000;
    mat::dense2D<float> A(n, n);
    vec::dense_vector<float> x_naive(n), x_wide(n);
    for (std::size_t j = 0; j < n; ++j) { x_naive(static_cast<int>(j)) = 1.0f; x_wide(static_cast<int>(j)) = 1.0f; }
    // Upper triangle only (row i uses columns i..n-1), values chosen so
    // row 0's sum nearly cancels in fp32.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j)
            A(i, j) = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;

    double ref = 0.0;
    for (std::size_t j = 0; j < n; ++j) ref += static_cast<double>(A(0, j));

    trmv(A, x_naive, /*upper=*/true, /*unit_diag=*/false);          // fp32 accumulate
    trmv<double>(A, x_wide, /*upper=*/true, /*unit_diag=*/false);   // fp64 accumulate, fp32 result

    double e_naive = std::abs(static_cast<double>(x_naive(0)) - ref);
    double e_wide  = std::abs(static_cast<double>(x_wide(0))  - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("trmv accumulator/result types are honored", "[operation][trmv][accumulator]") {
    mat::dense2D<float> A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=2; A(1,1)=1;
    vec::dense_vector<float> x_wide = {1.0f, 1.0f};
    trmv<double>(A, x_wide, /*upper=*/true, /*unit_diag=*/false);
    // row0: 1*1 + 2*1 = 3 ; row1: 1*1 = 1
    REQUIRE_THAT(x_wide(0), WithinRel(3.0f, 1e-6f));
    REQUIRE_THAT(x_wide(1), WithinRel(1.0f, 1e-6f));
}
