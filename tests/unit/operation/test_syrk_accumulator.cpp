// MTL5 -- accumulator policy for syrk (#261, Part C, BLAS L3).
// syrk sums each lower-triangle reduction A(i,l)*A(j,l) over l, seeded with
// clear -- a zero-seeded reduction -- then combines with alpha/beta once,
// outside the reduction. The mirror to the upper triangle afterward is a
// plain unrounded copy, not a re-sum: new shape for L3, no L2 analogue.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/syrk.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

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

TEST_CASE("syrk default behavior is unchanged -- produces a full symmetric result",
          "[operation][syrk][accumulator]") {
    mat::dense2D<double> A(2, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    mat::dense2D<double> C(2, 2);
    C(0,0)=0; C(0,1)=0;
    C(1,0)=0; C(1,1)=0;
    syrk(1.0, A, 0.0, C);
    REQUIRE_THAT(C(0,0), WithinRel(14.0, 1e-12));
    REQUIRE_THAT(C(1,1), WithinRel(77.0, 1e-12));
    REQUIRE_THAT(C(0,1), WithinRel(32.0, 1e-12));
    REQUIRE_THAT(C(1,0), WithinRel(32.0, 1e-12));
}

TEST_CASE("syrk mirrors the lower triangle exactly, not by re-summing",
          "[operation][syrk][accumulator]") {
    mat::dense2D<double> A(3, 2);
    A(0,0)=1; A(0,1)=1;
    A(1,0)=2; A(1,1)=1;
    A(2,0)=3; A(2,1)=1;
    mat::dense2D<double> C(3, 3);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) C(i,j) = 0;

    counting_acc::reset();
    syrk<counting_acc>(1.0, A, 0.0, C);

    const int m = 3;
    const int lower_count = m * (m + 1) / 2;
    REQUIRE(counting_acc::clears   == lower_count);
    REQUIRE(counting_acc::values   == lower_count);
    REQUIRE(counting_acc::assigns  == 0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(C(i,j) == C(j,i));
}

TEST_CASE("syrk fp64 accumulator beats fp32 on a near-cancelling row",
          "[operation][syrk][accumulator]") {
    const std::size_t n = 2000;
    mat::dense2D<float> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;

    double ref = 0.0;
    for (std::size_t l = 0; l < n; ++l)
        ref += static_cast<double>(A(0, l)) * static_cast<double>(A(0, l));

    mat::dense2D<float> C_naive(n, n), C_wide(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) { C_naive(i,j) = 0.0f; C_wide(i,j) = 0.0f; }

    syrk(1.0f, A, 0.0f, C_naive);
    syrk<double>(1.0, A, 0.0, C_wide);

    double e_naive = std::abs(static_cast<double>(C_naive(0,0)) - ref);
    double e_wide  = std::abs(static_cast<double>(C_wide(0,0))  - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("syrk accumulator/result types are honored", "[operation][syrk][accumulator]") {
    mat::dense2D<float> A(2, 2);
    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;
    mat::dense2D<float> C_wide(2, 2);
    C_wide(0,0)=0.0f; C_wide(0,1)=0.0f; C_wide(1,0)=0.0f; C_wide(1,1)=0.0f;
    syrk<double>(1.0f, A, 0.0f, C_wide);
    REQUIRE_THAT(C_wide(0,0), WithinRel(5.0f, 1e-6f));
    REQUIRE_THAT(C_wide(1,1), WithinRel(25.0f, 1e-6f));
    REQUIRE_THAT(C_wide(0,1), WithinRel(11.0f, 1e-6f));
}
