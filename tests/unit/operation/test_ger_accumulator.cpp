// MTL5 -- accumulator policy for ger (#261, Part C, BLAS L2).
// ger is a single-term update per entry: A(i,j) += alpha*x(i)*y(j), seeded with
// the existing A(i,j) via `assign` -- not `clear` -- matching axpy's one-term
// seeded pattern (#511), not symv/gemv's zero-seeded row reduction.
//
// ger's product has THREE operands (alpha, x(i), y(j)); the accumulator contract
// fuses only two. alpha*x(i) is precomputed once per row, outside the
// accumulator, and paired with y(j) inside add_product -- so config 3 (quire)
// removes rounding on the second multiply and the final add, not on the first.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/ger.hpp>
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

TEST_CASE("ger default behavior is unchanged", "[operation][ger][accumulator]") {
    mat::dense2D<double> A(2, 2);
    A(0,0)=1; A(0,1)=1; A(1,0)=1; A(1,1)=1;
    vec::dense_vector<double> x = {1.0, 2.0}, y = {3.0, 4.0};
    ger(2.0, x, y, A);
    // A(i,j) += 2*x(i)*y(j)
    REQUIRE_THAT(A(0,0), WithinRel(7.0, 1e-12));    // 1 + 2*1*3
    REQUIRE_THAT(A(0,1), WithinRel(9.0, 1e-12));    // 1 + 2*1*4
    REQUIRE_THAT(A(1,0), WithinRel(13.0, 1e-12));   // 1 + 2*2*3
    REQUIRE_THAT(A(1,1), WithinRel(17.0, 1e-12));   // 1 + 2*2*4
}

TEST_CASE("ger drives assign, not clear -- it is seeded with A(i,j)",
          "[operation][ger][accumulator]") {
    mat::dense2D<double> A(2, 2);
    A(0,0)=1; A(0,1)=1; A(1,0)=1; A(1,1)=1;
    vec::dense_vector<double> x = {1.0, 1.0}, y = {1.0, 1.0};

    counting_acc::reset();
    ger<counting_acc>(3.0, x, y, A);

    const int entries = 4;   // 2x2
    REQUIRE(counting_acc::assigns  == entries);   // seeded per entry with A(i,j)
    REQUIRE(counting_acc::products == entries);   // one fused term per entry
    REQUIRE(counting_acc::values   == entries);
    REQUIRE(counting_acc::clears   == 0);         // NOT cleared
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            REQUIRE(A(i,j) == 4.0);               // 1 + 3*1*1
}

TEST_CASE("the accumulator sees alpha at its own precision, not the element's",
          "[operation][ger][accumulator]") {
    // Same bug class caught in review of #511 for axpy/scale: alpha must not be
    // rounded to the element type before add_product sees it.
    const double alpha = 1.0 / 3.0;
    REQUIRE(static_cast<double>(static_cast<float>(alpha)) != alpha);   // premise

    mat::dense2D<float> A(1, 1);
    A(0,0) = 0.0f;
    vec::dense_vector<float> x = {1.0f}, y = {1.0f};
    ger<double>(alpha, x, y, A);

    const double want = 0.0 + alpha * 1.0 * 1.0;   // unrounded alpha
    REQUIRE(A(0,0) == static_cast<float>(want));
}

TEST_CASE("ger: a narrower accumulator is observably different from the default",
          "[operation][ger][accumulator]") {
    // Mirrors axpy's proven separating case (#511), not the widening case its
    // own file tried and documented as non-separating for a single fused term:
    // double operands, narrowed through a float accumulator. The narrowing
    // happens inside accumulator_traits' own storage (Acc = float), so it is
    // visible on every dispatch, unlike a claim about the default's rounding.
    mat::dense2D<double> A(1, 1), A_narrow(1, 1);
    A(0,0) = 1.0 / 3.0;
    A_narrow(0,0) = A(0,0);
    const double alpha = 1.0 / 7.0;
    vec::dense_vector<double> x = {1.0}, y = {1.0 / 11.0};

    ger(alpha, x, y, A);                // default: double throughout
    ger<float>(alpha, x, y, A_narrow);  // config 1, accumulate in float

    // NO ORACLE for the default's own rounding -- it may be BLAS, SIMD, or the
    // generic loop, so any exact expectation for it is a claim about the build.
    // What is asserted is that the float accumulator is observably different,
    // which holds under every dispatch because the storage precision differs.
    REQUIRE(A(0,0) != A_narrow(0,0));
}
