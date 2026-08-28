// Accumulator policy on the BLAS L1 remainder: axpy and scale (#261, Part C).
//
// `dot`, `mult` and `norms` have been accumulator-aware since #157; `axpy` and
// `scale` were the two L1 routines left. They are threaded now, and the point of
// this file is to pin what each one actually gains, because the two answers are
// different and only one of them is interesting.
//
// AXPY IS A ONE-TERM SUM OF PRODUCTS: `y(i) + alpha * x(i)`, seeded with y(i)
// rather than with zero. All three configurations separate, exactly as they do
// in a reduction -- config 2 fuses away the product rounding, which is the whole
// difference between `y + a*x` and `fma(a, x, y)`.
//
// SCALE IS A BARE PRODUCT and the configurations do NOT separate: `fma(m, v, 0)`
// is `m * v` rounded once, and so is a plain multiply, so config 2 and config 3
// buy nothing a quire could improve on. What a non-default accumulator buys is
// WIDENING -- the product formed in Acc rather than in the element type. That is
// asserted here as a real difference rather than implied to be more than it is.
//
// Every expected value is computed from the IEEE operations directly, in the
// precision the policy selects, never by running a second mtl kernel.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/axpy.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;

namespace {

/// Counts the contract operations, so a test can assert WHICH primitives a
/// routine drives -- not just that the answer came out right.
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

TEST_CASE("axpy: Accumulator = void is unchanged", "[operation][axpy][accumulator]") {
    // The default must stay byte-identical -- it is the BLAS/SIMD dispatch.
    const std::size_t n = 64;
    vec::dense_vector<double> x(n), y(n), yref(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i) = 0.5 + 0.25 * static_cast<double>(i);
        y(i) = -1.0 + 0.125 * static_cast<double>(i);
        yref(i) = y(i);
    }
    axpy(2.5, x, y);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(y(i) == yref(i) + 2.5 * x(i));
}

TEST_CASE("axpy: the accumulator precision is what the policy selects",
          "[operation][axpy][accumulator]") {
    // A NARROWER accumulator than the element type, because that is the case
    // that separates unambiguously on every machine. `axpy<float>` over double
    // vectors forms the product and the sum in float and widens back on store,
    // so the result is visibly not the double answer.
    //
    // The obvious test -- a WIDER accumulator over float vectors -- was tried
    // first and does not separate: for float operands, double holds the product
    // exactly and the sum almost always, so config 1 in double and a fused
    // multiply-add agree. A search over two families of values found no
    // disagreement at all. Asserting a difference there would have been
    // asserting a coincidence.
    const std::size_t n = 6;
    vec::dense_vector<double> x(n), y(n), y_narrow(n);
    const double alpha = 1.0 / 3.0;
    for (std::size_t i = 0; i < n; ++i) {
        x(i) = 1.0 / static_cast<double>(i + 7);
        y(i) = 1.0 / static_cast<double>(i + 3);
        y_narrow(i) = y(i);
    }
    std::vector<double> y0(n);
    for (std::size_t i = 0; i < n; ++i) y0[i] = y(i);

    axpy(alpha, x, y);                 // default: double throughout
    axpy<float>(alpha, x, y_narrow);   // config 1, accumulate in float

    for (std::size_t i = 0; i < n; ++i) {
        // The float accumulator: seed rounds to float, product in float, sum in
        // float, widened back to double on store.
        float acc = static_cast<float>(y0[i]);
        acc += static_cast<float>(alpha) * static_cast<float>(x(i));
        REQUIRE(y_narrow(i) == static_cast<double>(acc));
        // NO ORACLE FOR THE DEFAULT. Its rounding is dispatch-dependent -- BLAS,
        // fused SIMD body, unfused SIMD tail, or the generic loop -- so any exact
        // expectation for it is a claim about the build. This test asserted
        // std::fma here and the Highway and LAPACK CI lanes disproved it. What is
        // asserted instead is that the float accumulator is observably different,
        // which holds under every dispatch because the precision differs.
        REQUIRE(y(i)        != y_narrow(i));
    }
}

TEST_CASE("axpy: config 2 reproduces std::fma exactly",
          "[operation][axpy][accumulator]") {
    // fma_accumulator<float> must give the fused result on EVERY dispatch: that
    // is what asking for the policy buys. Asserted against std::fma directly.
    //
    // Deliberately NOT compared against the default. The default may be BLAS, a
    // fused SIMD body, an unfused SIMD tail or the generic loop, and those do not
    // agree with one another -- so "config 2 matches the default" is a statement
    // about the build, not about the policy. This file made that claim and CI
    // falsified it on the Highway and LAPACK lanes.
    const std::size_t n = 8;
    vec::dense_vector<float> x(n), y(n), y_fma(n);
    const float alpha = 1.0f / 3.0f;
    for (std::size_t i = 0; i < n; ++i) {
        x(i) = 1.0f / static_cast<float>(i + 7);
        y(i) = static_cast<float>(i + 1) / 9.0f;
        y_fma(i) = y(i);
    }
    std::vector<float> y0(n);
    for (std::size_t i = 0; i < n; ++i) y0[i] = y(i);

    axpy<math::fma_accumulator<float>>(alpha, x, y_fma);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(y_fma(i) == std::fma(alpha, x(i), y0[i]));
}

TEST_CASE("axpy drives assign, not clear -- it is seeded with y(i)",
          "[operation][axpy][accumulator]") {
    // The contract's `assign` had no caller before this: every reduction kernel
    // starts from zero and uses `clear`. A quire specialization that stubbed
    // assign out would pass the whole suite and fail here.
    const std::size_t n = 5;
    vec::dense_vector<double> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) { x(i) = 2.0; y(i) = 1.0; }

    counting_acc::reset();
    axpy<counting_acc>(3.0, x, y);

    REQUIRE(counting_acc::assigns  == static_cast<int>(n));   // seeded per element
    REQUIRE(counting_acc::products == static_cast<int>(n));
    REQUIRE(counting_acc::values   == static_cast<int>(n));
    REQUIRE(counting_acc::clears   == 0);                     // NOT cleared
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(y(i) == 7.0);                                 // 1 + 3*2
}

TEST_CASE("scale: Accumulator = void is unchanged", "[operation][scale][accumulator]") {
    const std::size_t n = 64;
    vec::dense_vector<double> c(n);
    std::vector<double> c0(n);
    for (std::size_t i = 0; i < n; ++i) { c(i) = 0.5 + 0.25 * static_cast<double>(i); c0[i] = c(i); }
    scale(0.5, c);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(c(i) == c0[i] * 0.5);
}

TEST_CASE("scale drives clear -- a bare product from a zero seed",
          "[operation][scale][accumulator]") {
    // The mirror of the axpy case above: a scale sums nothing, so it starts from
    // zero and uses `clear`, never `assign`.
    const std::size_t n = 5;
    vec::dense_vector<double> c(n);
    for (std::size_t i = 0; i < n; ++i) c(i) = 4.0;

    counting_acc::reset();
    scale<counting_acc>(2.5, c);

    REQUIRE(counting_acc::clears   == static_cast<int>(n));
    REQUIRE(counting_acc::products == static_cast<int>(n));
    REQUIRE(counting_acc::values   == static_cast<int>(n));
    REQUIRE(counting_acc::assigns  == 0);                     // NOT seeded
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(c(i) == 10.0);
}

TEST_CASE("scale: what a wider accumulator buys is widening, and nothing more",
          "[operation][scale][accumulator]") {
    // The claim the header makes, asserted rather than asserted-away. A float
    // scale by a factor whose float product is inexact:
    //   default  the product is formed and rounded in float
    //   Acc=double  formed in double, rounded once on store
    // Those agree here -- an IEEE multiply is already correctly rounded, so
    // double-rounding a single product to the same type cannot change it. The
    // difference only appears when the ELEMENT type is narrower than the product
    // the caller wants, which is the mixed-precision case.
    const std::size_t n = 16;
    vec::dense_vector<float> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) { a(i) = 1.0f / static_cast<float>(i + 3); b(i) = a(i); }

    scale(0.1f, a);
    scale<double>(0.1f, b);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(a(i) == b(i));      // same answer: there was no rounding to remove

    // config 2 likewise cannot differ: fma(m, v, 0) is a correctly rounded
    // product, which is what the multiply already was.
    vec::dense_vector<float> d(n);
    for (std::size_t i = 0; i < n; ++i) d(i) = 1.0f / static_cast<float>(i + 3);
    scale<math::fma_accumulator<float>>(0.1f, d);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(d(i) == a(i));
}

TEST_CASE("scaled forwards the accumulator policy", "[operation][scale][accumulator]") {
    const std::size_t n = 6;
    vec::dense_vector<double> c(n);
    for (std::size_t i = 0; i < n; ++i) c(i) = 4.0;

    counting_acc::reset();
    const auto out = scaled<counting_acc>(2.5, c);

    REQUIRE(counting_acc::products == static_cast<int>(n));   // policy reached it
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(out(i) == 10.0);
        REQUIRE(c(i)   == 4.0);                               // input untouched
    }
}

TEST_CASE("the accumulator sees the scalar at its own precision, not the element's",
          "[operation][axpy][scale][accumulator]") {
    // Both routines first cast alpha to the ELEMENT type before add_product,
    // which rounded a wider scalar away before the accumulator could use it --
    // exactly defeating the narrow-elements/wider-scalar case the headers name.
    // Caught in review of #511. The operand type now includes S.
    //
    // alpha = 1/3 as a double is not representable in float, so rounding it to
    // the element type first is visible in the result.
    const std::size_t n = 4;
    const double alpha = 1.0 / 3.0;
    REQUIRE(static_cast<double>(static_cast<float>(alpha)) != alpha);   // premise

    SECTION("axpy") {
        vec::dense_vector<float> x(n), y(n);
        std::vector<float> y0(n);
        for (std::size_t i = 0; i < n; ++i) {
            x(i) = 1.0f / static_cast<float>(i + 3);
            y(i) = 1.0f / static_cast<float>(i + 5);
            y0[i] = y(i);
        }
        axpy<double>(alpha, x, y);
        for (std::size_t i = 0; i < n; ++i) {
            const double want = static_cast<double>(y0[i]) +
                                alpha * static_cast<double>(x(i));   // UNROUNDED alpha
            const double stale = static_cast<double>(y0[i]) +
                                 static_cast<double>(static_cast<float>(alpha)) *
                                 static_cast<double>(x(i));          // the old behaviour
            REQUIRE(y(i) == static_cast<float>(want));
            if (static_cast<float>(want) != static_cast<float>(stale))
                REQUIRE(y(i) != static_cast<float>(stale));
        }
    }

    SECTION("scale") {
        vec::dense_vector<float> c(n);
        std::vector<float> c0(n);
        for (std::size_t i = 0; i < n; ++i) { c(i) = 1.0f / static_cast<float>(i + 3); c0[i] = c(i); }
        scale<double>(alpha, c);
        for (std::size_t i = 0; i < n; ++i) {
            const double want = static_cast<double>(c0[i]) * alpha;
            REQUIRE(c(i) == static_cast<float>(want));
        }
    }
}
