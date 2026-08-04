// MTL5 -- accumulator policy for gemv and the sum-of-squares norms (#160, #162).
// Mirrors dot/gemm: an explicit Accumulator sums in a precision distinct from the
// element type, with the result delivered in the natural output/magnitude type.
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("gemv default behavior is unchanged", "[operation][gemv][accumulator]") {
    mat::dense2D<double> A(2, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3; A(1,0)=4; A(1,1)=5; A(1,2)=6;
    vec::dense_vector<double> x = {1.0, 1.0, 1.0}, y(2, 0.0);
    mult(A, x, y);
    REQUIRE_THAT(y(0), WithinRel(6.0, 1e-12));    // 1+2+3
    REQUIRE_THAT(y(1), WithinRel(15.0, 1e-12));   // 4+5+6
}

TEST_CASE("gemv fp64 accumulator beats fp32 on a long contraction",
          "[operation][gemv][accumulator]") {
    const std::size_t m = 4, n = 100000;
    mat::dense2D<float> A(m, n);
    vec::dense_vector<float> x(n);
    for (std::size_t j = 0; j < n; ++j) {
        x(static_cast<int>(j)) = 1.0f;
        for (std::size_t i = 0; i < m; ++i)
            A(i, j) = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;   // near-cancelling
    }
    double ref = 0.0;
    for (std::size_t j = 0; j < n; ++j) ref += static_cast<double>(A(0, j));

    vec::dense_vector<float> y_naive(m, 0.0f);
    mult(A, x, y_naive);                 // fp32 accumulate (default generic for non-contiguous? still float)

    vec::dense_vector<float> y_wide(m, 0.0f);
    mult<double>(A, x, y_wide);          // fp64 accumulate, fp32 result

    double e_naive = std::abs(static_cast<double>(y_naive(0)) - ref);
    double e_wide  = std::abs(static_cast<double>(y_wide(0))  - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("two_norm accumulator policy: default unchanged, wide is accurate",
          "[operation][norms][accumulator]") {
    vec::dense_vector<double> v = {3.0, 4.0};
    REQUIRE_THAT(two_norm(v), WithinRel(5.0, 1e-12));
    REQUIRE_THAT(two_norm<double>(v), WithinRel(5.0, 1e-12));

    // fp64 accumulate over many float squares beats fp32 accumulation.
    const std::size_t n = 200000;
    vec::dense_vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) x(static_cast<int>(i)) = 1.0f;   // exact answer sqrt(n)
    double ref = std::sqrt(static_cast<double>(n));

    float  nf = two_norm(x);             // fp32 accumulate
    auto   nw = two_norm<double>(x);     // fp64 accumulate
    static_assert(std::is_same_v<decltype(nw), float>);   // returns element magnitude type
    double e_naive = std::abs(static_cast<double>(nf) - ref);
    double e_wide  = std::abs(static_cast<double>(nw) - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("frobenius_norm accumulator policy", "[operation][norms][accumulator]") {
    mat::dense2D<double> m(2, 2);
    m(0,0)=1; m(0,1)=2; m(1,0)=2; m(1,1)=4;   // sum of squares = 25
    REQUIRE_THAT(frobenius_norm(m), WithinRel(5.0, 1e-12));
    REQUIRE_THAT(frobenius_norm<double>(m), WithinRel(5.0, 1e-12));
}

// ---------------------------------------------------------------------------
// Regression: #324 -- two_norm<Acc>/frobenius_norm<Acc> rounded the accumulator
// out to the ACCUMULATOR type and then took its square root. That is a no-op
// for a plain arithmetic accumulator, which is why the cases above passed and
// hid this, but it yields an fma_accumulator<T> for configuration 2 and a
// super-accumulator for configuration 3 -- neither of which has a sqrt, so
// neither non-trivial configuration compiled at all.
//
// These are primarily compile-time pins: the failure was at instantiation.
// ---------------------------------------------------------------------------

namespace {
// Exact sum of squares in double-double, via FMA. Portable: needs only IEEE
// double, unlike a `long double` reference, which is 80-bit on x86-64 and only
// 64-bit on Apple ARM64 -- where it is no better than the values under test and
// WORSE than a compensated accumulator, so the more accurate answer scores as
// the less accurate one. That is precisely how the first version of this test
// failed on macOS ARM64 while passing on x86-64.
struct dd { double hi{}, lo{}; };
inline void dd_add(dd& a, double x) {          // Knuth two-sum
    double s = a.hi + x;
    double bb = s - a.hi;
    a.lo += (a.hi - (s - bb)) + (x - bb);
    a.hi = s;
}
inline double exact_sum_of_squares(const float* p, std::size_t n) {
    dd acc;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(p[i]);
        const double prod = x * x;                       // exact: 24-bit input
        const double err  = std::fma(x, x, -prod);       // the rounded-off part
        dd_add(acc, prod);
        dd_add(acc, err);
    }
    return acc.hi + acc.lo;
}
}  // namespace

// Configuration 3 stand-in: an exact compensated super-accumulator, playing the
// role of a Universal quire. MTL5 stays free of any external number library, so
// the real quire specialization lives in the peer repo; this exercises the same
// contract -- a custom Acc with no sqrt of its own.
//
// Accumulates in `double`, deliberately NOT `long double`. `long double` is
// 80-bit on x86-64, 64-bit (i.e. plain double) on Apple ARM64 and MSVC, and
// 128-bit on ARM64 Linux -- three different precisions across lanes this repo
// already builds on. Compensated summation recovers roughly twice the base
// type's precision on its own, so this is bit-identically accurate here without
// depending on a type whose width varies by ~60 bits between targets.
namespace {
struct kahan_acc { double sum{}, c{}; };
}
namespace mtl::math {
template <typename Value>
struct accumulator_traits<kahan_acc, Value> {
    using Acc = kahan_acc;
    static void clear(Acc& a) { a.sum = 0; a.c = 0; }
    static void assign(Acc& a, const Value& v) { a.sum = static_cast<double>(v); a.c = 0; }
    template <typename Result = Value>
    static Result value(const Acc& a) { return static_cast<Result>(a.sum); }
    static void add_product(Acc& a, const Value& m, const Value& v) {
        double y = static_cast<double>(m) * static_cast<double>(v) - a.c;
        double t = a.sum + y;
        a.c = (t - a.sum) - y;
        a.sum = t;
    }
};
}

TEST_CASE("two_norm/frobenius_norm accept every accumulator configuration (#324)",
          "[operation][norms][accumulator][regression]") {
    const std::size_t n = 20000;
    vec::dense_vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = 1.0f + static_cast<float>(i % 7) * 1e-6f;

    const double exact = std::sqrt(exact_sum_of_squares(v.data(), n));

    // Configuration 1: plain arithmetic accumulator (already worked).
    const auto c1 = two_norm<double>(v);
    // Configuration 2: fused multiply-add accumulator -- did not compile.
    const auto c2 = two_norm<math::fma_accumulator<double>>(v);
    // Configuration 3: custom super-accumulator -- did not compile.
    const auto c3 = two_norm<kahan_acc>(v);

    // All deliver the element magnitude type, as the docstring promises.
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(c1)>, float>);
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(c2)>, float>);
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(c3)>, float>);

    // Each is far more accurate than a bare float reduction over this vector.
    const auto plain = two_norm(v);
    REQUIRE(std::abs(c1 - exact) <= std::abs(static_cast<double>(plain) - exact));
    REQUIRE_THAT(static_cast<double>(c1), WithinRel(exact, 1e-6));
    REQUIRE_THAT(static_cast<double>(c2), WithinRel(exact, 1e-6));
    REQUIRE_THAT(static_cast<double>(c3), WithinRel(exact, 1e-6));

    mat::dense2D<float> M(120, 120);
    for (std::size_t r = 0; r < 120; ++r)
        for (std::size_t c = 0; c < 120; ++c)
            M(r, c) = 1.0f + static_cast<float>((r + c) % 5) * 1e-6f;

    std::vector<float> Mflat;
    Mflat.reserve(120 * 120);
    for (std::size_t r = 0; r < 120; ++r)
        for (std::size_t c = 0; c < 120; ++c) Mflat.push_back(M(r, c));
    const double fexact = std::sqrt(exact_sum_of_squares(Mflat.data(), Mflat.size()));

    const auto f1 = frobenius_norm<double>(M);
    const auto f2 = frobenius_norm<math::fma_accumulator<double>>(M);
    const auto f3 = frobenius_norm<kahan_acc>(M);
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(f2)>, float>);
    REQUIRE_THAT(static_cast<double>(f1), WithinRel(fexact, 1e-6));
    REQUIRE_THAT(static_cast<double>(f2), WithinRel(fexact, 1e-6));
    REQUIRE_THAT(static_cast<double>(f3), WithinRel(fexact, 1e-6));
}

TEST_CASE("accumulator_round_type names the accumulator's own precision (#324)",
          "[operation][norms][accumulator][regression]") {
    // Rounding out to the magnitude type would compile, but would narrow the
    // sum BEFORE the square root and discard what the accumulator bought.
    STATIC_REQUIRE(std::is_same_v<math::accumulator_round_type_t<double, float>, double>);
    STATIC_REQUIRE(std::is_same_v<
        math::accumulator_round_type_t<math::fma_accumulator<double>, float>, double>);
    // A custom accumulator has no nameable arithmetic type, so it delivers in
    // the magnitude type -- external specializations need supply nothing new.
    STATIC_REQUIRE(std::is_same_v<math::accumulator_round_type_t<kahan_acc, float>, float>);
}

// ---------------------------------------------------------------------------
// #379: two_norm/frobenius_norm gain a Result parameter, as dot already has.
//
// The subtle part is what Result governs. It is NOT only the final cast.
// accumulator_round_type_t<Acc, Mag> maps a non-arithmetic accumulator to Mag,
// so if the round-out stayed at the element magnitude type, an exact
// accumulation would be flattened to element precision BEFORE the sqrt and no
// return type could recover it. Result therefore feeds BOTH the round-out and
// the cast -- otherwise `two_norm<quire, double>` is not merely unhelpful, it is
// strictly WORSE than `two_norm<double, double>`, which inverts the ordering a
// caller is choosing between.
// ---------------------------------------------------------------------------


TEST_CASE("two_norm/frobenius_norm Result parameter (#379)",
          "[operation][norms][accumulator][regression]") {
    const std::size_t n = 20000;
    vec::dense_vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = 1.0f + static_cast<float>(i % 7) * 1e-6f;

    const double exact = std::sqrt(exact_sum_of_squares(v.data(), n));
    const auto rel = [&](double x) { return std::abs(x - exact) / exact; };

    SECTION("Result = void is byte-for-byte today's behaviour") {
        // The whole change must be additive: no existing call may move.
        const auto a = two_norm<double>(v);
        const auto b = two_norm<kahan_acc>(v);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(a)>, float>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(b)>, float>);
        // Both bottleneck on the float delivery type -- which is exactly the
        // observation that motivated #379.
        REQUIRE(a == b);
    }

    SECTION("Result widens the delivery type") {
        const auto c = two_norm<double, double>(v);
        const auto d = two_norm<kahan_acc, double>(v);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(c)>, double>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(d)>, double>);

        // Both are far better than the float-delivered versions.
        REQUIRE(rel(c) < 1e-12);
        REQUIRE(rel(d) < 1e-12);
    }

    SECTION("the accumulator becomes observable, and in the right direction") {
        const auto c = two_norm<double, double>(v);     // fp64 accumulation
        const auto d = two_norm<kahan_acc, double>(v);  // exact accumulation

        // Observable at all: this is what #379 asked for.
        REQUIRE(d != c);
        // ...and better, not worse. Implementing Result as only the final cast
        // would have made the exact accumulator LESS accurate than the fp64 one
        // (2.4e-08 against 1.4e-14), inverting the choice.
        //
        // Ordering, not an exact value. `REQUIRE(rel(d) == 0.0)` held on x86-64
        // and failed on macOS ARM64: it asserted a rounding coincidence of the
        // reference rather than anything about the feature.
        REQUIRE(rel(d) <= rel(c));
    }

    SECTION("frobenius_norm behaves the same way") {
        mat::dense2D<float> m(100, 200);
        std::vector<float> flat;
        flat.reserve(100 * 200);
        for (std::size_t i = 0; i < 100; ++i)
            for (std::size_t j = 0; j < 200; ++j) {
                m(i, j) = 1.0f + static_cast<float>((i + j) % 7) * 1e-6f;
                flat.push_back(m(i, j));
            }
        const double fexact = std::sqrt(exact_sum_of_squares(flat.data(), flat.size()));

        const auto f0 = frobenius_norm<double>(m);
        const auto f1 = frobenius_norm<double, double>(m);
        const auto f2 = frobenius_norm<kahan_acc, double>(m);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(f0)>, float>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(f1)>, double>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(f2)>, double>);

        REQUIRE(std::abs(f1 - fexact) / fexact < 1e-12);
        REQUIRE(std::abs(f2 - fexact) / fexact <= std::abs(f1 - fexact) / fexact);
    }

    SECTION("Result also works with the fma_accumulator configuration") {
        // accumulator_round_type_t<fma_accumulator<T>, Mag> is T regardless of
        // Mag, which is correct: the accumulator really does hold T precision,
        // so Result widens the delivery without falsely claiming more.
        const auto e = two_norm<math::fma_accumulator<double>, double>(v);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(e)>, double>);
        REQUIRE(rel(e) < 1e-12);
    }
}
