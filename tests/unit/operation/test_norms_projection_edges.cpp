// Norms and projection follow-ups of issue 865 (QA sweep #704, from the
// review of e9dc960 and 8a1a1dc):
//
//   - two_norm / frobenius_norm with an explicit Accumulator at extreme values
//     and with NaN; NaN with Inf; a complex matrix; a float matrix with NaN;
//     the long double rescan;
//   - saturating_cast for bool, wchar_t and, under GNU, __int128 (which a
//     widening to long long truncated).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/projection.hpp>

using namespace mtl;

namespace {
constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf  = std::numeric_limits<double>::infinity();
}

TEST_CASE("two_norm and frobenius_norm with an explicit Accumulator at the extremes", "[operation][norms][865]") {
    // float elements whose squares overflow float: summed in double.
    vec::dense_vector<float> v = {3e30f, 4e30f};
    CHECK(two_norm<double>(v) == 5e30f);
    mat::dense2D<float> m = {{3e30f, 0.0f}, {0.0f, 4e30f}};
    CHECK(frobenius_norm<double>(m) == 5e30f);
    // ... and squares that underflow float.
    vec::dense_vector<float> t = {3e-30f, 4e-30f};
    CHECK(std::abs(two_norm<double>(t) / 5e-30f - 1.0f) < 1e-6f);
    // NaN propagates through the accumulator path.
    vec::dense_vector<float> n = {1.0f, std::numeric_limits<float>::quiet_NaN()};
    CHECK(std::isnan(two_norm<double>(n)));
    mat::dense2D<float> mn = {{1.0f, std::numeric_limits<float>::quiet_NaN()}};
    CHECK(std::isnan(frobenius_norm<double>(mn)));
}

TEST_CASE("norms: NaN with Inf is NaN", "[operation][norms][865]") {
    // inf^2 + NaN is NaN; the rescan must not turn it into inf.
    vec::dense_vector<double> v = {inf, qnan, 1.0};
    CHECK(std::isnan(two_norm(v)));
    vec::dense_vector<double> w = {qnan, inf};
    CHECK(std::isnan(two_norm(w)));
    mat::dense2D<double> m = {{inf, 1.0}, {qnan, 2.0}};
    CHECK(std::isnan(frobenius_norm(m)));
    // Inf alone is Inf.
    vec::dense_vector<double> u = {inf, 1.0};
    CHECK(two_norm(u) == inf);
}

TEST_CASE("frobenius_norm of a complex matrix", "[operation][norms][865]") {
    using C = std::complex<double>;
    mat::dense2D<C> m = {{C(3, 4), C(0, 0)}, {C(0, 0), C(0, 0)}};
    CHECK(frobenius_norm(m) == 5.0);
    // |1e200 + 1e200 i| squared overflows: the rescan keeps it finite.
    mat::dense2D<C> big = {{C(1e200, 1e200), C(1e200, -1e200)}};
    CHECK(std::abs(frobenius_norm(big) / 2e200 - 1.0) < 1e-14);
    mat::dense2D<C> withnan = {{C(1, qnan)}};
    CHECK(std::isnan(frobenius_norm(withnan)));
}

TEST_CASE("frobenius_norm of a float matrix with NaN", "[operation][norms][865]") {
    mat::dense2D<float> m = {{1.0f, 2.0f}, {std::numeric_limits<float>::quiet_NaN(), 3.0f}};
    CHECK(std::isnan(frobenius_norm(m)));
}

TEST_CASE("two_norm and frobenius_norm rescan in long double", "[operation][norms][865]") {
    using L = long double;
    if constexpr (std::numeric_limits<L>::max_exponent > std::numeric_limits<double>::max_exponent) {
        // Beyond double's range: only long double holds these.
        const L big = std::ldexp(L(1), 12000), tiny = std::ldexp(L(1), -12000);
        vec::dense_vector<L> v = {big, big};
        CHECK(std::abs(two_norm(v) / (big * std::sqrt(L(2))) - 1) < 1e-15L);
        vec::dense_vector<L> t = {tiny, tiny};
        CHECK(std::abs(two_norm(t) / (tiny * std::sqrt(L(2))) - 1) < 1e-15L);
        mat::dense2D<L> m = {{big, big}, {big, big}};
        CHECK(std::abs(frobenius_norm(m) / (2 * big) - 1) < 1e-15L);
    } else {
        // long double == double (MSVC, Apple ARM64): the double rescan.
        vec::dense_vector<L> v = {L(1e200), L(1e200)};
        CHECK(std::abs(two_norm(v) / (L(1e200) * std::sqrt(L(2))) - 1) < 1e-15L);
    }
}

TEST_CASE("saturating_cast for bool and wchar_t", "[operation][projection][865]") {
    using detail::saturating_cast;
    CHECK(saturating_cast<bool>(5) == true);
    CHECK(saturating_cast<bool>(-3) == false);
    CHECK(saturating_cast<bool>(0) == false);
    CHECK(saturating_cast<int>(true) == 1);
    constexpr auto wmax = std::numeric_limits<wchar_t>::max();
    constexpr auto wmin = std::numeric_limits<wchar_t>::min();
    CHECK(saturating_cast<wchar_t>(std::numeric_limits<long long>::max()) == wmax);
    CHECK(saturating_cast<wchar_t>(std::numeric_limits<long long>::min()) == wmin);
    CHECK(saturating_cast<wchar_t>(65) == wchar_t(65));
    CHECK(saturating_cast<unsigned char>(wchar_t(300)) == 255);
}

#if defined(__SIZEOF_INT128__)
TEST_CASE("saturating_cast with __int128", "[operation][projection][865]") {
    using detail::saturating_cast;
    __extension__ typedef __int128 i128;
    __extension__ typedef unsigned __int128 u128;
    // 2^100 was truncated to long long (0) before the comparison, and the
    // cast returned 0 for INT64_MAX.
    const i128 big = i128(1) << 100;
    CHECK(saturating_cast<std::int64_t>(big) == std::numeric_limits<std::int64_t>::max());
    CHECK(saturating_cast<std::int64_t>(-big) == std::numeric_limits<std::int64_t>::min());
    CHECK(saturating_cast<std::uint64_t>(-big) == 0u);
    CHECK(saturating_cast<std::int32_t>(i128(-5)) == -5);
    CHECK(saturating_cast<std::uint64_t>(u128(1) << 70) == std::numeric_limits<std::uint64_t>::max());
    CHECK(saturating_cast<i128>(std::numeric_limits<std::uint64_t>::max())
          == i128(std::numeric_limits<std::uint64_t>::max()));
    CHECK(saturating_cast<u128>(-1) == 0u);
}
#endif
