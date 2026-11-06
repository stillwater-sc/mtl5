#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/projection.hpp>

using namespace mtl;

// ============================================================================
// project_onto tests
// ============================================================================

TEST_CASE("project_onto: double vector to float vector", "[operation][projection]") {
    vec::dense_vector<double> v(3);
    v(0) = 1.0; v(1) = 2.5; v(2) = -3.14159265358979;

    auto vf = project_onto<float>(v);

    REQUIRE(vf.size() == 3);
    REQUIRE(vf(0) == 1.0f);
    REQUIRE(vf(1) == 2.5f);
    // float has ~7 decimal digits; check within float epsilon
    REQUIRE_THAT(static_cast<double>(vf(2)),
                 Catch::Matchers::WithinAbs(-3.14159265358979, 1e-6));
}

TEST_CASE("project_onto: double matrix to float matrix", "[operation][projection]") {
    mat::dense2D<double> A(2, 3);
    A(0,0) = 1.0; A(0,1) = 2.0; A(0,2) = 3.0;
    A(1,0) = 4.0; A(1,1) = 5.0; A(1,2) = 6.0;

    auto Af = project_onto<float>(A);

    REQUIRE(Af.num_rows() == 2);
    REQUIRE(Af.num_cols() == 3);
    REQUIRE(Af(0,0) == 1.0f);
    REQUIRE(Af(1,2) == 6.0f);
}

TEST_CASE("project_onto: identity projection (same type)", "[operation][projection]") {
    vec::dense_vector<double> v(2);
    v(0) = 1.5; v(1) = 2.5;

    auto v2 = project_onto<double>(v);

    REQUIRE(v2(0) == 1.5);
    REQUIRE(v2(1) == 2.5);
}

TEST_CASE("project_onto: precision loss is measurable", "[operation][projection]") {
    // A value that is exactly representable in double but not in float
    // 1.0 + 2^-24 is distinguishable in double but rounds in float
    double val = 1.0 + std::ldexp(1.0, -24);
    vec::dense_vector<double> v(1);
    v(0) = val;

    auto vf = project_onto<float>(v);

    // In float, this should round to 1.0 (the 24th bit is lost)
    REQUIRE(vf(0) == 1.0f);
    // Confirm it was different in double
    REQUIRE(val != 1.0);
}

// ============================================================================
// embed_into tests
// ============================================================================

TEST_CASE("embed_into: float vector to double vector", "[operation][projection]") {
    vec::dense_vector<float> v(3);
    v(0) = 1.0f; v(1) = 2.5f; v(2) = -3.14f;

    auto vd = embed_into<double>(v);

    REQUIRE(vd.size() == 3);
    REQUIRE(vd(0) == 1.0);
    REQUIRE(vd(1) == 2.5);
    // Value should be exactly the float value (lossless)
    REQUIRE(vd(2) == static_cast<double>(-3.14f));
}

TEST_CASE("embed_into: float matrix to double matrix", "[operation][projection]") {
    mat::dense2D<float> A(2, 2);
    A(0,0) = 1.0f; A(0,1) = 2.0f;
    A(1,0) = 3.0f; A(1,1) = 4.0f;

    auto Ad = embed_into<double>(A);

    REQUIRE(Ad.num_rows() == 2);
    REQUIRE(Ad.num_cols() == 2);
    REQUIRE(Ad(0,0) == 1.0);
    REQUIRE(Ad(1,1) == 4.0);
}

// ============================================================================
// Roundtrip tests
// ============================================================================

TEST_CASE("roundtrip: embed then project recovers original", "[operation][projection]") {
    // Start with float, embed to double, project back -- should be exact
    vec::dense_vector<float> v(4);
    v(0) = 1.0f; v(1) = -2.5f; v(2) = 3.14f; v(3) = 0.0f;

    auto vd = embed_into<double>(v);
    auto vf = project_onto<float>(vd);

    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(vf(i) == v(i));
}

TEST_CASE("roundtrip: matrix embed then project", "[operation][projection]") {
    mat::dense2D<float> A(2, 2);
    A(0,0) = 1.5f; A(0,1) = -2.0f;
    A(1,0) = 0.0f; A(1,1) = 3.0f;

    auto Ad = embed_into<double>(A);
    auto Af = project_onto<float>(Ad);

    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            REQUIRE(Af(i, j) == A(i, j));
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_CASE("project_onto: empty vector", "[operation][projection]") {
    vec::dense_vector<double> v(0);
    auto vf = project_onto<float>(v);
    REQUIRE(vf.size() == 0);
}

TEST_CASE("project_onto: empty matrix", "[operation][projection]") {
    mat::dense2D<double> A(0, 0);
    auto Af = project_onto<float>(A);
    REQUIRE(Af.num_rows() == 0);
    REQUIRE(Af.num_cols() == 0);
}

TEST_CASE("embed_into: 1x1 matrix", "[operation][projection]") {
    mat::dense2D<float> A(1, 1);
    A(0, 0) = 42.0f;
    auto Ad = embed_into<double>(A);
    REQUIRE(Ad(0, 0) == 42.0);
}

// ============================================================================
// Scaled projection tests (quantization workflows)
// ============================================================================

TEST_CASE("scaled project_onto: float vector to int8 with scale", "[operation][projection][scaled]") {
    vec::dense_vector<float> v(4);
    v(0) = 0.5f; v(1) = -1.0f; v(2) = 0.0f; v(3) = 0.25f;

    double scale = 127.0;
    auto vi = project_onto<int8_t>(v, scale);

    REQUIRE(vi.size() == 4);
    REQUIRE(vi(0) == 63);    // 127 * 0.5 = 63.5 -> 63
    REQUIRE(vi(1) == -127);  // 127 * -1.0 = -127
    REQUIRE(vi(2) == 0);     // 127 * 0.0 = 0
    REQUIRE(vi(3) == 31);    // 127 * 0.25 = 31.75 -> 31
}

TEST_CASE("scaled project_onto: saturation prevents overflow", "[operation][projection][scaled]") {
    vec::dense_vector<float> v(3);
    v(0) = 2.0f;   // 127 * 2.0 = 254 > 127 -> saturate to 127
    v(1) = -2.0f;  // 127 * -2.0 = -254 < -128 -> saturate to -128
    v(2) = 0.5f;   // 127 * 0.5 = 63.5 -> 63 (within range)

    auto vi = project_onto<int8_t>(v, 127.0);

    REQUIRE(vi(0) == 127);   // saturated to max
    REQUIRE(vi(1) == -128);  // saturated to min
    REQUIRE(vi(2) == 63);    // normal conversion
}

TEST_CASE("scaled embed_into: int8 vector to float with inverse scale", "[operation][projection][scaled]") {
    vec::dense_vector<int8_t> vi(3);
    vi(0) = 63; vi(1) = -127; vi(2) = 0;

    double inv_scale = 1.0 / 127.0;
    auto vf = embed_into<float>(vi, inv_scale);

    REQUIRE(vf.size() == 3);
    REQUIRE_THAT(static_cast<double>(vf(0)), Catch::Matchers::WithinAbs(63.0 / 127.0, 1e-6));
    REQUIRE_THAT(static_cast<double>(vf(1)), Catch::Matchers::WithinAbs(-1.0, 1e-6));
    REQUIRE(vf(2) == 0.0f);
}

TEST_CASE("scaled roundtrip: float -> int8 -> float quantization", "[operation][projection][scaled]") {
    vec::dense_vector<float> v(5);
    v(0) = 1.0f; v(1) = -1.0f; v(2) = 0.5f; v(3) = -0.5f; v(4) = 0.0f;

    double scale = 127.0;
    auto quantized = project_onto<int8_t>(v, scale);
    auto recovered = embed_into<float>(quantized, 1.0 / scale);

    // Roundtrip error should be bounded by 1/scale ~ 0.008
    for (std::size_t i = 0; i < 5; ++i)
        REQUIRE_THAT(static_cast<double>(recovered(i)),
                     Catch::Matchers::WithinAbs(static_cast<double>(v(i)), 1.0 / scale + 1e-6));
}

TEST_CASE("scaled project_onto: float matrix to int8", "[operation][projection][scaled]") {
    mat::dense2D<float> A(2, 2);
    A(0,0) = 1.0f; A(0,1) = -0.5f;
    A(1,0) = 0.0f; A(1,1) = 0.75f;

    auto Ai = project_onto<int8_t>(A, 127.0);

    REQUIRE(Ai(0,0) == 127);
    REQUIRE(Ai(0,1) == -63);  // 127 * -0.5 = -63.5 -> -63
    REQUIRE(Ai(1,0) == 0);
    REQUIRE(Ai(1,1) == 95);   // 127 * 0.75 = 95.25 -> 95
}

TEST_CASE("scaled embed_into: int8 matrix to float", "[operation][projection][scaled]") {
    mat::dense2D<int8_t> A(2, 2);
    A(0,0) = 127; A(0,1) = -63;
    A(1,0) = 0;   A(1,1) = 95;

    auto Af = embed_into<float>(A, 1.0 / 127.0);

    REQUIRE_THAT(static_cast<double>(Af(0,0)), Catch::Matchers::WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(static_cast<double>(Af(0,1)), Catch::Matchers::WithinAbs(-63.0/127.0, 1e-6));
}

TEST_CASE("scaled project_onto: double to float with scale=1 is same as unscaled", "[operation][projection][scaled]") {
    vec::dense_vector<double> v(3);
    v(0) = 1.5; v(1) = -2.5; v(2) = 3.14;

    auto vf_unscaled = project_onto<float>(v);
    auto vf_scaled = project_onto<float>(v, 1.0);

    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE(vf_scaled(i) == vf_unscaled(i));
}

// ============================================================================
// Concept enforcement (compile-time)
// ============================================================================

// These would fail to compile if uncommented -- verifying concept constraints:
// auto bad1 = embed_into<float>(double_vector);   // float has fewer digits than double
// auto bad2 = project_onto<double>(float_vector);  // double has more digits than float

TEST_CASE("concept check: ProjectableOnto and EmbeddableInto", "[operation][projection]") {
    // Verify concepts at compile time via static_assert
    static_assert(ProjectableOnto<float, double>);
    static_assert(ProjectableOnto<float, float>);   // same type is fine
    static_assert(ProjectableOnto<double, double>);
    static_assert(!ProjectableOnto<double, float>);  // can't project wider

    static_assert(EmbeddableInto<double, float>);
    static_assert(EmbeddableInto<double, double>);
    static_assert(EmbeddableInto<float, float>);
    static_assert(!EmbeddableInto<float, double>);   // can't embed narrower

    // Integer types: int8_t has 7 digits, float has 24
    static_assert(ProjectableOnto<int8_t, float>);   // 7 <= 24
    static_assert(EmbeddableInto<float, int8_t>);    // 24 >= 7
    static_assert(!ProjectableOnto<float, int8_t>);  // 24 > 7
    static_assert(!EmbeddableInto<int8_t, float>);   // 7 < 24
    REQUIRE(true);  // test needs at least one assertion
}

// ============================================================================
// Integer to integer saturation across signedness (#691)
// ============================================================================
//
// saturating_cast used to cast Target's limits to the source type before
// comparing, which wraps when the signedness differs: uint32 -> int32 clamped
// every value to INT32_MIN, and int -> uint32 clamped every value to UINT32_MAX.

TEST_CASE("scaled project_onto: unsigned to signed of equal width", "[operation][projection][scaled][691]") {
    constexpr auto u32max = std::numeric_limits<uint32_t>::max();
    vec::dense_vector<uint32_t> v = {5u, 100u, 0u, u32max, uint32_t{1} << 31,
                                     static_cast<uint32_t>(std::numeric_limits<int32_t>::max())};
    auto r = project_onto<int32_t>(v, uint32_t{1});
    REQUIRE(r(0) == 5);
    REQUIRE(r(1) == 100);
    REQUIRE(r(2) == 0);
    REQUIRE(r(3) == std::numeric_limits<int32_t>::max());   // saturated
    REQUIRE(r(4) == std::numeric_limits<int32_t>::max());   // 2^31, one past max
    REQUIRE(r(5) == std::numeric_limits<int32_t>::max());   // exactly max
}

TEST_CASE("scaled embed_into: signed to unsigned", "[operation][projection][scaled][691]") {
    // int * int16 is int, so this is int -> uint32.
    vec::dense_vector<int16_t> v = {5, 7, 0, -1, std::numeric_limits<int16_t>::min(),
                                    std::numeric_limits<int16_t>::max()};
    auto r = embed_into<uint32_t>(v, 1);
    REQUIRE(r(0) == 5u);
    REQUIRE(r(1) == 7u);
    REQUIRE(r(2) == 0u);
    REQUIRE(r(3) == 0u);                                     // negative saturates to 0
    REQUIRE(r(4) == 0u);
    REQUIRE(r(5) == 32767u);
}

TEST_CASE("scaled project_onto: narrowing signed to unsigned, int64 to uint16", "[operation][projection][scaled][691]") {
    vec::dense_vector<int64_t> v = {5, -5, 65535, 65536, std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::min()};
    auto r = project_onto<uint16_t>(v, int64_t{1});
    REQUIRE(r(0) == 5);
    REQUIRE(r(1) == 0);
    REQUIRE(r(2) == 65535);
    REQUIRE(r(3) == 65535);
    REQUIRE(r(4) == 65535);
    REQUIRE(r(5) == 0);
}

TEST_CASE("scaled project_onto: narrowing unsigned to signed, uint64 to int8", "[operation][projection][scaled][691]") {
    vec::dense_vector<uint64_t> v = {3u, 127u, 128u, std::numeric_limits<uint64_t>::max()};
    auto r = project_onto<int8_t>(v, uint64_t{2});
    REQUIRE(r(0) == 6);
    REQUIRE(r(1) == 127);                                    // 254 saturates
    REQUIRE(r(2) == 127);
    REQUIRE(r(3) == 127);                                    // 2 * max wraps to max - 1, still > 127
}

TEST_CASE("scaled project_onto: same signedness still saturates", "[operation][projection][scaled][691]") {
    vec::dense_vector<int32_t> v = {-200, 200, 50};
    auto r = project_onto<int8_t>(v, int32_t{1});
    REQUIRE(r(0) == -128);
    REQUIRE(r(1) == 127);
    REQUIRE(r(2) == 50);
    dense2D<uint32_t> m = {{5u, 70000u}, {0u, 1u}};
    auto rm = project_onto<uint16_t>(m, uint32_t{1});
    REQUIRE(rm(0, 0) == 5);
    REQUIRE(rm(0, 1) == 65535);
    REQUIRE(rm(1, 1) == 1);
}

TEST_CASE("saturating_cast to and from character types", "[operation][projection][691]") {
    // std::cmp_less rejects char types; saturating_cast widens them first.
    REQUIRE(detail::saturating_cast<char>(1000) == std::numeric_limits<char>::max());
    REQUIRE(detail::saturating_cast<char>(-1000) == std::numeric_limits<char>::min());
    REQUIRE(detail::saturating_cast<char>(65) == 'A');
    REQUIRE(detail::saturating_cast<uint8_t>('A') == 65);
    REQUIRE(detail::saturating_cast<char16_t>(-1) == char16_t{0});
    REQUIRE(detail::saturating_cast<int>(std::numeric_limits<unsigned long long>::max())
            == std::numeric_limits<int>::max());
}

TEST_CASE("scaled embed_into: widening between signed types", "[operation][projection][scaled][691]") {
    // Old bound cast: static_cast<int>(LLONG_MAX) == -1 and
    // static_cast<int>(LLONG_MIN) == 0, so every positive value clamped to
    // LLONG_MAX and every negative one to LLONG_MIN.
    vec::dense_vector<int32_t> v = {5, -5, 0, std::numeric_limits<int32_t>::max(),
                                    std::numeric_limits<int32_t>::min()};
    auto r = embed_into<long long>(v, 1);
    REQUIRE(r(0) == 5);
    REQUIRE(r(1) == -5);
    REQUIRE(r(2) == 0);
    REQUIRE(r(3) == std::numeric_limits<int32_t>::max());
    REQUIRE(r(4) == std::numeric_limits<int32_t>::min());
    REQUIRE(detail::saturating_cast<long long>(int{5}) == 5);
    // Unsigned widening was unaffected; check it stays so.
    vec::dense_vector<uint16_t> u = {0, 7, std::numeric_limits<uint16_t>::max()};
    auto ru = embed_into<uint64_t>(u, uint64_t{1});
    REQUIRE(ru(1) == 7u);
    REQUIRE(ru(2) == 65535u);
}
