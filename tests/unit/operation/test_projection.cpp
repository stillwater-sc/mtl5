#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
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
    REQUIRE(true);  // test needs at least one assertion
}
