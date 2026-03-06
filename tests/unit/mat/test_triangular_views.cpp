#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/view/upper_view.hpp>
#include <mtl/mat/view/lower_view.hpp>
#include <mtl/mat/view/strict_upper_view.hpp>
#include <mtl/mat/view/strict_lower_view.hpp>

using namespace mtl;

// Helper: fill a 4x4 matrix with values 1..16
static mat::dense2D<double> make_4x4() {
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);
    return A;
}

// ── upper_view tests ────────────────────────────────────────────────────

TEST_CASE("upper_view: includes diagonal", "[mat][view][upper]") {
    auto A = make_4x4();
    auto U = upper(A);

    REQUIRE(U.num_rows() == 4);
    REQUIRE(U.num_cols() == 4);

    // Diagonal: should match A
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(U(i, i), Catch::Matchers::WithinAbs(A(i, i), 1e-10));

    // Upper triangle: should match A
    REQUIRE_THAT(U(0, 1), Catch::Matchers::WithinAbs(A(0, 1), 1e-10));
    REQUIRE_THAT(U(0, 3), Catch::Matchers::WithinAbs(A(0, 3), 1e-10));
    REQUIRE_THAT(U(1, 2), Catch::Matchers::WithinAbs(A(1, 2), 1e-10));

    // Below diagonal: should be zero
    REQUIRE_THAT(U(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(U(2, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(U(2, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(U(3, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(U(3, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(U(3, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

// ── lower_view tests ────────────────────────────────────────────────────

TEST_CASE("lower_view: includes diagonal", "[mat][view][lower]") {
    auto A = make_4x4();
    auto L = lower(A);

    // Diagonal: should match A
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(L(i, i), Catch::Matchers::WithinAbs(A(i, i), 1e-10));

    // Lower triangle: should match A
    REQUIRE_THAT(L(1, 0), Catch::Matchers::WithinAbs(A(1, 0), 1e-10));
    REQUIRE_THAT(L(3, 0), Catch::Matchers::WithinAbs(A(3, 0), 1e-10));
    REQUIRE_THAT(L(2, 1), Catch::Matchers::WithinAbs(A(2, 1), 1e-10));

    // Above diagonal: should be zero
    REQUIRE_THAT(L(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(L(0, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(L(0, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(L(1, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(L(1, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(L(2, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

// ── strict_upper_view tests ─────────────────────────────────────────────

TEST_CASE("strict_upper_view: excludes diagonal", "[mat][view][strict_upper]") {
    auto A = make_4x4();
    auto SU = strict_upper(A);

    // Diagonal: should be zero
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(SU(i, i), Catch::Matchers::WithinAbs(0.0, 1e-10));

    // Strictly above diagonal: should match A
    REQUIRE_THAT(SU(0, 1), Catch::Matchers::WithinAbs(A(0, 1), 1e-10));
    REQUIRE_THAT(SU(0, 3), Catch::Matchers::WithinAbs(A(0, 3), 1e-10));
    REQUIRE_THAT(SU(1, 2), Catch::Matchers::WithinAbs(A(1, 2), 1e-10));

    // Below diagonal: should be zero
    REQUIRE_THAT(SU(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(SU(3, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

// ── strict_lower_view tests ─────────────────────────────────────────────

TEST_CASE("strict_lower_view: excludes diagonal", "[mat][view][strict_lower]") {
    auto A = make_4x4();
    auto SL = strict_lower(A);

    // Diagonal: should be zero
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(SL(i, i), Catch::Matchers::WithinAbs(0.0, 1e-10));

    // Strictly below diagonal: should match A
    REQUIRE_THAT(SL(1, 0), Catch::Matchers::WithinAbs(A(1, 0), 1e-10));
    REQUIRE_THAT(SL(3, 0), Catch::Matchers::WithinAbs(A(3, 0), 1e-10));
    REQUIRE_THAT(SL(2, 1), Catch::Matchers::WithinAbs(A(2, 1), 1e-10));

    // Above diagonal: should be zero
    REQUIRE_THAT(SL(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(SL(0, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

// ── triu / tril MATLAB-compat tests ─────────────────────────────────────

TEST_CASE("triu(A, 0) equals upper", "[mat][view][triu]") {
    auto A = make_4x4();
    auto T = triu(A, 0);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            double expected = (j >= i) ? A(i, j) : 0.0;
            REQUIRE_THAT(T(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("triu(A, 1) equals strict_upper", "[mat][view][triu]") {
    auto A = make_4x4();
    auto T = triu(A, 1);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            double expected = (j > i) ? A(i, j) : 0.0;
            REQUIRE_THAT(T(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("tril(A, 0) equals lower", "[mat][view][tril]") {
    auto A = make_4x4();
    auto T = tril(A, 0);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            double expected = (i >= j) ? A(i, j) : 0.0;
            REQUIRE_THAT(T(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("tril(A, -1) equals strict_lower", "[mat][view][tril]") {
    auto A = make_4x4();
    auto T = tril(A, -1);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            double expected = (i > j) ? A(i, j) : 0.0;
            REQUIRE_THAT(T(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("triu(A,2) keeps only second superdiagonal and above", "[mat][view][triu]") {
    auto A = make_4x4();
    auto T = triu(A, 2);

    // Only elements where c - r >= 2
    REQUIRE_THAT(T(0, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(T(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(T(0, 2), Catch::Matchers::WithinAbs(A(0, 2), 1e-10)); // c-r=2
    REQUIRE_THAT(T(0, 3), Catch::Matchers::WithinAbs(A(0, 3), 1e-10)); // c-r=3
    REQUIRE_THAT(T(1, 3), Catch::Matchers::WithinAbs(A(1, 3), 1e-10)); // c-r=2
    REQUIRE_THAT(T(1, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));     // c-r=1
}

// ── upper + lower decomposition ─────────────────────────────────────────

TEST_CASE("upper + strict_lower = original matrix", "[mat][view][decomposition]") {
    auto A = make_4x4();
    auto U  = upper(A);
    auto SL = strict_lower(A);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE_THAT(U(i, j) + SL(i, j),
                         Catch::Matchers::WithinAbs(A(i, j), 1e-10));
}
