#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randspd.hpp>
#include <mtl/generators/pascal.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/lehmer.hpp>

using namespace mtl;

TEST_CASE("LDL^T factorization: L*D*L^T reproduces A", "[operation][ldlt]") {
    // SPD matrix: A = {{4,2,1},{2,5,3},{1,3,6}}
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // Extract L (unit lower triangular) and D from A
    mat::dense2D<double> L(3, 3);
    mat::dense2D<double> D(3, 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (i == j) {
                L(i, j) = 1.0;  // unit diagonal
                D(i, j) = A(i, j);  // D on diagonal of A
            } else if (j < i) {
                L(i, j) = A(i, j);  // strictly lower triangle
                D(i, j) = 0.0;
            } else {
                L(i, j) = 0.0;
                D(i, j) = 0.0;
            }
        }
    }

    // L * D * L^T should equal Aorig
    auto LDLt = L * D * trans(L);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(LDLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("LDL^T solve", "[operation][ldlt]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);
    ldlt_solve(A, x, b);

    // Verify Aorig * x = b
    auto r = Aorig * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("LDL^T handles symmetric indefinite matrix", "[operation][ldlt]") {
    // Symmetric indefinite: eigenvalues of different signs
    // A = {{1, 2}, {2, -1}}  eigenvalues: +/- sqrt(5)
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1;  A(0,1) = 2;
    A(1,0) = 2;  A(1,1) = -1;

    mat::dense2D<double> Aorig(2, 2);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // D should have entries of different signs
    // D(0) = 1, D(1) = -1 - 4/1 = -5
    REQUIRE_THAT(A(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(A(1, 1), Catch::Matchers::WithinAbs(-5.0, 1e-12));

    // Solve
    vec::dense_vector<double> b = {5.0, 3.0};
    vec::dense_vector<double> x(2);
    ldlt_solve(A, x, b);

    auto r = Aorig * x;
    for (std::size_t i = 0; i < 2; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("LDL^T detects zero pivot", "[operation][ldlt]") {
    // Singular: A = {{0, 1}, {1, 0}}
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0;  A(0,1) = 1;
    A(1,0) = 1;  A(1,1) = 0;

    int info = ldlt_factor(A);
    REQUIRE(info != 0);  // D(0,0) = 0 → returns 1
    REQUIRE(info == 1);
}

TEST_CASE("LDL^T on randspd with known eigenvalues", "[operation][ldlt][generator]") {
    constexpr std::size_t n = 5;
    auto A = generators::randspd<double>(n, {8.0, 4.0, 2.0, 1.0, 0.5});

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // Extract L and D, verify L*D*L^T = A
    mat::dense2D<double> L(n, n);
    mat::dense2D<double> D(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                L(i, j) = 1.0;
                D(i, j) = A(i, j);
            } else if (j < i) {
                L(i, j) = A(i, j);
                D(i, j) = 0.0;
            } else {
                L(i, j) = 0.0;
                D(i, j) = 0.0;
            }
        }
    }

    auto LDLt = L * D * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LDLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("LDL^T on Pascal matrix", "[operation][ldlt][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::pascal<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // Pascal LDL^T: L should have integer entries (Pascal's triangle)
    // and D should be all 1s
    for (std::size_t j = 0; j < n; ++j)
        REQUIRE_THAT(A(j, j), Catch::Matchers::WithinAbs(1.0, 1e-10));
}

TEST_CASE("LDL^T on Moler matrix", "[operation][ldlt][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::moler<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    mat::dense2D<double> L(n, n);
    mat::dense2D<double> D(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) { L(i, j) = 1.0; D(i, j) = A(i, j); }
            else if (j < i) { L(i, j) = A(i, j); D(i, j) = 0.0; }
            else { L(i, j) = 0.0; D(i, j) = 0.0; }
        }
    }

    auto LDLt = L * D * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LDLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

TEST_CASE("LDL^T on Lehmer matrix", "[operation][ldlt][generator]") {
    constexpr std::size_t n = 6;
    generators::lehmer<double> L_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = L_gen(i, j);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // All D entries should be positive (Lehmer is SPD)
    for (std::size_t j = 0; j < n; ++j)
        REQUIRE(A(j, j) > 0.0);

    // Verify via solve
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n);
    ldlt_solve(A, x, b);

    auto r = Aorig * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("LDL^T on 1x1 matrix", "[operation][ldlt]") {
    mat::dense2D<double> A(1, 1);
    A(0, 0) = 7.0;

    int info = ldlt_factor(A);
    REQUIRE(info == 0);
    REQUIRE_THAT(A(0, 0), Catch::Matchers::WithinAbs(7.0, 1e-14));

    // Re-create for solve (factor overwrites D on diagonal)
    mat::dense2D<double> Af(1, 1);
    Af(0, 0) = 7.0;
    ldlt_factor(Af);

    vec::dense_vector<double> b = {21.0};
    vec::dense_vector<double> x(1);
    ldlt_solve(Af, x, b);
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("LDL^T on empty (0x0) matrix", "[operation][ldlt]") {
    mat::dense2D<double> A(0, 0);
    int info = ldlt_factor(A);
    REQUIRE(info == 0);
}
