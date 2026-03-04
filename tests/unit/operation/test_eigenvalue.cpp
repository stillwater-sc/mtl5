#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/hessenberg.hpp>

#include <algorithm>
#include <cmath>
#include <complex>

using namespace mtl;

TEST_CASE("Hessenberg reduction preserves eigenvalues", "[operation][hessenberg]") {
    // 3x3 non-symmetric matrix
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = -1;
    A(1,0) = 2; A(1,1) = 3; A(1,2) =  0;
    A(2,0) = 1; A(2,1) = 0; A(2,2) =  2;

    auto H = hessenberg(A);

    // Verify upper Hessenberg form: H(i,j) == 0 for i > j+1
    for (std::size_t i = 2; i < 3; ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            REQUIRE_THAT(H(i, j), Catch::Matchers::WithinAbs(0.0, 1e-10));

    // Trace and determinant (sum/product of eigenvalues) should be preserved
    double trace_A = A(0,0) + A(1,1) + A(2,2);
    double trace_H = H(0,0) + H(1,1) + H(2,2);
    REQUIRE_THAT(trace_H, Catch::Matchers::WithinAbs(trace_A, 1e-10));
}

TEST_CASE("Symmetric eigenvalue: known 2x2 SPD", "[operation][eigenvalue]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 2; A(0,1) = 1;
    A(1,0) = 1; A(1,1) = 3;

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == 2);

    // Eigenvalues of {{2,1},{1,3}}: (5 +/- sqrt(5))/2
    double expected1 = (5.0 - std::sqrt(5.0)) / 2.0;
    double expected2 = (5.0 + std::sqrt(5.0)) / 2.0;
    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(expected1, 1e-8));
    REQUIRE_THAT(eigs(1), Catch::Matchers::WithinAbs(expected2, 1e-8));
}

TEST_CASE("Symmetric eigenvalue: 3x3 SPD", "[operation][eigenvalue]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == 3);

    // Sum of eigenvalues should equal trace
    double trace = A(0,0) + A(1,1) + A(2,2);
    double eig_sum = eigs(0) + eigs(1) + eigs(2);
    REQUIRE_THAT(eig_sum, Catch::Matchers::WithinAbs(trace, 1e-8));

    // All eigenvalues should be positive (SPD)
    REQUIRE(eigs(0) > 0);
    REQUIRE(eigs(1) > 0);
    REQUIRE(eigs(2) > 0);
}

TEST_CASE("Symmetric eigenvalue: diagonal matrix", "[operation][eigenvalue]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 5; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 3;

    auto eigs = eigenvalue_symmetric(A);
    // Sorted: 1, 3, 5
    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(eigs(1), Catch::Matchers::WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(eigs(2), Catch::Matchers::WithinAbs(5.0, 1e-10));
}

TEST_CASE("General eigenvalue: known 2x2", "[operation][eigenvalue]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0; A(0,1) = 1;
    A(1,0) = -2; A(1,1) = -3;

    auto eigs = eigenvalue(A);
    REQUIRE(eigs.size() == 2);

    // Eigenvalues of {{0,1},{-2,-3}} are -1 and -2
    std::vector<double> real_parts = {eigs(0).real(), eigs(1).real()};
    std::sort(real_parts.begin(), real_parts.end());

    REQUIRE_THAT(real_parts[0], Catch::Matchers::WithinAbs(-2.0, 1e-8));
    REQUIRE_THAT(real_parts[1], Catch::Matchers::WithinAbs(-1.0, 1e-8));
    REQUIRE_THAT(eigs(0).imag(), Catch::Matchers::WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(eigs(1).imag(), Catch::Matchers::WithinAbs(0.0, 1e-8));
}

TEST_CASE("General eigenvalue: complex conjugate pair", "[operation][eigenvalue]") {
    // A = {{0, -1}, {1, 0}} has eigenvalues +/- i
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0; A(0,1) = -1;
    A(1,0) = 1; A(1,1) =  0;

    auto eigs = eigenvalue(A);
    REQUIRE(eigs.size() == 2);

    // Should have purely imaginary eigenvalues +/- i
    std::vector<double> imag_parts = {eigs(0).imag(), eigs(1).imag()};
    std::sort(imag_parts.begin(), imag_parts.end());

    REQUIRE_THAT(eigs(0).real(), Catch::Matchers::WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(eigs(1).real(), Catch::Matchers::WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(imag_parts[0], Catch::Matchers::WithinAbs(-1.0, 1e-8));
    REQUIRE_THAT(imag_parts[1], Catch::Matchers::WithinAbs(1.0, 1e-8));
}
