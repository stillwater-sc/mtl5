#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/math/identity.hpp>

using namespace mtl;

// Helper: max eigenpair residual ||Av - λv|| / (||A|| * ||v||)
static double max_eigenpair_residual(const mat::dense2D<double>& A,
                                      const vec::dense_vector<double>& eigs,
                                      const mat::dense2D<double>& V) {
    std::size_t n = A.num_rows();
    double Anorm = frobenius_norm(A);
    double max_res = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        // Compute Av - λv
        double res_sq = 0.0;
        double v_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double Av_i = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                Av_i += A(i, j) * V(j, k);
            double ri = Av_i - eigs(k) * V(i, k);
            res_sq += ri * ri;
            v_sq += V(i, k) * V(i, k);
        }
        double res = std::sqrt(res_sq) / (Anorm * std::sqrt(v_sq));
        if (res > max_res) max_res = res;
    }
    return max_res;
}

// Helper: orthonormality error ||V^T V - I||_F
static double orthonormality_error(const mat::dense2D<double>& V) {
    std::size_t n = V.num_rows();
    auto VtV = trans(V) * V;
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double d = VtV(i, j) - expected;
            err += d * d;
        }
    return std::sqrt(err);
}

TEST_CASE("eigen_symmetric: 2x2 diagonal matrix", "[operation][eigen]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 3; A(0,1) = 0;
    A(1,0) = 0; A(1,1) = 1;

    auto [eigs, V] = eigen_symmetric(A);

    REQUIRE(eigs.size() == 2);
    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(eigs(1), Catch::Matchers::WithinAbs(3.0, 1e-10));

    REQUIRE(orthonormality_error(V) < 1e-12);
    REQUIRE(max_eigenpair_residual(A, eigs, V) < 1e-12);
}

TEST_CASE("eigen_symmetric: 3x3 SPD matrix", "[operation][eigen]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    auto [eigs, V] = eigen_symmetric(A);

    REQUIRE(eigs.size() == 3);
    // Eigenvalues should be positive (SPD)
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE(eigs(i) > 0.0);

    // Sorted ascending
    for (std::size_t i = 0; i + 1 < 3; ++i)
        REQUIRE(eigs(i) <= eigs(i + 1));

    // Sum of eigenvalues = trace
    double trace = A(0,0) + A(1,1) + A(2,2);
    double eig_sum = eigs(0) + eigs(1) + eigs(2);
    REQUIRE_THAT(eig_sum, Catch::Matchers::WithinAbs(trace, 1e-10));

    REQUIRE(orthonormality_error(V) < 1e-12);
    REQUIRE(max_eigenpair_residual(A, eigs, V) < 1e-12);
}

TEST_CASE("eigen_symmetric: spectral reconstruction A = V*Lambda*V^T", "[operation][eigen]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 2; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 4;

    auto [eigs, V] = eigen_symmetric(A);

    // Reconstruct: A_rec = V * diag(eigs) * V^T
    std::size_t n = 3;
    mat::dense2D<double> Lambda(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Lambda(i, j) = (i == j) ? eigs(i) : 0.0;

    auto A_rec = V * Lambda * trans(V);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(A_rec(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-10));
}

TEST_CASE("eigen_symmetric: 1x1 matrix", "[operation][eigen]") {
    mat::dense2D<double> A(1, 1);
    A(0, 0) = 7.5;

    auto [eigs, V] = eigen_symmetric(A);

    REQUIRE(eigs.size() == 1);
    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(7.5, 1e-12));
    REQUIRE_THAT(V(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("eigen_symmetric: 4x4 with known eigenvalues", "[operation][eigen]") {
    // Diagonal: eigenvalues are the diagonal entries
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = 0.0;
    A(0,0) = 10; A(1,1) = 1; A(2,2) = 5; A(3,3) = 3;

    auto [eigs, V] = eigen_symmetric(A);

    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(eigs(1), Catch::Matchers::WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(eigs(2), Catch::Matchers::WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(eigs(3), Catch::Matchers::WithinAbs(10.0, 1e-10));

    REQUIRE(orthonormality_error(V) < 1e-12);
    REQUIRE(max_eigenpair_residual(A, eigs, V) < 1e-12);
}

TEST_CASE("eigen_symmetric: 6x6 Lehmer matrix", "[operation][eigen]") {
    constexpr std::size_t n = 6;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = double(std::min(i, j) + 1) / double(std::max(i, j) + 1);

    auto [eigs, V] = eigen_symmetric(A);

    // All eigenvalues positive (Lehmer is SPD)
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(eigs(i) > 0.0);

    double tol = double(n) * 1e-14;
    REQUIRE(orthonormality_error(V) < tol);
    REQUIRE(max_eigenpair_residual(A, eigs, V) < tol);
}

TEST_CASE("eigen_symmetric: backward compat eigenvalue_symmetric still works", "[operation][eigen]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == 3);
    double trace = 15.0;
    double sum = eigs(0) + eigs(1) + eigs(2);
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(trace, 1e-10));
}
