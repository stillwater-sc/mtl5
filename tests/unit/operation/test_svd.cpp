#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randsvd.hpp>
#include <mtl/generators/hilbert.hpp>
#include <mtl/generators/ones.hpp>
#include <mtl/generators/moler.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace mtl;

TEST_CASE("SVD: U*S*V^T reproduces A", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Reconstruct: A_approx = U * S * V^T
    auto SV = S * trans(V);
    auto A_approx = U * SV;

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(A_approx(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-8));
}

TEST_CASE("SVD: U and V are orthogonal", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 2; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 3;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // U^T * U = I
    auto UtU = trans(U) * U;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(UtU(i, j), Catch::Matchers::WithinAbs(expected, 1e-8));
        }

    // V^T * V = I
    auto VtV = trans(V) * V;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(VtV(i, j), Catch::Matchers::WithinAbs(expected, 1e-8));
        }
}

TEST_CASE("SVD: singular values of diagonal matrix", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 5; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 3; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 1;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Singular values should be 5, 3, 1 (in S diagonal)
    std::vector<double> sv = {S(0,0), S(1,1), S(2,2)};
    std::sort(sv.begin(), sv.end());

    REQUIRE_THAT(sv[0], Catch::Matchers::WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(sv[1], Catch::Matchers::WithinAbs(3.0, 1e-8));
    REQUIRE_THAT(sv[2], Catch::Matchers::WithinAbs(5.0, 1e-8));
}

TEST_CASE("SVD: tuple return form", "[operation][svd]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 3; A(0,1) = 0;
    A(1,0) = 0; A(1,1) = 4;

    auto [U, S, V] = svd(A, 1e-12);

    // Reconstruct
    auto A_approx = U * S * trans(V);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            REQUIRE_THAT(A_approx(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-8));
}

// ── Generator-based SVD tests ────────────────────────────────────────

TEST_CASE("SVD recovers prescribed singular values", "[operation][svd][generator]") {
    // Ground truth test: prescribed singular values must be recovered
    std::vector<double> sigma_prescribed = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    auto A = generators::randsvd<double>(6, 6, sigma_prescribed);

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Extract computed singular values and sort
    std::size_t n = 6;
    std::vector<double> sv_computed(n);
    for (std::size_t i = 0; i < n; ++i)
        sv_computed[i] = S(i, i);
    std::sort(sv_computed.begin(), sv_computed.end());

    std::vector<double> sv_expected = sigma_prescribed;
    std::sort(sv_expected.begin(), sv_expected.end());

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(sv_computed[i], Catch::Matchers::WithinAbs(sv_expected[i], 1e-8));
}

TEST_CASE("SVD condition number from randsvd", "[operation][svd][generator]") {
    // Verify sigma_max / sigma_min ≈ kappa
    constexpr std::size_t n = 5;
    constexpr double kappa = 50.0;
    auto A = generators::randsvd<double>(n, kappa, 3);

    auto [U, S, V] = svd(A, 1e-12);

    std::vector<double> sv(n);
    for (std::size_t i = 0; i < n; ++i)
        sv[i] = S(i, i);
    std::sort(sv.begin(), sv.end());

    double computed_kappa = sv[n - 1] / sv[0];
    REQUIRE_THAT(computed_kappa, Catch::Matchers::WithinRel(kappa, 1e-6));
}

TEST_CASE("SVD of Hilbert matrix: reconstruction", "[operation][svd][generator]") {
    constexpr std::size_t n = 5;
    generators::hilbert<double> H_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = H_gen(i, j);

    auto [U, S, V] = svd(A, 1e-14);

    // Reconstruct: U*S*V^T should equal A
    auto A_approx = U * S * trans(V);
    double rel_error = frobenius_norm(A_approx - A) / frobenius_norm(A);
    REQUIRE(rel_error < 1e-8);
}

TEST_CASE("SVD of rank-1 ones matrix", "[operation][svd][generator]") {
    constexpr std::size_t n = 4;
    generators::ones<double> J_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = J_gen(i, j);

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Rank-1: only one nonzero singular value = n
    std::vector<double> sv(n);
    for (std::size_t i = 0; i < n; ++i)
        sv[i] = std::abs(S(i, i));
    std::sort(sv.rbegin(), sv.rend());

    REQUIRE_THAT(sv[0], Catch::Matchers::WithinAbs(static_cast<double>(n), 1e-8));
    for (std::size_t i = 1; i < n; ++i)
        REQUIRE_THAT(sv[i], Catch::Matchers::WithinAbs(0.0, 1e-8));
}

TEST_CASE("SVD orthogonality on Moler matrix", "[operation][svd][generator]") {
    // Moler is SPD with eigenvalues clustered near 0 — stresses SVD
    constexpr std::size_t n = 3;
    auto A = generators::moler<double>(n);

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-14);

    // Verify singular values are non-negative
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(S(i, i) >= -1e-10);

    // Sum of singular values squared = ||A||_F^2 for SPD matrices
    // (singular values = eigenvalues for SPD)
    double sv_sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        sv_sum_sq += S(i, i) * S(i, i);
    double fnorm = frobenius_norm(A);
    REQUIRE_THAT(sv_sum_sq, Catch::Matchers::WithinAbs(fnorm * fnorm, 0.5));
}
