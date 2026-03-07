#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/kahan.hpp>
#include <mtl/generators/randorth.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/vandermonde.hpp>
#include <mtl/generators/randsvd.hpp>

#include <cmath>
#include <vector>

using namespace mtl;

TEST_CASE("QR factorization: Q*R reproduces A", "[operation][qr]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 12; A(0,1) = -51; A(0,2) =   4;
    A(1,0) =  6; A(1,1) = 167; A(1,2) = -68;
    A(2,0) = -4; A(2,1) =  24; A(2,2) = -41;

    // Save original
    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(3);
    int info = qr_factor(A, tau);
    REQUIRE(info == 0);

    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Verify Q*R = A
    auto QR = Q * R;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(QR(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

TEST_CASE("QR factorization: Q is orthogonal", "[operation][qr]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) =  3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) =  6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    vec::dense_vector<double> tau(3);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);

    // Q^T * Q should be I
    auto QtQ = trans(Q) * Q;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(QtQ(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("QR solve: least-squares for square system", "[operation][qr]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) =  3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) =  6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    // Save original
    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);

    vec::dense_vector<double> tau(3);
    qr_factor(A, tau);
    qr_solve(A, tau, x, b);

    // Verify Aorig * x = b
    auto r = Aorig * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("LQ factorization: L*Q reproduces A", "[operation][lq]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 12; A(0,1) = -51; A(0,2) =   4;
    A(1,0) =  6; A(1,1) = 167; A(1,2) = -68;
    A(2,0) = -4; A(2,1) =  24; A(2,2) = -41;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(3);
    int info = lq_factor(A, tau);
    REQUIRE(info == 0);

    auto L = lq_extract_L(A);
    auto Q = lq_extract_Q(A, tau);

    auto LQ = L * Q;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(LQ(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

// ── Generator-based QR tests ─────────────────────────────────────────

TEST_CASE("QR on Kahan matrix", "[operation][qr][generator]") {
    // Kahan is upper triangular + ill-conditioned — classic QR stress test
    constexpr std::size_t n = 6;
    auto A = generators::kahan<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(n);
    int info = qr_factor(A, tau);
    REQUIRE(info == 0);

    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Verify Q*R = A (reconstruction)
    auto QR = Q * R;
    double residual = frobenius_norm(QR - Aorig);
    REQUIRE(residual / frobenius_norm(Aorig) < 1e-8);
}

TEST_CASE("QR orthogonality with randorth", "[operation][qr][generator]") {
    // QR of an already-orthogonal matrix: Q should be orthogonal
    constexpr std::size_t n = 8;
    auto Qorig = generators::randorth<double>(n);

    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = Qorig(i, j);

    vec::dense_vector<double> tau(n);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);

    // Q^T * Q should be I
    auto QtQ = trans(Q) * Q;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(QtQ(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("QR on Frank (Hessenberg) matrix", "[operation][qr][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::frank<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(n);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Q*R = A
    auto QR = Q * R;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(QR(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));

    // R should be upper triangular
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            REQUIRE_THAT(R(i, j), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("QR reconstruction on Vandermonde", "[operation][qr][generator]") {
    // Vandermonde is ill-conditioned — verify Q*R = A reconstruction
    auto A = generators::vandermonde<double>({1.0, 2.0, 3.0, 4.0, 5.0});
    std::size_t n = 5;

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(n);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    auto QR = Q * R;
    double rel_error = frobenius_norm(QR - Aorig) / frobenius_norm(Aorig);
    REQUIRE(rel_error < 1e-8);
}

TEST_CASE("QR on randsvd with known condition number", "[operation][qr][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::randsvd<double>(n, 100.0, 3);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(n);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Reconstruction accuracy
    auto QR = Q * R;
    double rel_error = frobenius_norm(QR - Aorig) / frobenius_norm(Aorig);
    REQUIRE(rel_error < 1e-8);

    // Q must be orthogonal
    auto QtQ = trans(Q) * Q;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(QtQ(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}
