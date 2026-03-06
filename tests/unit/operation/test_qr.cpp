#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>

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
