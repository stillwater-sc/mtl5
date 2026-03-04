#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/kron.hpp>

using namespace mtl;

TEST_CASE("Kronecker product: 2x2 (x) 2x2", "[operation][kron]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1; A(0,1) = 2;
    A(1,0) = 3; A(1,1) = 4;

    mat::dense2D<double> B(2, 2);
    B(0,0) = 5; B(0,1) = 6;
    B(1,0) = 7; B(1,1) = 8;

    auto C = kron(A, B);

    REQUIRE(C.num_rows() == 4);
    REQUIRE(C.num_cols() == 4);

    // C = [1*B, 2*B; 3*B, 4*B]
    // = [5, 6, 10, 12;
    //    7, 8, 14, 16;
    //   15, 18, 20, 24;
    //   21, 24, 28, 32]
    REQUIRE_THAT(C(0,0), Catch::Matchers::WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(C(0,1), Catch::Matchers::WithinAbs(6.0, 1e-10));
    REQUIRE_THAT(C(0,2), Catch::Matchers::WithinAbs(10.0, 1e-10));
    REQUIRE_THAT(C(0,3), Catch::Matchers::WithinAbs(12.0, 1e-10));
    REQUIRE_THAT(C(1,0), Catch::Matchers::WithinAbs(7.0, 1e-10));
    REQUIRE_THAT(C(1,1), Catch::Matchers::WithinAbs(8.0, 1e-10));
    REQUIRE_THAT(C(1,2), Catch::Matchers::WithinAbs(14.0, 1e-10));
    REQUIRE_THAT(C(1,3), Catch::Matchers::WithinAbs(16.0, 1e-10));
    REQUIRE_THAT(C(2,0), Catch::Matchers::WithinAbs(15.0, 1e-10));
    REQUIRE_THAT(C(2,1), Catch::Matchers::WithinAbs(18.0, 1e-10));
    REQUIRE_THAT(C(2,2), Catch::Matchers::WithinAbs(20.0, 1e-10));
    REQUIRE_THAT(C(2,3), Catch::Matchers::WithinAbs(24.0, 1e-10));
    REQUIRE_THAT(C(3,0), Catch::Matchers::WithinAbs(21.0, 1e-10));
    REQUIRE_THAT(C(3,1), Catch::Matchers::WithinAbs(24.0, 1e-10));
    REQUIRE_THAT(C(3,2), Catch::Matchers::WithinAbs(28.0, 1e-10));
    REQUIRE_THAT(C(3,3), Catch::Matchers::WithinAbs(32.0, 1e-10));
}

TEST_CASE("Kronecker product: I (x) A = block diagonal", "[operation][kron]") {
    mat::dense2D<double> I(2, 2);
    I(0,0) = 1; I(0,1) = 0;
    I(1,0) = 0; I(1,1) = 1;

    mat::dense2D<double> A(2, 2);
    A(0,0) = 3; A(0,1) = 4;
    A(1,0) = 5; A(1,1) = 6;

    auto C = kron(I, A);

    REQUIRE(C.num_rows() == 4);
    REQUIRE(C.num_cols() == 4);

    // Should be block diagonal: [A, 0; 0, A]
    REQUIRE_THAT(C(0,0), Catch::Matchers::WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(C(0,2), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(C(2,0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(C(2,2), Catch::Matchers::WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(C(3,3), Catch::Matchers::WithinAbs(6.0, 1e-10));
}
