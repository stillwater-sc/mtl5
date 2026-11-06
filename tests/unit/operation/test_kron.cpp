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

// -- Mixed value types (#694) ----------------------------------------------
//
// kron used to take its value type from A alone, so int (x) double was
// computed in int: kron({2}, {0.25}) gave 0.

#include <complex>
#include <cstdint>
#include <type_traits>

TEST_CASE("Kronecker product: int (x) double is double", "[operation][kron][694]") {
    mat::dense2D<int> A(1, 1);
    A(0, 0) = 2;
    mat::dense2D<double> B(1, 1);
    B(0, 0) = 0.25;
    auto C = kron(A, B);
    static_assert(std::is_same_v<typename decltype(C)::value_type, double>);
    REQUIRE(C(0, 0) == 0.5);
    // Either order.
    auto D = kron(B, A);
    static_assert(std::is_same_v<typename decltype(D)::value_type, double>);
    REQUIRE(D(0, 0) == 0.5);
}

TEST_CASE("Kronecker product: float (x) double is double", "[operation][kron][694]") {
    mat::dense2D<float> A(1, 2);
    A(0, 0) = 1.0f; A(0, 1) = 3.0f;
    mat::dense2D<double> B(1, 1);
    B(0, 0) = 0.1;                        // not representable in float
    auto C = kron(A, B);
    static_assert(std::is_same_v<typename decltype(C)::value_type, double>);
    REQUIRE(C(0, 0) == 1.0f * 0.1);
    REQUIRE(C(0, 1) == 3.0f * 0.1);
    REQUIRE(C(0, 1) != static_cast<double>(static_cast<float>(3.0f * 0.1)));
}

TEST_CASE("Kronecker product: double (x) complex<double> is complex", "[operation][kron][694]") {
    mat::dense2D<double> A(1, 1);
    A(0, 0) = 2.0;
    mat::dense2D<std::complex<double>> B(1, 1);
    B(0, 0) = {0.0, 1.5};
    auto C = kron(A, B);
    static_assert(std::is_same_v<typename decltype(C)::value_type, std::complex<double>>);
    REQUIRE(C(0, 0) == std::complex<double>(0.0, 3.0));
}

TEST_CASE("Kronecker product: same value type is unchanged", "[operation][kron][694]") {
    mat::dense2D<double> Ad(1, 1), Bd(1, 1);
    Ad(0, 0) = 3.0; Bd(0, 0) = 0.5;
    auto Cd = kron(Ad, Bd);
    static_assert(std::is_same_v<typename decltype(Cd)::value_type, double>);
    REQUIRE(Cd(0, 0) == 1.5);

    // int8 (x) int8 stays int8, although int8 * int8 is int; the product
    // converts back to int8 as it always did.
    mat::dense2D<std::int8_t> A8(1, 2), B8(1, 1);
    A8(0, 0) = 3; A8(0, 1) = -4; B8(0, 0) = 5;
    auto C8 = kron(A8, B8);
    static_assert(std::is_same_v<typename decltype(C8)::value_type, std::int8_t>);
    REQUIRE(C8(0, 0) == 15);
    REQUIRE(C8(0, 1) == -20);
}

namespace kron_test {
// Two custom types with a product but no std::common_type: kron keeps A's
// value type, as it did before #694, instead of failing to compile.
struct ka { double v; };
struct kb { double v; };
inline ka operator*(ka a, kb b) { return {a.v * b.v}; }
} // namespace kron_test

TEST_CASE("Kronecker product: no common type keeps A's value type", "[operation][kron][694]") {
    using kron_test::ka;
    using kron_test::kb;
    mat::dense2D<ka> A(1, 1);
    A(0, 0) = ka{2.0};
    mat::dense2D<kb> B(1, 1);
    B(0, 0) = kb{0.25};
    auto C = kron(A, B);
    static_assert(std::is_same_v<typename decltype(C)::value_type, ka>);
    REQUIRE(C(0, 0).v == 0.5);
}
