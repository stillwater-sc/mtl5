#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <complex>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("dot product of double vectors", "[operation][dot]") {
    dense_vector<double> a = {1.0, 2.0, 3.0};
    dense_vector<double> b = {4.0, 5.0, 6.0};
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    REQUIRE(dot(a, b) == 32.0);
}

TEST_CASE("dot product of int vectors", "[operation][dot]") {
    dense_vector<int> a = {1, 2, 3};
    dense_vector<int> b = {2, 3, 4};
    // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
    REQUIRE(dot(a, b) == 20);
}

TEST_CASE("dot product of orthogonal vectors is zero", "[operation][dot]") {
    dense_vector<double> a = {1.0, 0.0, 0.0};
    dense_vector<double> b = {0.0, 1.0, 0.0};
    REQUIRE(dot(a, b) == 0.0);
}

TEST_CASE("dot product with uniform vectors", "[operation][dot]") {
    dense_vector<double> a(3, 1.0);
    dense_vector<double> b(3, 2.0);
    REQUIRE(dot(a, b) == 6.0);
}

TEST_CASE("dot product of complex vectors is Hermitian", "[operation][dot]") {
    using cd = std::complex<double>;
    dense_vector<cd> a = {cd(1.0, 1.0), cd(2.0, -1.0)};
    dense_vector<cd> b = {cd(3.0, 0.0), cd(0.0, 1.0)};
    // conj(a[0])*b[0] + conj(a[1])*b[1]
    // = (1-i)*(3+0i) + (2+i)*(0+i)
    // = (3-3i) + (0+2i+0-1) = (3-3i) + (-1+2i) = (2-i)
    auto result = dot(a, b);
    REQUIRE_THAT(result.real(), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(result.imag(), WithinAbs(-1.0, 1e-10));
}

TEST_CASE("dot_real skips conjugation", "[operation][dot]") {
    using cd = std::complex<double>;
    dense_vector<cd> a = {cd(1.0, 1.0), cd(2.0, -1.0)};
    dense_vector<cd> b = {cd(3.0, 0.0), cd(0.0, 1.0)};
    // a[0]*b[0] + a[1]*b[1]
    // = (1+i)*(3+0i) + (2-i)*(0+i)
    // = (3+3i) + (0+2i-0+1) = (3+3i) + (1+2i) = (4+5i)
    auto result = dot_real(a, b);
    REQUIRE_THAT(result.real(), WithinAbs(4.0, 1e-10));
    REQUIRE_THAT(result.imag(), WithinAbs(5.0, 1e-10));
}

TEST_CASE("dot product of single-element vectors", "[operation][dot]") {
    dense_vector<double> a = {5.0};
    dense_vector<double> b = {3.0};
    REQUIRE(dot(a, b) == 15.0);
}
