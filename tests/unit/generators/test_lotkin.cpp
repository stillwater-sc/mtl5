#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/lotkin.hpp>
#include <mtl/generators/hilbert.hpp>
#include <mtl/concepts/matrix.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("lotkin satisfies Matrix concept", "[generators][lotkin]") {
    STATIC_REQUIRE(Matrix<generators::lotkin<double>>);
}

TEST_CASE("lotkin first row is all ones", "[generators][lotkin]") {
    generators::lotkin<double> L(5);
    for (std::size_t j = 0; j < 5; ++j)
        REQUIRE_THAT(L(0, j), WithinAbs(1.0, 1e-15));
}

TEST_CASE("lotkin rows 1..n match hilbert", "[generators][lotkin]") {
    std::size_t n = 5;
    generators::lotkin<double> L(n);
    generators::hilbert<double> H(n);

    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(L(i, j), WithinAbs(H(i, j), 1e-15));
}

TEST_CASE("lotkin is not symmetric", "[generators][lotkin]") {
    generators::lotkin<double> L(4);
    // Row 0: all 1s. But L(1,0) = 1/(1+0+1) = 0.5 != L(0,1) = 1
    // So L(0,1) != L(1,0) for n >= 2
    REQUIRE_THAT(L(0, 1), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(L(1, 0), WithinAbs(0.5, 1e-15));
}
