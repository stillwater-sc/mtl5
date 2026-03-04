#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/ell_matrix.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

static mat::compressed2D<double> make_tridiag(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0)     ins[i][i-1] << -1.0;
            if (i < n - 1) ins[i][i+1] << -1.0;
        }
    }
    return A;
}

TEST_CASE("ell_matrix: construct from compressed2D", "[mat][ell_matrix]") {
    auto crs = make_tridiag(5);
    mat::ell_matrix<double> ell(crs);

    REQUIRE(ell.num_rows() == 5);
    REQUIRE(ell.num_cols() == 5);
    REQUIRE(ell.max_width() == 3);  // interior rows have 3 elements
}

TEST_CASE("ell_matrix: element access matches CRS", "[mat][ell_matrix]") {
    auto crs = make_tridiag(4);
    mat::ell_matrix<double> ell(crs);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE_THAT(ell(i, j), Catch::Matchers::WithinAbs(crs(i, j), 1e-10));
}

TEST_CASE("ell_matrix: absent elements return zero", "[mat][ell_matrix]") {
    auto crs = make_tridiag(5);
    mat::ell_matrix<double> ell(crs);

    REQUIRE_THAT(ell(0, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ell(0, 4), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ell(4, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("ell_matrix: manual construction", "[mat][ell_matrix]") {
    mat::ell_matrix<double> ell(2, 2, 2);

    // Default: all zeros
    REQUIRE_THAT(ell(0, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(ell(1, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
}
