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

// ---------------------------------------------------------------------------
// #355: ell_matrix accepted tag::col_major and ignored it.
//
// Both arrays are nrows*width and every access is indices_[r*width_ + k], so a
// col_major instantiation was byte-for-byte the row-major layout. It is now a
// compile error, pinned by tests/unit/compile_fail/ell_matrix_col_major.cpp.
// What is pinned here is the layout the container actually implements.
// ---------------------------------------------------------------------------

TEST_CASE("ell_matrix is explicitly row-padded (#355)", "[mat][ell_matrix][regression]") {
    static_assert(std::is_same_v<
        typename mat::ell_matrix<double>::param_type::orientation,
        tag::row_major>, "the default parameter bundle must be row-major");

    // Storage is nrows*width -- padded per ROW. A column-padded ELL of the same
    // matrix would be ncols*width, and for a non-square matrix those differ,
    // which is what a col_major instantiation silently misdescribed.
    const std::size_t nrows = 2, ncols = 3, width = 2;
    mat::ell_matrix<double> A(nrows, ncols, width);
    REQUIRE(A.ref_indices().size() == nrows * width);
    REQUIRE(A.ref_indices().size() != ncols * width);
    REQUIRE(A.ref_data().size()    == nrows * width);

    // Round-tripping a non-square CSR keeps row semantics, not the transpose.
    //   B = [[1,2,0],
    //        [0,3,4]]
    mat::compressed2D<double> B(nrows, ncols);
    {
        mat::inserter<mat::compressed2D<double>> ins(B);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][1] << 3.0; ins[1][2] << 4.0;
    }
    mat::ell_matrix<double> E(B);
    REQUIRE(E.num_rows() == nrows);
    REQUIRE(E.num_cols() == ncols);
    REQUIRE(E.ref_indices().size() == E.num_rows() * E.max_width());

    for (std::size_t i = 0; i < nrows; ++i)
        for (std::size_t j = 0; j < ncols; ++j) {
            INFO("i=" << i << " j=" << j);
            REQUIRE(E(i, j) == B(i, j));
        }
    // Explicitly not transposed.
    REQUIRE(E(0, 1) == 2.0);
    REQUIRE(E(1, 2) == 4.0);
    REQUIRE(E(0, 2) == 0.0);
}
