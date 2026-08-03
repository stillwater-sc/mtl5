#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>

using namespace mtl;

TEST_CASE("compressed2D raw CSR construction and element access", "[mat][compressed2D]") {
    // 3x3 matrix:
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    using size_type = std::size_t;
    size_type starts[]  = {0, 2, 3, 5};
    size_type indices[] = {0, 2, 1, 0, 2};
    double data[]       = {1.0, 2.0, 3.0, 4.0, 5.0};

    mat::compressed2D<double> A(3, 3, 5, starts, indices, data);

    REQUIRE(A.num_rows() == 3);
    REQUIRE(A.num_cols() == 3);
    REQUIRE(A.nnz() == 5);

    // Check existing entries
    REQUIRE(A(0, 0) == 1.0);
    REQUIRE(A(0, 2) == 2.0);
    REQUIRE(A(1, 1) == 3.0);
    REQUIRE(A(2, 0) == 4.0);
    REQUIRE(A(2, 2) == 5.0);

    // Check absent entries return zero
    REQUIRE(A(0, 1) == 0.0);
    REQUIRE(A(1, 0) == 0.0);
    REQUIRE(A(1, 2) == 0.0);
    REQUIRE(A(2, 1) == 0.0);
}

TEST_CASE("compressed2D inserter store mode", "[mat][compressed2D][inserter]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0;
        ins[0][1] << 1.0;
        ins[1][0] << 1.0;
        ins[1][1] << 3.0;
        ins[1][2] << 1.0;
        ins[2][1] << 1.0;
        ins[2][2] << 2.0;
    } // destructor finalizes

    REQUIRE(A.nnz() == 7);
    REQUIRE(A(0, 0) == 4.0);
    REQUIRE(A(0, 1) == 1.0);
    REQUIRE(A(1, 0) == 1.0);
    REQUIRE(A(1, 1) == 3.0);
    REQUIRE(A(1, 2) == 1.0);
    REQUIRE(A(2, 1) == 1.0);
    REQUIRE(A(2, 2) == 2.0);
    REQUIRE(A(0, 2) == 0.0); // absent
    REQUIRE(A(2, 0) == 0.0); // absent
}

TEST_CASE("compressed2D inserter accumulate mode (update_plus)", "[mat][compressed2D][inserter]") {
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>, mat::update_plus<double>> ins(A);
        ins[0][0] << 1.0;
        ins[0][0] << 2.0;  // should accumulate to 3.0
        ins[1][1] << 5.0;
    }

    REQUIRE(A(0, 0) == 3.0);
    REQUIRE(A(1, 1) == 5.0);
}

TEST_CASE("compressed2D satisfies SparseMatrix concept", "[mat][compressed2D][concept]") {
    STATIC_REQUIRE(Matrix<mat::compressed2D<double>>);
    STATIC_REQUIRE(SparseMatrix<mat::compressed2D<double>>);
    STATIC_REQUIRE(!DenseMatrix<mat::compressed2D<double>>);
}

TEST_CASE("Sparse matvec matches dense matvec", "[mat][compressed2D][matvec]") {
    // Build sparse version of:
    // A = {{4,1,0},{1,3,1},{0,1,2}}
    mat::compressed2D<double> As(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(As);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    // Same matrix dense
    mat::dense2D<double> Ad(3, 3);
    Ad(0,0) = 4; Ad(0,1) = 1; Ad(0,2) = 0;
    Ad(1,0) = 1; Ad(1,1) = 3; Ad(1,2) = 1;
    Ad(2,0) = 0; Ad(2,1) = 1; Ad(2,2) = 2;

    vec::dense_vector<double> x = {1.0, 2.0, 3.0};

    auto ys = As * x;
    auto yd = Ad * x;

    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(ys(i), Catch::Matchers::WithinAbs(yd(i), 1e-12));
    }
}

TEST_CASE("Transposed sparse matvec correctness", "[mat][compressed2D][trans][matvec]") {
    // Non-symmetric matrix:
    // A = {{1,2},{3,4},{5,6}}  (3x2)
    mat::compressed2D<double> A(3, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][0] << 3.0; ins[1][1] << 4.0;
        ins[2][0] << 5.0; ins[2][1] << 6.0;
    }

    vec::dense_vector<double> x = {1.0, 2.0, 3.0};

    // trans(A) is 2x3, so trans(A)*x should be length 2
    auto At = trans(A);
    auto y = At * x;

    // trans(A)*x = [1*1+3*2+5*3, 2*1+4*2+6*3] = [22, 28]
    REQUIRE(y.size() == 2);
    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(22.0, 1e-12));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(28.0, 1e-12));
}

// ---------------------------------------------------------------------------
// #355: compressed2D accepted tag::col_major and ignored it.
//
// The container is CSR unconditionally -- starts_ is indexed by row in
// operator(), in the array constructor's assert, in the inserter and in every
// mult kernel -- but nothing ever read Parameters::orientation. A col_major
// instantiation was therefore byte-for-byte a CSR matrix, so a caller who fed
// it genuine CSC arrays silently got the transpose while the constructor
// reported success.
//
// It is now a compile error. The rejection itself is pinned by the
// compile-failure test tests/unit/compile_fail/compressed2d_col_major.cpp,
// which cannot be expressed here; what these cases pin is the other half of the
// contract -- that the row-major layout the container actually implements is
// the one it documents.
// ---------------------------------------------------------------------------

TEST_CASE("compressed2D is explicitly row-major (#355)", "[mat][compressed2D][regression]") {
    static_assert(std::is_same_v<
        typename mat::compressed2D<double>::param_type::orientation,
        tag::row_major>, "the default parameter bundle must be row-major");

    // Spelling row_major explicitly must be accepted and identical to the default.
    using explicit_row = mat::compressed2D<double, mat::parameters<tag::row_major>>;
    static_assert(std::is_same_v<
        typename explicit_row::param_type::orientation, tag::row_major>);

    // The major array is indexed by ROW: its length is nrows+1, not ncols+1.
    // This is the invariant that a col_major instantiation violated silently --
    // the issue's reproducer built a 2x3 matrix and got 3 starts where CSC
    // needs 4.
    const std::size_t nrows = 2, ncols = 3;
    mat::compressed2D<double> A(nrows, ncols);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][1] << 3.0; ins[1][2] << 4.0;
    }
    REQUIRE(A.ref_major().size() == nrows + 1);
    REQUIRE(A.ref_major().size() != ncols + 1);
    REQUIRE(A.nnz() == 4);

    // Row-major reading of the raw arrays: row r occupies [major[r], major[r+1]).
    REQUIRE(A.ref_major()[0] == 0);
    REQUIRE(A.ref_major()[1] == 2);   // row 0 has 2 entries
    REQUIRE(A.ref_major()[2] == 4);   // row 1 has 2 entries

    // Element access agrees with that reading, and is NOT transposed.
    REQUIRE_THAT(A(0, 1), Catch::Matchers::WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(A(1, 2), Catch::Matchers::WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(A(0, 2), Catch::Matchers::WithinAbs(0.0, 1e-12));
}

TEST_CASE("compressed2D built from raw CSR arrays is not transposed (#355)",
          "[mat][compressed2D][regression]") {
    // The issue's second symptom: real CSC arrays fed to a col_major matrix came
    // back as the transpose, with mult returning A^T*x. The same arrays read as
    // CSR -- which is what the container has always done -- must give the
    // row-major matrix, and the two are genuinely different matrices here.
    //
    //   as CSR: [[1,5,0],[1,2,0],[0,3,4]]
    const std::size_t nrows = 3, ncols = 3, nnz = 6;
    std::size_t starts[4]  = {0, 2, 4, 6};
    std::size_t indices[6] = {0, 1, 0, 1, 1, 2};
    double      data[6]    = {1, 5, 1, 2, 3, 4};

    mat::compressed2D<double> A(nrows, ncols, nnz, starts, indices, data);

    REQUIRE_THAT(A(0, 1), Catch::Matchers::WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(A(1, 0), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(A(2, 2), Catch::Matchers::WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(A(0, 2), Catch::Matchers::WithinAbs(0.0, 1e-12));

    // A*x, not A^T*x.  A*[1,2,3] = [1+10, 1+4, 6+12] = [11, 5, 18]
    vec::dense_vector<double> x = {1.0, 2.0, 3.0};
    auto y = A * x;
    REQUIRE(y.size() == nrows);
    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(11.0, 1e-12));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs( 5.0, 1e-12));
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(18.0, 1e-12));

    // And the transposed product, which is what CSC would have been reached for,
    // is available correctly through trans().
    // A^T*[1,2,3] = [1+2, 5+4+9, 12] = [3, 18, 12]
    auto yt = trans(A) * x;
    REQUIRE_THAT(yt(0), Catch::Matchers::WithinAbs( 3.0, 1e-12));
    REQUIRE_THAT(yt(1), Catch::Matchers::WithinAbs(18.0, 1e-12));
    REQUIRE_THAT(yt(2), Catch::Matchers::WithinAbs(12.0, 1e-12));
}
