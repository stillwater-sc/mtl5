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
