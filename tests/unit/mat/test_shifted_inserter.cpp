#include <catch2/catch_test_macros.hpp>
#include <mtl/mat/shifted_inserter.hpp>
#include <mtl/mat/inserter.hpp>

using namespace mtl;

TEST_CASE("shifted_inserter with zero offset = normal insertion", "[mat][shifted_inserter]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::shifted_inserter<mat::inserter<mat::compressed2D<double>>> ins(A, 3, 0, 0);
        ins[0][0] << 1.0;
        ins[1][1] << 2.0;
        ins[2][2] << 3.0;
    }
    REQUIRE(A(0, 0) == 1.0);
    REQUIRE(A(1, 1) == 2.0);
    REQUIRE(A(2, 2) == 3.0);
    REQUIRE(A(0, 1) == 0.0);
}

TEST_CASE("shifted_inserter with row offset", "[mat][shifted_inserter]") {
    mat::compressed2D<double> A(4, 3);
    {
        mat::shifted_inserter<mat::inserter<mat::compressed2D<double>>> ins(A, 3, 2, 0);
        // Inserting at logical row 0 → actual row 2
        ins[0][0] << 5.0;
        ins[0][1] << 6.0;
        ins[1][2] << 7.0;
    }
    REQUIRE(A(2, 0) == 5.0);
    REQUIRE(A(2, 1) == 6.0);
    REQUIRE(A(3, 2) == 7.0);
}

TEST_CASE("shifted_inserter with col offset", "[mat][shifted_inserter]") {
    mat::compressed2D<double> A(3, 4);
    {
        mat::shifted_inserter<mat::inserter<mat::compressed2D<double>>> ins(A, 3, 0, 1);
        ins[0][0] << 10.0;  // actual col = 1
        ins[1][1] << 20.0;  // actual col = 2
        ins[2][2] << 30.0;  // actual col = 3
    }
    REQUIRE(A(0, 1) == 10.0);
    REQUIRE(A(1, 2) == 20.0);
    REQUIRE(A(2, 3) == 30.0);
    REQUIRE(A(0, 0) == 0.0);
}

TEST_CASE("shifted_inserter with both offsets", "[mat][shifted_inserter]") {
    mat::compressed2D<double> A(4, 4);
    {
        mat::shifted_inserter<mat::inserter<mat::compressed2D<double>>> ins(A, 3, 1, 2);
        ins[0][0] << 42.0;  // actual (1, 2)
        ins[1][1] << 43.0;  // actual (2, 3)
    }
    REQUIRE(A(1, 2) == 42.0);
    REQUIRE(A(2, 3) == 43.0);
}

TEST_CASE("shifted_inserter change offsets mid-insertion", "[mat][shifted_inserter]") {
    mat::compressed2D<double> A(4, 4);
    {
        mat::shifted_inserter<mat::inserter<mat::compressed2D<double>>> ins(A, 3, 0, 0);
        ins[0][0] << 1.0;   // (0, 0)
        ins.set_row_offset(2);
        ins.set_col_offset(2);
        ins[0][0] << 2.0;   // (2, 2)
        REQUIRE(ins.get_row_offset() == 2);
        REQUIRE(ins.get_col_offset() == 2);
    }
    REQUIRE(A(0, 0) == 1.0);
    REQUIRE(A(2, 2) == 2.0);
}

TEST_CASE("shifted_inserter FEM-style subblock assembly", "[mat][shifted_inserter]") {
    // Simulate assembling 2x2 element stiffness matrices into a 4x4 global matrix
    // Element 0 maps to dofs {0,1}, element 1 maps to dofs {2,3}
    mat::compressed2D<double> K(4, 4);
    {
        mat::shifted_inserter<mat::inserter<mat::compressed2D<double>, mat::update_plus<double>>>
            ins(K, 4, 0, 0);

        // Element 0: 2x2 block at offset (0,0)
        ins[0][0] << 4.0;
        ins[0][1] << -1.0;
        ins[1][0] << -1.0;
        ins[1][1] << 4.0;

        // Element 1: 2x2 block at offset (2,2)
        ins.set_row_offset(2);
        ins.set_col_offset(2);
        ins[0][0] << 4.0;
        ins[0][1] << -1.0;
        ins[1][0] << -1.0;
        ins[1][1] << 4.0;
    }
    REQUIRE(K(0, 0) == 4.0);
    REQUIRE(K(0, 1) == -1.0);
    REQUIRE(K(1, 0) == -1.0);
    REQUIRE(K(1, 1) == 4.0);
    REQUIRE(K(2, 2) == 4.0);
    REQUIRE(K(2, 3) == -1.0);
    REQUIRE(K(3, 2) == -1.0);
    REQUIRE(K(3, 3) == 4.0);
}
