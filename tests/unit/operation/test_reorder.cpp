#include <catch2/catch_test_macros.hpp>
#include <mtl/operation/reorder.hpp>
#include <numeric>

using namespace mtl;

TEST_CASE("reorder_rows with identity permutation", "[operation][reorder]") {
    mat::dense2D<double> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::size_t> perm = {0, 1, 2};
    auto B = mat::reorder_rows(A, perm);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(B(i, j) == A(i, j));
}

TEST_CASE("reorder_rows with reversal permutation", "[operation][reorder]") {
    mat::dense2D<double> A = {
        {1, 2},
        {3, 4},
        {5, 6}
    };
    std::vector<std::size_t> perm = {2, 1, 0};
    auto B = mat::reorder_rows(A, perm);
    // Row 0 of B = Row 2 of A
    REQUIRE(B(0, 0) == 5.0);
    REQUIRE(B(0, 1) == 6.0);
    // Row 1 unchanged
    REQUIRE(B(1, 0) == 3.0);
    REQUIRE(B(1, 1) == 4.0);
    // Row 2 of B = Row 0 of A
    REQUIRE(B(2, 0) == 1.0);
    REQUIRE(B(2, 1) == 2.0);
}

TEST_CASE("reorder_cols with specific permutation", "[operation][reorder]") {
    mat::dense2D<double> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    // Swap columns 0 and 2
    std::vector<std::size_t> perm = {2, 1, 0};
    auto B = mat::reorder_cols(A, perm);
    REQUIRE(B(0, 0) == 3.0);
    REQUIRE(B(0, 1) == 2.0);
    REQUIRE(B(0, 2) == 1.0);
    REQUIRE(B(1, 0) == 6.0);
    REQUIRE(B(1, 1) == 5.0);
    REQUIRE(B(1, 2) == 4.0);
}

TEST_CASE("symmetric reorder preserves trace", "[operation][reorder]") {
    // PAP^T preserves trace (sum of eigenvalues)
    mat::dense2D<double> A = {
        {4, 1, 2},
        {1, 3, 0},
        {2, 0, 5}
    };
    double trace_A = A(0, 0) + A(1, 1) + A(2, 2);

    std::vector<std::size_t> perm = {2, 0, 1};
    auto B = mat::reorder(A, perm);
    double trace_B = B(0, 0) + B(1, 1) + B(2, 2);

    REQUIRE(trace_A == trace_B);
}

TEST_CASE("symmetric reorder with identity perm", "[operation][reorder]") {
    mat::dense2D<double> A = {
        {1, 2},
        {3, 4}
    };
    std::vector<std::size_t> perm = {0, 1};
    auto B = mat::reorder(A, perm);
    REQUIRE(B(0, 0) == 1.0);
    REQUIRE(B(0, 1) == 2.0);
    REQUIRE(B(1, 0) == 3.0);
    REQUIRE(B(1, 1) == 4.0);
}

TEST_CASE("symmetric reorder correctness", "[operation][reorder]") {
    // B(i,j) = A(perm[i], perm[j])
    mat::dense2D<double> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::size_t> perm = {1, 2, 0};
    auto B = mat::reorder(A, perm);
    // B(0,0) = A(1,1) = 5
    REQUIRE(B(0, 0) == 5.0);
    // B(0,1) = A(1,2) = 6
    REQUIRE(B(0, 1) == 6.0);
    // B(2,0) = A(0,1) = 2
    REQUIRE(B(2, 0) == 2.0);
}

TEST_CASE("P * A operator matches reorder_rows", "[operation][reorder]") {
    mat::dense2D<double> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::size_t> perm = {2, 0, 1};
    mat::permutation_matrix<double> P(perm);

    auto B1 = P * A;
    auto B2 = mat::reorder_rows(A, perm);

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(B1(i, j) == B2(i, j));
}
