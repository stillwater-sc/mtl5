#include <catch2/catch_test_macros.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/util/permutation.hpp>

using namespace mtl;
using namespace mtl::sparse::util;

TEST_CASE("Identity permutation", "[sparse][permutation]") {
    auto p = identity_permutation(5);
    REQUIRE(p.size() == 5);
    for (std::size_t i = 0; i < 5; ++i)
        REQUIRE(p[i] == i);
    REQUIRE(is_valid_permutation(p));
}

TEST_CASE("Invert permutation", "[sparse][permutation]") {
    // p = [2, 0, 1] means: new[0]=old[2], new[1]=old[0], new[2]=old[1]
    std::vector<std::size_t> p = {2, 0, 1};
    auto pinv = invert_permutation(p);
    // pinv[old] = new: pinv[2]=0, pinv[0]=1, pinv[1]=2
    REQUIRE(pinv[0] == 1);
    REQUIRE(pinv[1] == 2);
    REQUIRE(pinv[2] == 0);

    // Double inversion should give identity
    auto pp = invert_permutation(pinv);
    REQUIRE(pp == p);
}

TEST_CASE("Compose permutations", "[sparse][permutation]") {
    std::vector<std::size_t> a = {2, 0, 1};
    std::vector<std::size_t> b = {1, 2, 0};
    auto c = compose_permutations(a, b);
    // c[i] = a[b[i]]: c[0]=a[1]=0, c[1]=a[2]=1, c[2]=a[0]=2
    REQUIRE(c[0] == 0);
    REQUIRE(c[1] == 1);
    REQUIRE(c[2] == 2);
}

TEST_CASE("Permutation validity", "[sparse][permutation]") {
    REQUIRE(is_valid_permutation({0, 1, 2}));
    REQUIRE(is_valid_permutation({2, 0, 1}));
    REQUIRE(!is_valid_permutation({0, 0, 1}));  // duplicate
    REQUIRE(!is_valid_permutation({0, 1, 3}));  // out of range
}

TEST_CASE("Symmetric permute sparse matrix", "[sparse][permutation]") {
    // A = [[4 1 0]
    //      [1 3 1]
    //      [0 1 2]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    // Permute with p = [2, 0, 1]: new row/col 0 = old 2, etc.
    std::vector<std::size_t> perm = {2, 0, 1};
    auto B = symmetric_permute(A, perm);

    // B(i,j) = A(perm[i], perm[j])
    // B(0,0) = A(2,2) = 2, B(0,1) = A(2,0) = 0, B(0,2) = A(2,1) = 1
    // B(1,0) = A(0,2) = 0, B(1,1) = A(0,0) = 4, B(1,2) = A(0,1) = 1
    // B(2,0) = A(1,2) = 1, B(2,1) = A(1,0) = 1, B(2,2) = A(1,1) = 3
    REQUIRE(B(0, 0) == 2.0);
    REQUIRE(B(0, 1) == 0.0);
    REQUIRE(B(0, 2) == 1.0);
    REQUIRE(B(1, 1) == 4.0);
    REQUIRE(B(1, 2) == 1.0);
    REQUIRE(B(2, 0) == 1.0);
    REQUIRE(B(2, 1) == 1.0);
    REQUIRE(B(2, 2) == 3.0);
}

TEST_CASE("Permute dense vector", "[sparse][permutation]") {
    std::vector<double> x = {10.0, 20.0, 30.0};
    std::vector<std::size_t> perm = {2, 0, 1};

    auto y = permute_vector(x, perm);
    // y[i] = x[perm[i]]: y[0]=x[2]=30, y[1]=x[0]=10, y[2]=x[1]=20
    REQUIRE(y[0] == 30.0);
    REQUIRE(y[1] == 10.0);
    REQUIRE(y[2] == 20.0);

    auto z = ipermute_vector(y, perm);
    // ipermute undoes permute
    REQUIRE(z[0] == 10.0);
    REQUIRE(z[1] == 20.0);
    REQUIRE(z[2] == 30.0);
}
