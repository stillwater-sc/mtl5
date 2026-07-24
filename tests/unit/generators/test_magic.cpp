#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <mtl/generators/magic.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

namespace {
    // the magic constant: every row/column/diagonal sums to N(N^2+1)/2
    double magic_constant(std::size_t N) { return double(N) * (double(N) * double(N) + 1.0) / 2.0; }

    template <typename M>
    void require_magic(const M& A, std::size_t N) {
        const double M0 = magic_constant(N);
        // rows and columns
        for (std::size_t i = 0; i < N; ++i) {
            double rsum = 0.0, csum = 0.0;
            for (std::size_t j = 0; j < N; ++j) { rsum += A(i, j); csum += A(j, i); }
            REQUIRE_THAT(rsum, WithinAbs(M0, 1e-9));
            REQUIRE_THAT(csum, WithinAbs(M0, 1e-9));
        }
        // both main diagonals
        double d1 = 0.0, d2 = 0.0;
        for (std::size_t i = 0; i < N; ++i) { d1 += A(i, i); d2 += A(i, N - 1 - i); }
        REQUIRE_THAT(d1, WithinAbs(M0, 1e-9));
        REQUIRE_THAT(d2, WithinAbs(M0, 1e-9));
    }
}

TEST_CASE("magic dimensions", "[generators][magic]") {
    auto A = generators::magic<double>(5);
    REQUIRE(A.num_rows() == 5);
    REQUIRE(A.num_cols() == 5);
}

TEST_CASE("magic order 3 known values", "[generators][magic]") {
    auto A = generators::magic<double>(3);
    // Siamese 3x3:  8 1 6 / 3 5 7 / 4 9 2
    REQUIRE_THAT(A(0, 0), WithinAbs(8.0, 1e-12));
    REQUIRE_THAT(A(0, 1), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(A(0, 2), WithinAbs(6.0, 1e-12));
    REQUIRE_THAT(A(1, 1), WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(A(2, 0), WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(A(2, 1), WithinAbs(9.0, 1e-12));
    require_magic(A, 3);
}

TEST_CASE("magic odd orders are magic", "[generators][magic]") {
    require_magic(generators::magic<double>(5), 5);   // constant 65
    require_magic(generators::magic<double>(7), 7);   // constant 175
    require_magic(generators::magic<double>(9), 9);   // constant 369
}

TEST_CASE("magic doubly-even orders are magic", "[generators][magic]") {
    require_magic(generators::magic<double>(4), 4);   // constant 34
    require_magic(generators::magic<double>(8), 8);   // constant 260
    require_magic(generators::magic<double>(12), 12); // constant 870
}

TEST_CASE("magic contains a permutation of 1..N^2", "[generators][magic]") {
    const std::size_t N = 5;
    auto A = generators::magic<double>(N);
    std::vector<bool> seen(N * N + 1, false);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            auto v = static_cast<std::size_t>(A(i, j) + 0.5);
            REQUIRE(v >= 1);
            REQUIRE(v <= N * N);
            REQUIRE_FALSE(seen[v]);   // no duplicates
            seen[v] = true;
        }
}

TEST_CASE("magic unsupported orders throw", "[generators][magic]") {
    REQUIRE_THROWS_AS(generators::magic<double>(0), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::magic<double>(6), std::invalid_argument);  // singly-even
    REQUIRE_THROWS_AS(generators::magic<double>(10), std::invalid_argument); // singly-even
}
