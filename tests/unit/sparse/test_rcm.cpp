#include <catch2/catch_test_macros.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/ordering/rcm.hpp>
#include <mtl/sparse/ordering/ordering_concepts.hpp>
#include <mtl/sparse/util/permutation.hpp>

#include <set>

using namespace mtl;
using namespace mtl::sparse;

TEST_CASE("RCM ordering produces valid permutation", "[sparse][rcm]") {
    // Tridiagonal 5x5
    mat::compressed2D<double> A(5, 5);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < 5; ++i) {
            ins[i][i] << 2.0;
            if (i + 1 < 5) {
                ins[i][i + 1] << 1.0;
                ins[i + 1][i] << 1.0;
            }
        }
    }

    ordering::rcm order;
    auto perm = order(A);

    REQUIRE(perm.size() == 5);
    REQUIRE(util::is_valid_permutation(perm));
}

TEST_CASE("RCM ordering reduces bandwidth", "[sparse][rcm]") {
    // Create a matrix with poor bandwidth:
    // Connect node 0 to node 4 (bandwidth = 4)
    // RCM should reduce this
    mat::compressed2D<double> A(5, 5);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        // Tridiagonal + (0,4) connection
        for (std::size_t i = 0; i < 5; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < 5) {
                ins[i][i + 1] << 1.0;
                ins[i + 1][i] << 1.0;
            }
        }
        ins[0][4] << 1.0;
        ins[4][0] << 1.0;
    }

    // Compute bandwidth of original
    auto bandwidth = [](const mat::compressed2D<double>& M) -> std::size_t {
        std::size_t bw = 0;
        const auto& starts = M.ref_major();
        const auto& indices = M.ref_minor();
        for (std::size_t r = 0; r < M.num_rows(); ++r) {
            for (std::size_t k = starts[r]; k < starts[r + 1]; ++k) {
                std::size_t diff = (r > indices[k]) ? (r - indices[k]) : (indices[k] - r);
                if (diff > bw) bw = diff;
            }
        }
        return bw;
    };

    std::size_t orig_bw = bandwidth(A);
    REQUIRE(orig_bw == 4);

    ordering::rcm order;
    auto perm = order(A);
    auto B = util::symmetric_permute(A, perm);

    std::size_t new_bw = bandwidth(B);
    // RCM should not increase bandwidth
    REQUIRE(new_bw <= orig_bw);
}

TEST_CASE("RCM ordering handles diagonal matrix", "[sparse][rcm]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 2.0;
        ins[2][2] << 3.0;
    }

    ordering::rcm order;
    auto perm = order(A);

    REQUIRE(perm.size() == 3);
    REQUIRE(util::is_valid_permutation(perm));
}

TEST_CASE("RCM satisfies FillReducingOrdering concept", "[sparse][rcm][concept]") {
    constexpr bool is_ordering = sparse::FillReducingOrdering<ordering::rcm, mat::compressed2D<double>>;
    STATIC_REQUIRE(is_ordering);
}
