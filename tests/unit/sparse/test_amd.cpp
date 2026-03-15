#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/ordering_concepts.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>

using namespace mtl;
using namespace mtl::sparse;

TEST_CASE("AMD produces valid permutation", "[sparse][amd]") {
    mat::compressed2D<double> A(5, 5);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < 5; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < 5) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }

    ordering::amd order;
    auto perm = order(A);

    REQUIRE(perm.size() == 5);
    REQUIRE(util::is_valid_permutation(perm));
}

TEST_CASE("AMD on diagonal matrix", "[sparse][amd]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 2.0;
        ins[2][2] << 3.0;
    }

    ordering::amd order;
    auto perm = order(A);

    REQUIRE(perm.size() == 3);
    REQUIRE(util::is_valid_permutation(perm));
    // All nodes have degree 0, any order is valid
}

TEST_CASE("AMD on empty matrix", "[sparse][amd]") {
    mat::compressed2D<double> A(0, 0);

    ordering::amd order;
    auto perm = order(A);

    REQUIRE(perm.empty());
}

TEST_CASE("AMD on arrow matrix reduces fill", "[sparse][amd]") {
    // Arrow matrix: dense first row/col, rest diagonal
    // Natural ordering causes maximal fill in Cholesky
    // AMD should reorder to reduce fill
    std::size_t n = 6;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 20.0;
        for (std::size_t i = 1; i < n; ++i) {
            ins[0][i] << 1.0;
            ins[i][0] << 1.0;
            ins[i][i] << 5.0;
        }
    }

    // Cholesky with natural ordering
    auto sym_natural = factorization::sparse_cholesky_symbolic(A);
    std::size_t nnz_natural = sym_natural.nnz_L;

    // Cholesky with AMD ordering
    auto sym_amd = factorization::sparse_cholesky_symbolic(A, ordering::amd{});
    std::size_t nnz_amd = sym_amd.nnz_L;

    // AMD should produce less or equal fill than natural ordering
    REQUIRE(nnz_amd <= nnz_natural);
}

TEST_CASE("AMD ordering works with Cholesky solve", "[sparse][amd]") {
    std::size_t n = 5;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_cholesky_solve(A, x, b, ordering::amd{});

    // Verify residual
    double res = 0.0, bnorm = 0.0;
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x(indices[k]);
        double ri = Ax_i - b(i);
        res += ri * ri;
        bnorm += b(i) * b(i);
    }
    REQUIRE(std::sqrt(res / bnorm) < 1e-12);
}

TEST_CASE("AMD satisfies FillReducingOrdering concept", "[sparse][amd][concept]") {
    constexpr bool is_ordering = FillReducingOrdering<ordering::amd, mat::compressed2D<double>>;
    STATIC_REQUIRE(is_ordering);
}

TEST_CASE("AMD prefers low-degree nodes first", "[sparse][amd]") {
    // Star graph: node 0 connects to all others, others only connect to 0
    // Node 0 has degree n-1, others have degree 1
    // AMD should eliminate the degree-1 nodes before the hub
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 10.0;
        for (std::size_t i = 1; i < 4; ++i) {
            ins[0][i] << 1.0;
            ins[i][0] << 1.0;
            ins[i][i] << 5.0;
        }
    }

    ordering::amd order;
    auto perm = order(A);

    REQUIRE(perm.size() == 4);
    REQUIRE(util::is_valid_permutation(perm));

    // The leaf nodes (degree 1) should appear before the hub (degree 3)
    // Find position of hub node 0 in the ordering
    std::size_t hub_pos = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        if (perm[i] == 0) { hub_pos = i; break; }
    }
    // Hub should not be first (leaves have lower degree)
    REQUIRE(hub_pos > 0);
}
