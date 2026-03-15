#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/sparse/ordering/ordering_concepts.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>

using namespace mtl;
using namespace mtl::sparse;

// Helper: compute relative residual
static double relative_residual(
    const mat::compressed2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t n = b.size();
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
    if (bnorm == 0.0) return std::sqrt(res);
    return std::sqrt(res / bnorm);
}

TEST_CASE("COLAMD produces valid permutation", "[sparse][colamd]") {
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 4.0; ins[2][3] << 1.0;
        ins[3][2] << 1.0; ins[3][3] << 2.0;
    }

    ordering::colamd order;
    auto perm = order(A);

    REQUIRE(perm.size() == 4);
    REQUIRE(util::is_valid_permutation(perm));
}

TEST_CASE("COLAMD on empty matrix", "[sparse][colamd]") {
    mat::compressed2D<double> A(0, 0);

    ordering::colamd order;
    auto perm = order(A);

    REQUIRE(perm.empty());
}

TEST_CASE("COLAMD on rectangular matrix", "[sparse][colamd]") {
    // 4x3 matrix
    mat::compressed2D<double> A(4, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 1.0;
        ins[1][1] << 2.0; ins[1][2] << 1.0;
        ins[2][0] << 1.0; ins[2][2] << 3.0;
        ins[3][0] << 1.0; ins[3][1] << 1.0; ins[3][2] << 1.0;
    }

    ordering::colamd order;
    auto perm = order(A);

    // Permutation should be over columns (size 3)
    REQUIRE(perm.size() == 3);
    REQUIRE(util::is_valid_permutation(perm));
}

TEST_CASE("COLAMD ordering works with LU solve", "[sparse][colamd]") {
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
        // Add unsymmetric entries
        ins[0][n - 1] << 0.5;
        ins[n - 1][0] << -0.5;
    }

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_lu_solve(A, x, b, ordering::colamd{});

    REQUIRE(relative_residual(A, x, b) < 1e-12);
}

TEST_CASE("COLAMD satisfies FillReducingOrdering concept", "[sparse][colamd][concept]") {
    constexpr bool is_ordering = FillReducingOrdering<ordering::colamd, mat::compressed2D<double>>;
    STATIC_REQUIRE(is_ordering);
}

TEST_CASE("COLAMD on diagonal matrix", "[sparse][colamd]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 2.0;
        ins[2][2] << 3.0;
    }

    ordering::colamd order;
    auto perm = order(A);

    REQUIRE(perm.size() == 3);
    REQUIRE(util::is_valid_permutation(perm));
}

TEST_CASE("COLAMD on unsymmetric arrow matrix with LU", "[sparse][colamd]") {
    // Dense first column, sparse rest
    std::size_t n = 6;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][0] << 1.0;     // dense first column
            ins[i][i] << 5.0;     // diagonal
        }
        ins[0][0] << 10.0;  // overwrite diagonal
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_lu_solve(A, x, b, ordering::colamd{});

    REQUIRE(relative_residual(A, x, b) < 1e-12);
}
