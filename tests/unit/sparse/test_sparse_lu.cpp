#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/ordering/rcm.hpp>

using namespace mtl;
using namespace mtl::sparse;

// Helper: compute relative residual ||Ax - b|| / ||b||
static double relative_residual(
    const mat::compressed2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t n = b.size();
    double res_norm = 0.0;
    double b_norm = 0.0;
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x(indices[k]);
        double ri = Ax_i - b(i);
        res_norm += ri * ri;
        b_norm += b(i) * b(i);
    }
    if (b_norm == 0.0) return std::sqrt(res_norm);
    return std::sqrt(res_norm) / std::sqrt(b_norm);
}

TEST_CASE("Sparse LU symbolic analysis", "[sparse][lu]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    auto sym = factorization::sparse_lu_symbolic(A);

    REQUIRE(sym.n == 3);
    REQUIRE(sym.col_perm.size() == 3);
    REQUIRE(sym.col_pinv.size() == 3);
}

TEST_CASE("Sparse LU solve: 3x3 symmetric system", "[sparse][lu]") {
    // A = [[4 -1  0]
    //      [-1  4 -1]
    //      [0  -1  4]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0;  ins[0][1] << -1.0;
        ins[1][0] << -1.0; ins[1][1] << 4.0;  ins[1][2] << -1.0;
        ins[2][1] << -1.0; ins[2][2] << 4.0;
    }

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: 3x3 unsymmetric system", "[sparse][lu]") {
    // Unsymmetric matrix:
    // A = [[3  1  0]
    //      [0  4  2]
    //      [1  0  5]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 3.0; ins[0][1] << 1.0;
        ins[1][1] << 4.0; ins[1][2] << 2.0;
        ins[2][0] << 1.0; ins[2][2] << 5.0;
    }

    vec::dense_vector<double> b = {4.0, 10.0, 11.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: requires pivoting", "[sparse][lu]") {
    // Matrix where natural ordering needs pivoting:
    // A = [[0  1]
    //      [1  1]]
    // Without pivoting, first pivot is 0 -> singular.
    // With pivoting, swap rows -> works.
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 1.0;
    }

    vec::dense_vector<double> b = {3.0, 4.0};
    vec::dense_vector<double> x(2, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: 5x5 tridiagonal", "[sparse][lu]") {
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

    vec::dense_vector<double> b = {1.0, 0.0, -1.0, 2.0, 0.5};
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve with RCM ordering", "[sparse][lu]") {
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

    factorization::sparse_lu_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: dense unsymmetric via sparse", "[sparse][lu]") {
    // Fully dense 3x3 unsymmetric:
    // A = [[2  3  1]
    //      [4  7  5]
    //      [6  18 22]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0;  ins[0][1] << 3.0;  ins[0][2] << 1.0;
        ins[1][0] << 4.0;  ins[1][1] << 7.0;  ins[1][2] << 5.0;
        ins[2][0] << 6.0;  ins[2][1] << 18.0; ins[2][2] << 22.0;
    }

    vec::dense_vector<double> b = {6.0, 16.0, 46.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: 1x1 matrix", "[sparse][lu]") {
    mat::compressed2D<double> A(1, 1);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 5.0;
    }

    vec::dense_vector<double> b = {15.0};
    vec::dense_vector<double> x(1, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("Sparse LU throws for singular matrix", "[sparse][lu]") {
    // Singular: [[1 1], [1 1]]
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 1.0;
    }

    auto sym = factorization::sparse_lu_symbolic(A);
    REQUIRE_THROWS_AS(
        factorization::sparse_lu_numeric(A, sym),
        std::runtime_error);
}

TEST_CASE("Sparse LU with threshold pivoting", "[sparse][lu]") {
    // Matrix where threshold pivoting matters:
    // A = [[1e-10  1]
    //      [1      1]]
    // With full pivoting (threshold=1): swaps rows
    // With threshold=0 (no pivoting): doesn't swap
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1e-10; ins[0][1] << 1.0;
        ins[1][0] << 1.0;   ins[1][1] << 1.0;
    }

    vec::dense_vector<double> b = {1.0, 2.0};

    // Full pivoting (threshold = 1.0)
    vec::dense_vector<double> x1(2, 0.0);
    factorization::sparse_lu_solve(A, x1, b, ordering::rcm{}, 1.0);
    REQUIRE(relative_residual(A, x1, b) < 1e-10);

    // Relaxed threshold
    vec::dense_vector<double> x2(2, 0.0);
    factorization::sparse_lu_solve(A, x2, b, ordering::rcm{}, 0.5);
    REQUIRE(relative_residual(A, x2, b) < 1e-10);
}

TEST_CASE("Sparse LU solve: larger 10x10 system", "[sparse][lu]") {
    std::size_t n = 10;
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
        // Add some unsymmetric entries
        ins[0][n - 1] << 0.5;
        ins[n - 1][0] << -0.5;
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_lu_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU symbolic reuse for same pattern", "[sparse][lu]") {
    mat::compressed2D<double> A1(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A1);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    mat::compressed2D<double> A2(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A2);
        ins[0][0] << 5.0; ins[0][1] << 2.0;
        ins[1][0] << 2.0; ins[1][1] << 8.0; ins[1][2] << 3.0;
        ins[2][1] << 3.0; ins[2][2] << 6.0;
    }

    auto sym = factorization::sparse_lu_symbolic(A1);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};

    auto num1 = factorization::sparse_lu_numeric(A1, sym);
    vec::dense_vector<double> x1(3, 0.0);
    num1.solve(x1, b);
    REQUIRE(relative_residual(A1, x1, b) < 1e-12);

    auto num2 = factorization::sparse_lu_numeric(A2, sym);
    vec::dense_vector<double> x2(3, 0.0);
    num2.solve(x2, b);
    REQUIRE(relative_residual(A2, x2, b) < 1e-12);
}
