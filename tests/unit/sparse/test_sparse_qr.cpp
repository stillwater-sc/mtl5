#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/ordering/rcm.hpp>

using namespace mtl;
using namespace mtl::sparse;

// Helper: compute relative residual ||Ax - b|| / ||b||
static double relative_residual(
    const mat::compressed2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t m = A.num_rows();
    std::size_t n = A.num_cols();
    double res_norm = 0.0;
    double b_norm = 0.0;
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    for (std::size_t i = 0; i < m; ++i) {
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

TEST_CASE("Sparse QR symbolic analysis", "[sparse][qr]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    auto sym = factorization::sparse_qr_symbolic(A);

    REQUIRE(sym.nrows == 3);
    REQUIRE(sym.ncols == 3);
    REQUIRE(sym.col_perm.size() == 3);
}

TEST_CASE("Sparse QR solve: 3x3 square system", "[sparse][qr]") {
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

    factorization::sparse_qr_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse QR solve: 3x3 unsymmetric", "[sparse][qr]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 3.0; ins[0][1] << 1.0;
        ins[1][1] << 4.0; ins[1][2] << 2.0;
        ins[2][0] << 1.0; ins[2][2] << 5.0;
    }

    vec::dense_vector<double> b = {4.0, 10.0, 11.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_qr_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse QR solve: 1x1 matrix", "[sparse][qr]") {
    mat::compressed2D<double> A(1, 1);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 5.0;
    }

    vec::dense_vector<double> b = {15.0};
    vec::dense_vector<double> x(1, 0.0);

    factorization::sparse_qr_solve(A, x, b);

    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("Sparse QR solve: 5x5 tridiagonal", "[sparse][qr]") {
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

    factorization::sparse_qr_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse QR solve with RCM ordering", "[sparse][qr]") {
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

    factorization::sparse_qr_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse QR solve: overdetermined (least squares) 4x3", "[sparse][qr]") {
    // A = [[1  0  0]
    //      [0  1  0]
    //      [0  0  1]
    //      [1  1  1]]
    // b = [1, 2, 3, 6]
    // Exact solution: x = [1, 2, 3] since A*x = b is consistent
    mat::compressed2D<double> A(4, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 1.0;
        ins[2][2] << 1.0;
        ins[3][0] << 1.0; ins[3][1] << 1.0; ins[3][2] << 1.0;
    }

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 6.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_qr_solve(A, x, b);

    // Consistent system, exact solution
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(x(1), Catch::Matchers::WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(x(2), Catch::Matchers::WithinAbs(3.0, 1e-10));
}

TEST_CASE("Sparse QR solve: overdetermined (true least squares) 4x2", "[sparse][qr]") {
    // A = [[1  1]
    //      [1 -1]
    //      [1  0]
    //      [0  1]]
    // b = [2, 0, 1.5, 0.5]
    // Least-squares solution: minimize ||Ax - b||
    mat::compressed2D<double> A(4, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << -1.0;
        ins[2][0] << 1.0;
        ins[3][1] << 1.0;
    }

    vec::dense_vector<double> b = {2.0, 0.0, 1.5, 0.5};
    vec::dense_vector<double> x(2, 0.0);

    factorization::sparse_qr_solve(A, x, b);

    // Normal equations: A^T A x = A^T b
    // A^T A = [[3, 0], [0, 3]]
    // A^T b = [3.5, 2.5]
    // x = [3.5/3, 2.5/3] = [7/6, 5/6]
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(7.0 / 6.0, 1e-10));
    REQUIRE_THAT(x(1), Catch::Matchers::WithinAbs(5.0 / 6.0, 1e-10));
}

TEST_CASE("Sparse QR solve: dense 3x3 via sparse", "[sparse][qr]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0;  ins[0][1] << 3.0;  ins[0][2] << 1.0;
        ins[1][0] << 4.0;  ins[1][1] << 7.0;  ins[1][2] << 5.0;
        ins[2][0] << 6.0;  ins[2][1] << 18.0; ins[2][2] << 22.0;
    }

    vec::dense_vector<double> b = {6.0, 16.0, 46.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_qr_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-10);
}

TEST_CASE("Sparse QR rejects underdetermined system", "[sparse][qr]") {
    // 2 x 3 matrix (m < n)
    mat::compressed2D<double> A(2, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 2.0; ins[0][2] << 3.0;
        ins[1][0] << 4.0; ins[1][1] << 5.0; ins[1][2] << 6.0;
    }

    REQUIRE_THROWS_AS(
        factorization::sparse_qr_symbolic(A),
        std::invalid_argument);
}

TEST_CASE("Sparse QR solve: 10x10 system", "[sparse][qr]") {
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
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_qr_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}
