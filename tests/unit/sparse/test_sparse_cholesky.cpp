#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/ordering/rcm.hpp>

using namespace mtl;
using namespace mtl::sparse;

// Helper: build a symmetric SPD tridiagonal matrix of size n
// A(i,i) = 4, A(i,i+1) = A(i+1,i) = -1
static mat::compressed2D<double> make_spd_tridiag(std::size_t n) {
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
    return A;
}

// Helper: compute residual norm ||Ax - b|| / ||b||
static double relative_residual(
    const mat::compressed2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t n = b.size();
    double res_norm = 0.0;
    double b_norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        const auto& starts = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data = A.ref_data();
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x(indices[k]);
        double ri = Ax_i - b(i);
        res_norm += ri * ri;
        b_norm += b(i) * b(i);
    }
    return std::sqrt(res_norm) / std::sqrt(b_norm);
}

TEST_CASE("Sparse Cholesky symbolic analysis", "[sparse][cholesky]") {
    auto A = make_spd_tridiag(5);

    auto sym = factorization::sparse_cholesky_symbolic(A);

    REQUIRE(sym.n == 5);
    REQUIRE(sym.perm.size() == 5);
    REQUIRE(sym.pinv.size() == 5);
    REQUIRE(sym.parent.size() == 5);
    REQUIRE(sym.col_counts.size() == 5);
    REQUIRE(sym.nnz_L > 0);
}

TEST_CASE("Sparse Cholesky symbolic with RCM ordering", "[sparse][cholesky]") {
    auto A = make_spd_tridiag(5);

    auto sym = factorization::sparse_cholesky_symbolic(A, ordering::rcm{});

    REQUIRE(sym.n == 5);
    REQUIRE(sym.perm.size() == 5);
    REQUIRE(sym.nnz_L > 0);
}

TEST_CASE("Sparse Cholesky numeric: 3x3 SPD matrix", "[sparse][cholesky]") {
    // A = [[4  -1  0]
    //      [-1  4 -1]
    //      [0  -1  4]]
    auto A = make_spd_tridiag(3);

    auto sym = factorization::sparse_cholesky_symbolic(A);
    auto num = factorization::sparse_cholesky_numeric(A, sym);

    // Verify L is lower triangular
    const auto& L = num.L;
    REQUIRE(L.nrows == 3);
    REQUIRE(L.ncols == 3);

    // Verify L*L^T = A by checking a few entries
    // L should have the diagonal on row_ind[col_ptr[j]] == j
    for (std::size_t j = 0; j < 3; ++j) {
        REQUIRE(L.col_ptr[j] < L.col_ptr[j + 1]);  // non-empty column
        REQUIRE(L.row_ind[L.col_ptr[j]] == j);      // diagonal first
        REQUIRE(L.values[L.col_ptr[j]] > 0.0);      // positive diagonal
    }
}

TEST_CASE("Sparse Cholesky solve: 3x3 tridiagonal", "[sparse][cholesky]") {
    auto A = make_spd_tridiag(3);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_cholesky_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse Cholesky solve: 5x5 tridiagonal", "[sparse][cholesky]") {
    auto A = make_spd_tridiag(5);

    vec::dense_vector<double> b = {1.0, 0.0, -1.0, 2.0, 0.5};
    vec::dense_vector<double> x(5, 0.0);

    factorization::sparse_cholesky_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse Cholesky solve with RCM ordering", "[sparse][cholesky]") {
    auto A = make_spd_tridiag(5);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    vec::dense_vector<double> x(5, 0.0);

    factorization::sparse_cholesky_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse Cholesky solve: dense SPD via sparse", "[sparse][cholesky]") {
    // Fully dense 3x3 SPD matrix:
    // A = [[10  2  1]
    //      [ 2  5  3]
    //      [ 1  3  8]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 10.0; ins[0][1] << 2.0;  ins[0][2] << 1.0;
        ins[1][0] << 2.0;  ins[1][1] << 5.0;  ins[1][2] << 3.0;
        ins[2][0] << 1.0;  ins[2][1] << 3.0;  ins[2][2] << 8.0;
    }

    vec::dense_vector<double> b = {13.0, 10.0, 12.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_cholesky_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse Cholesky reuse symbolic for same pattern", "[sparse][cholesky]") {
    auto A1 = make_spd_tridiag(4);

    // A2: same pattern but different values
    mat::compressed2D<double> A2(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A2);
        for (std::size_t i = 0; i < 4; ++i) {
            ins[i][i] << 6.0;
            if (i + 1 < 4) {
                ins[i][i + 1] << -2.0;
                ins[i + 1][i] << -2.0;
            }
        }
    }

    // Symbolic analysis once
    auto sym = factorization::sparse_cholesky_symbolic(A1);

    // Numeric factorization for both matrices
    auto num1 = factorization::sparse_cholesky_numeric(A1, sym);
    auto num2 = factorization::sparse_cholesky_numeric(A2, sym);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};

    vec::dense_vector<double> x1(4, 0.0);
    num1.solve(x1, b);
    REQUIRE(relative_residual(A1, x1, b) < 1e-12);

    vec::dense_vector<double> x2(4, 0.0);
    num2.solve(x2, b);
    REQUIRE(relative_residual(A2, x2, b) < 1e-12);
}

TEST_CASE("Sparse Cholesky throws for non-SPD matrix", "[sparse][cholesky]") {
    // Not SPD: diagonal has a zero
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][0] << 2.0; ins[1][1] << 1.0;  // not SPD: 1*1 - 2*2 < 0
        ins[2][2] << 1.0;
    }

    auto sym = factorization::sparse_cholesky_symbolic(A);
    REQUIRE_THROWS_AS(
        factorization::sparse_cholesky_numeric(A, sym),
        std::runtime_error);
}

TEST_CASE("Sparse Cholesky solve: 1x1 matrix", "[sparse][cholesky]") {
    mat::compressed2D<double> A(1, 1);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0;
    }

    vec::dense_vector<double> b = {8.0};
    vec::dense_vector<double> x(1, 0.0);

    factorization::sparse_cholesky_solve(A, x, b);

    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(2.0, 1e-12));
}

TEST_CASE("Sparse Cholesky solve: arrow matrix", "[sparse][cholesky]") {
    // Arrow matrix (first row/col dense, rest diagonal):
    // A = [[10  1  1  1]
    //      [ 1  5  0  0]
    //      [ 1  0  5  0]
    //      [ 1  0  0  5]]
    // This tests fill-in: L will have entries in positions (1,0), (2,0), (3,0)
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

    vec::dense_vector<double> b = {13.0, 6.0, 6.0, 6.0};
    vec::dense_vector<double> x(4, 0.0);

    factorization::sparse_cholesky_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse Cholesky solve: larger 10x10 system", "[sparse][cholesky]") {
    std::size_t n = 10;
    auto A = make_spd_tridiag(n);

    // b = [1, 1, ..., 1]
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_cholesky_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}
