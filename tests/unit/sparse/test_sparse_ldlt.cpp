#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
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
    if (b_norm == 0.0)
        return std::sqrt(res_norm);
    return std::sqrt(res_norm) / std::sqrt(b_norm);
}

TEST_CASE("Sparse LDL^T symbolic analysis", "[sparse][ldlt]") {
    auto A = make_spd_tridiag(5);

    auto sym = factorization::sparse_ldlt_symbolic(A);

    REQUIRE(sym.n == 5);
    REQUIRE(sym.perm.size() == 5);
    REQUIRE(sym.pinv.size() == 5);
    REQUIRE(sym.parent.size() == 5);
    REQUIRE(sym.col_counts.size() == 5);
    REQUIRE(sym.nnz_L > 0);
}

TEST_CASE("Sparse LDL^T symbolic with RCM ordering", "[sparse][ldlt]") {
    auto A = make_spd_tridiag(5);

    auto sym = factorization::sparse_ldlt_symbolic(A, ordering::rcm{});

    REQUIRE(sym.n == 5);
    REQUIRE(sym.perm.size() == 5);
    REQUIRE(sym.nnz_L > 0);
}

TEST_CASE("Sparse LDL^T rejects rectangular matrix", "[sparse][ldlt]") {
    mat::compressed2D<double> A(3, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 2.0;
        ins[2][2] << 3.0;
    }

    REQUIRE_THROWS_AS(
        factorization::sparse_ldlt_symbolic(A),
        std::invalid_argument);
}

TEST_CASE("Sparse LDL^T numeric: 3x3 SPD matrix", "[sparse][ldlt]") {
    auto A = make_spd_tridiag(3);

    auto sym = factorization::sparse_ldlt_symbolic(A);
    auto num = factorization::sparse_ldlt_numeric(A, sym);

    // Verify L structure: unit lower triangular with no diagonal stored
    const auto& L = num.L;
    REQUIRE(L.nrows == 3);
    REQUIRE(L.ncols == 3);

    // L stores only strictly-lower entries (no diagonal — unit diagonal is implicit)
    // For a tridiagonal 3x3, each column except the last has 1 off-diagonal entry
    for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p)
            REQUIRE(L.row_ind[p] > j);  // all entries strictly below diagonal
    }

    // Verify D has positive entries (SPD matrix)
    REQUIRE(num.D.size() == 3);
    for (std::size_t j = 0; j < 3; ++j)
        REQUIRE(num.D[j] > 0.0);
}

TEST_CASE("Sparse LDL^T solve: 3x3 tridiagonal", "[sparse][ldlt]") {
    auto A = make_spd_tridiag(3);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_ldlt_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LDL^T solve: 5x5 tridiagonal", "[sparse][ldlt]") {
    auto A = make_spd_tridiag(5);

    vec::dense_vector<double> b = {1.0, 0.0, -1.0, 2.0, 0.5};
    vec::dense_vector<double> x(5, 0.0);

    factorization::sparse_ldlt_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LDL^T solve with RCM ordering", "[sparse][ldlt]") {
    auto A = make_spd_tridiag(5);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    vec::dense_vector<double> x(5, 0.0);

    factorization::sparse_ldlt_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LDL^T solve: dense SPD via sparse", "[sparse][ldlt]") {
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

    factorization::sparse_ldlt_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LDL^T reuse symbolic for same pattern", "[sparse][ldlt]") {
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

    auto sym = factorization::sparse_ldlt_symbolic(A1);

    auto num1 = factorization::sparse_ldlt_numeric(A1, sym);
    auto num2 = factorization::sparse_ldlt_numeric(A2, sym);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};

    vec::dense_vector<double> x1(4, 0.0);
    num1.solve(x1, b);
    REQUIRE(relative_residual(A1, x1, b) < 1e-12);

    vec::dense_vector<double> x2(4, 0.0);
    num2.solve(x2, b);
    REQUIRE(relative_residual(A2, x2, b) < 1e-12);
}

TEST_CASE("Sparse LDL^T throws for zero pivot", "[sparse][ldlt]") {
    // Singular matrix with zero pivot
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 0.0;
        ins[0][1] << 1.0;
        ins[1][0] << 1.0;
        ins[1][1] << 0.0;
    }

    auto sym = factorization::sparse_ldlt_symbolic(A);
    REQUIRE_THROWS_AS(
        factorization::sparse_ldlt_numeric(A, sym),
        std::runtime_error);
}

TEST_CASE("Sparse LDL^T solve: 1x1 matrix", "[sparse][ldlt]") {
    mat::compressed2D<double> A(1, 1);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0;
    }

    vec::dense_vector<double> b = {8.0};
    vec::dense_vector<double> x(1, 0.0);

    factorization::sparse_ldlt_solve(A, x, b);

    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(2.0, 1e-12));
}

TEST_CASE("Sparse LDL^T solve: arrow matrix", "[sparse][ldlt]") {
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

    factorization::sparse_ldlt_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LDL^T solve: larger 10x10 system", "[sparse][ldlt]") {
    std::size_t n = 10;
    auto A = make_spd_tridiag(n);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_ldlt_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LDL^T handles symmetric indefinite matrix", "[sparse][ldlt]") {
    // Symmetric indefinite: A = [[1, 3], [3, 1]]
    // Eigenvalues: 4, -2 (indefinite)
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[0][1] << 3.0;
        ins[1][0] << 3.0;
        ins[1][1] << 1.0;
    }

    vec::dense_vector<double> b = {4.0, 4.0};
    vec::dense_vector<double> x(2, 0.0);

    factorization::sparse_ldlt_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}
