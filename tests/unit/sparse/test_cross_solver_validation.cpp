#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/rcm.hpp>

#ifdef MTL5_HAS_UMFPACK
#include <mtl/interface/umfpack.hpp>
#endif

using namespace mtl;
using namespace mtl::sparse;

static double relative_residual(
    const mat::compressed2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t m = A.num_rows();
    double res = 0.0, bnorm = 0.0;
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    for (std::size_t i = 0; i < m; ++i) {
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

static double solution_difference(
    const vec::dense_vector<double>& x1,
    const vec::dense_vector<double>& x2)
{
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < x1.size(); ++i) {
        double d = x1(i) - x2(i);
        diff += d * d;
        norm += x1(i) * x1(i);
    }
    if (norm == 0.0) return std::sqrt(diff);
    return std::sqrt(diff / norm);
}

// Build SPD tridiagonal: A(i,i)=4, A(i,i+1)=A(i+1,i)=-1
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

// Build SPD arrow matrix
static mat::compressed2D<double> make_spd_arrow(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << static_cast<double>(3 * n);
        for (std::size_t i = 1; i < n; ++i) {
            ins[0][i] << 1.0;
            ins[i][0] << 1.0;
            ins[i][i] << 5.0;
        }
    }
    return A;
}

// Build unsymmetric tridiagonal with perturbation
static mat::compressed2D<double> make_unsym_tridiag(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -0.5;  // unsymmetric
            }
        }
        if (n > 2) {
            ins[0][n - 1] << 0.3;
            ins[n - 1][0] << -0.2;
        }
    }
    return A;
}

static vec::dense_vector<double> make_rhs(std::size_t n) {
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i)
        b(i) = static_cast<double>((i % 7) - 3);  // deterministic pattern
    return b;
}

// ---- Cross-solver consistency tests ----

TEST_CASE("Cholesky vs LU on SPD system", "[sparse][cross][consistency]") {
    for (std::size_t n : {3, 5, 10, 20, 50}) {
        auto A = make_spd_tridiag(n);
        auto b = make_rhs(n);

        vec::dense_vector<double> x_chol(n, 0.0);
        factorization::sparse_cholesky_solve(A, x_chol, b);

        vec::dense_vector<double> x_lu(n, 0.0);
        factorization::sparse_lu_solve(A, x_lu, b);

        REQUIRE(solution_difference(x_chol, x_lu) < 1e-10);
        REQUIRE(relative_residual(A, x_chol, b) < 1e-12);
        REQUIRE(relative_residual(A, x_lu, b) < 1e-12);
    }
}

TEST_CASE("Cholesky vs QR on SPD system", "[sparse][cross][consistency]") {
    for (std::size_t n : {3, 5, 10, 20}) {
        auto A = make_spd_tridiag(n);
        auto b = make_rhs(n);

        vec::dense_vector<double> x_chol(n, 0.0);
        factorization::sparse_cholesky_solve(A, x_chol, b);

        vec::dense_vector<double> x_qr(n, 0.0);
        factorization::sparse_qr_solve(A, x_qr, b);

        REQUIRE(solution_difference(x_chol, x_qr) < 1e-10);
    }
}

TEST_CASE("LU vs QR on unsymmetric system", "[sparse][cross][consistency]") {
    for (std::size_t n : {3, 5, 10, 20}) {
        auto A = make_unsym_tridiag(n);
        auto b = make_rhs(n);

        vec::dense_vector<double> x_lu(n, 0.0);
        factorization::sparse_lu_solve(A, x_lu, b);

        vec::dense_vector<double> x_qr(n, 0.0);
        factorization::sparse_qr_solve(A, x_qr, b);

        REQUIRE(solution_difference(x_lu, x_qr) < 1e-10);
        REQUIRE(relative_residual(A, x_lu, b) < 1e-12);
    }
}

TEST_CASE("All three solvers agree on arrow SPD matrix", "[sparse][cross][consistency]") {
    for (std::size_t n : {4, 8, 15, 30}) {
        auto A = make_spd_arrow(n);
        auto b = make_rhs(n);

        vec::dense_vector<double> x_chol(n, 0.0), x_lu(n, 0.0), x_qr(n, 0.0);
        factorization::sparse_cholesky_solve(A, x_chol, b, ordering::amd{});
        factorization::sparse_lu_solve(A, x_lu, b);
        factorization::sparse_qr_solve(A, x_qr, b);

        REQUIRE(solution_difference(x_chol, x_lu) < 1e-10);
        REQUIRE(solution_difference(x_chol, x_qr) < 1e-10);
    }
}

TEST_CASE("Ordering does not change solution", "[sparse][cross][ordering]") {
    auto A = make_spd_tridiag(15);
    auto b = make_rhs(15);

    vec::dense_vector<double> x_nat(15, 0.0), x_rcm(15, 0.0), x_amd(15, 0.0);
    factorization::sparse_cholesky_solve(A, x_nat, b);
    factorization::sparse_cholesky_solve(A, x_rcm, b, ordering::rcm{});
    factorization::sparse_cholesky_solve(A, x_amd, b, ordering::amd{});

    REQUIRE(solution_difference(x_nat, x_rcm) < 1e-12);
    REQUIRE(solution_difference(x_nat, x_amd) < 1e-12);
}

#ifdef MTL5_HAS_UMFPACK
TEST_CASE("UMFPACK vs native LU", "[sparse][cross][umfpack]") {
    for (std::size_t n : {3, 5, 10, 20}) {
        auto A = make_unsym_tridiag(n);
        auto b = make_rhs(n);

        vec::dense_vector<double> x_umf(n, 0.0);
        interface::umfpack_solve(A, x_umf, b);

        vec::dense_vector<double> x_lu(n, 0.0);
        factorization::sparse_lu_solve(A, x_lu, b);

        REQUIRE(solution_difference(x_umf, x_lu) < 1e-10);
    }
}

TEST_CASE("UMFPACK vs native Cholesky on SPD", "[sparse][cross][umfpack]") {
    for (std::size_t n : {5, 10, 20}) {
        auto A = make_spd_tridiag(n);
        auto b = make_rhs(n);

        vec::dense_vector<double> x_umf(n, 0.0);
        interface::umfpack_solve(A, x_umf, b);

        vec::dense_vector<double> x_chol(n, 0.0);
        factorization::sparse_cholesky_solve(A, x_chol, b);

        REQUIRE(solution_difference(x_umf, x_chol) < 1e-10);
    }
}
#endif

// ---- Medium and large scale tests ----

TEST_CASE("Cholesky on 50x50 tridiagonal", "[sparse][scale]") {
    auto A = make_spd_tridiag(50);
    auto b = make_rhs(50);
    vec::dense_vector<double> x(50, 0.0);
    factorization::sparse_cholesky_solve(A, x, b, ordering::amd{});
    REQUIRE(relative_residual(A, x, b) < 1e-12);
}

TEST_CASE("Cholesky on 100x100 tridiagonal", "[sparse][scale]") {
    auto A = make_spd_tridiag(100);
    auto b = make_rhs(100);
    vec::dense_vector<double> x(100, 0.0);
    factorization::sparse_cholesky_solve(A, x, b, ordering::amd{});
    REQUIRE(relative_residual(A, x, b) < 1e-12);
}

TEST_CASE("LU on 50x50 unsymmetric", "[sparse][scale]") {
    auto A = make_unsym_tridiag(50);
    auto b = make_rhs(50);
    vec::dense_vector<double> x(50, 0.0);
    factorization::sparse_lu_solve(A, x, b);
    REQUIRE(relative_residual(A, x, b) < 1e-12);
}

TEST_CASE("QR on 50x50 system", "[sparse][scale]") {
    auto A = make_spd_tridiag(50);
    auto b = make_rhs(50);
    vec::dense_vector<double> x(50, 0.0);
    factorization::sparse_qr_solve(A, x, b);
    REQUIRE(relative_residual(A, x, b) < 1e-11);
}

TEST_CASE("Cholesky on 30x30 arrow matrix", "[sparse][scale]") {
    auto A = make_spd_arrow(30);
    auto b = make_rhs(30);
    vec::dense_vector<double> x(30, 0.0);
    factorization::sparse_cholesky_solve(A, x, b, ordering::amd{});
    REQUIRE(relative_residual(A, x, b) < 1e-12);
}

// ---- Edge cases ----

TEST_CASE("Cholesky on block diagonal SPD", "[sparse][edge]") {
    // Two disconnected 3x3 SPD blocks
    std::size_t n = 6;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        // Block 1: rows/cols 0-2
        ins[0][0] << 4.0; ins[0][1] << -1.0;
        ins[1][0] << -1.0; ins[1][1] << 4.0; ins[1][2] << -1.0;
        ins[2][1] << -1.0; ins[2][2] << 4.0;
        // Block 2: rows/cols 3-5
        ins[3][3] << 5.0; ins[3][4] << -2.0;
        ins[4][3] << -2.0; ins[4][4] << 5.0; ins[4][5] << -2.0;
        ins[5][4] << -2.0; ins[5][5] << 5.0;
    }

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    vec::dense_vector<double> x(n, 0.0);
    factorization::sparse_cholesky_solve(A, x, b);
    REQUIRE(relative_residual(A, x, b) < 1e-12);
}

TEST_CASE("LU on diagonal matrix", "[sparse][edge]") {
    std::size_t n = 5;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i)
            ins[i][i] << static_cast<double>(i + 1);
    }
    vec::dense_vector<double> b = {1.0, 4.0, 9.0, 16.0, 25.0};
    vec::dense_vector<double> x(n, 0.0);
    factorization::sparse_lu_solve(A, x, b);
    // x should be [1, 2, 3, 4, 5]
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(x(i), Catch::Matchers::WithinAbs(static_cast<double>(i + 1), 1e-12));
}

TEST_CASE("LU on upper triangular matrix", "[sparse][edge]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 3.0; ins[0][2] << 1.0;
        ins[1][1] << 4.0; ins[1][2] << 2.0;
        ins[2][2] << 5.0;
    }
    vec::dense_vector<double> b = {11.0, 14.0, 10.0};
    vec::dense_vector<double> x(3, 0.0);
    factorization::sparse_lu_solve(A, x, b);
    REQUIRE(relative_residual(A, x, b) < 1e-12);
}

TEST_CASE("LU on permutation matrix", "[sparse][edge]") {
    // P = [[0,1,0],[0,0,1],[1,0,0]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][1] << 1.0;
        ins[1][2] << 1.0;
        ins[2][0] << 1.0;
    }
    vec::dense_vector<double> b = {10.0, 20.0, 30.0};
    vec::dense_vector<double> x(3, 0.0);
    factorization::sparse_lu_solve(A, x, b);
    // P*x = b => x = P^T * b = [30, 10, 20]
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(30.0, 1e-12));
    REQUIRE_THAT(x(1), Catch::Matchers::WithinAbs(10.0, 1e-12));
    REQUIRE_THAT(x(2), Catch::Matchers::WithinAbs(20.0, 1e-12));
}

TEST_CASE("QR least-squares on tall skinny system", "[sparse][edge]") {
    // 10x3 overdetermined system
    std::size_t m = 10, n = 3;
    mat::compressed2D<double> A(m, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < m; ++i) {
            ins[i][0] << 1.0;                               // constant
            ins[i][1] << static_cast<double>(i);             // linear
            ins[i][2] << static_cast<double>(i * i);         // quadratic
        }
    }
    // b = 1 + 2*i + 3*i^2 (exact fit)
    vec::dense_vector<double> b(m);
    for (std::size_t i = 0; i < m; ++i)
        b(i) = 1.0 + 2.0 * i + 3.0 * static_cast<double>(i * i);

    vec::dense_vector<double> x(n, 0.0);
    factorization::sparse_qr_solve(A, x, b);

    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(x(1), Catch::Matchers::WithinAbs(2.0, 1e-8));
    REQUIRE_THAT(x(2), Catch::Matchers::WithinAbs(3.0, 1e-8));
}

TEST_CASE("Identity matrix solve (all solvers)", "[sparse][edge]") {
    std::size_t n = 5;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i)
            ins[i][i] << 1.0;
    }
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};

    vec::dense_vector<double> x_chol(n, 0.0), x_lu(n, 0.0), x_qr(n, 0.0);
    factorization::sparse_cholesky_solve(A, x_chol, b);
    factorization::sparse_lu_solve(A, x_lu, b);
    factorization::sparse_qr_solve(A, x_qr, b);

    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(x_chol(i), Catch::Matchers::WithinAbs(b(i), 1e-12));
        REQUIRE_THAT(x_lu(i), Catch::Matchers::WithinAbs(b(i), 1e-12));
        REQUIRE_THAT(x_qr(i), Catch::Matchers::WithinAbs(b(i), 1e-12));
    }
}

TEST_CASE("Cholesky symbolic reuse across multiple solves", "[sparse][edge]") {
    auto A = make_spd_tridiag(10);
    auto sym = factorization::sparse_cholesky_symbolic(A, ordering::amd{});
    auto num = factorization::sparse_cholesky_numeric(A, sym);

    // Solve 5 different RHS with the same factorization
    for (int trial = 0; trial < 5; ++trial) {
        vec::dense_vector<double> b(10);
        for (std::size_t i = 0; i < 10; ++i)
            b(i) = static_cast<double>((i + trial) % 7);

        vec::dense_vector<double> x(10, 0.0);
        num.solve(x, b);
        REQUIRE(relative_residual(A, x, b) < 1e-12);
    }
}
