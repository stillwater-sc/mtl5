#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/generators/laplacian.hpp>

using namespace mtl;
using namespace mtl::sparse;

namespace {

double relative_residual(const mat::compressed2D<double>& A,
                         const vec::dense_vector<double>& x,
                         const vec::dense_vector<double>& b) {
    std::size_t n = b.size();
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    double res_norm = 0.0, b_norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x(indices[k]);
        double ri = Ax_i - b(i);
        res_norm += ri * ri;
        b_norm += b(i) * b(i);
    }
    if (b_norm == 0.0) return std::sqrt(res_norm);
    return std::sqrt(res_norm / b_norm);
}

void report(const char* matrix, std::size_t n, std::size_t nnz,
            double rr, double tol) {
    std::cout << std::left << std::setw(16) << matrix
              << "  n=" << std::setw(7) << n
              << "  nnz=" << std::setw(8) << nnz
              << "  residual=" << std::scientific << std::setprecision(3) << rr
              << "  tol=" << tol
              << (rr < tol ? "  PASS" : "  FAIL")
              << std::defaultfloat << std::endl;
}

} // anonymous namespace

TEST_CASE("Sparse QR regression: 1D Laplacian (square)", "[regression][sparse][qr]") {
    auto n = GENERATE(1000, 5000);

    auto A = generators::laplacian_1d<double>(n);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_qr_solve(A, x, b);
    double rr = relative_residual(A, x, b);
    // 1D Laplacian cond ~ O(n^2)
    double tol = double(n) * double(n) * std::numeric_limits<double>::epsilon();
    report("Laplacian-1D", n, A.nnz(), rr, tol);
    REQUIRE(rr < tol);
}

TEST_CASE("Sparse QR regression: 2D Laplacian (square)", "[regression][sparse][qr]") {
    auto k = GENERATE(32, 50);
    std::size_t n = std::size_t(k) * std::size_t(k);

    auto A = generators::laplacian_2d<double>(k, k);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_qr_solve(A, x, b);
    double rr = relative_residual(A, x, b);
    double tol = double(n) * std::numeric_limits<double>::epsilon();
    report("Laplacian-2D", n, A.nnz(), rr, tol);
    REQUIRE(rr < tol);
}

TEST_CASE("Sparse QR regression: tall-skinny least-squares", "[regression][sparse][qr]") {
    // Overdetermined system: m rows, n cols, m > n
    // Stack two copies of a 1D Laplacian vertically: A is (2n x n)
    auto n = GENERATE(500, 1000);
    std::size_t m = 2 * static_cast<std::size_t>(n);

    mat::compressed2D<double> A(m, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        // Two copies of 1D Laplacian [-1, 2, -1] stacked
        for (std::size_t block = 0; block < 2; ++block) {
            std::size_t row_off = block * static_cast<std::size_t>(n);
            for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
                ins[row_off + i][i] << 2.0;
                if (i > 0)
                    ins[row_off + i][i - 1] << -1.0;
                if (i + 1 < static_cast<std::size_t>(n))
                    ins[row_off + i][i + 1] << -1.0;
            }
        }
    }

    // RHS: b = A * ones (so x_exact = ones is the least-squares solution)
    vec::dense_vector<double> b(m, 0.0);
    {
        const auto& starts = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data = A.ref_data();
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
                b(i) += data[k]; // A * ones
    }

    vec::dense_vector<double> x(n, 0.0);
    factorization::sparse_qr_solve(A, x, b);

    // Check normal equations residual: ||A^T(Ax - b)|| / ||A^T b||
    // First compute r = Ax - b
    std::vector<double> r(m, 0.0);
    {
        const auto& starts = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data = A.ref_data();
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
                r[i] += data[k] * x(indices[k]);
            r[i] -= b(i);
        }
    }

    // Compute A^T r (n-vector) and A^T b (for normalization)
    std::vector<double> Atr(n, 0.0), Atb(n, 0.0);
    {
        const auto& starts = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data = A.ref_data();
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t k = starts[i]; k < starts[i + 1]; ++k) {
                Atr[indices[k]] += data[k] * r[i];
                Atb[indices[k]] += data[k] * b(i);
            }
    }

    double Atr_norm = 0.0, Atb_norm = 0.0;
    for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
        Atr_norm += Atr[j] * Atr[j];
        Atb_norm += Atb[j] * Atb[j];
    }
    Atr_norm = std::sqrt(Atr_norm);
    Atb_norm = std::sqrt(Atb_norm);

    double rn = (Atb_norm > 0.0) ? Atr_norm / Atb_norm : Atr_norm;
    // Tolerance accounts for condition number of stacked Laplacian
    double tol = double(n) * double(n) * std::numeric_limits<double>::epsilon();

    std::cout << std::left << std::setw(16) << "TallSkinny"
              << "  m=" << std::setw(5) << m << " n=" << std::setw(5) << n
              << "  ||A'(Ax-b)||/||A'b||=" << std::scientific << std::setprecision(3) << rn
              << "  tol=" << tol
              << (rn < tol ? "  PASS" : "  FAIL")
              << std::defaultfloat << std::endl;
    REQUIRE(rn < tol);
}
