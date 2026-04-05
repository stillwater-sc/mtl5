#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>

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
