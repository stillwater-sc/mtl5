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
#include <mtl/operation/norms.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/diagonal.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/generators/laplacian.hpp>

using namespace mtl;

namespace {

mat::compressed2D<double> make_convdiff_1d(std::size_t n, double eps = 0.1) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0;
            if (i > 0)     ins[i][i - 1] << (-1.0 - eps);
            if (i + 1 < n) ins[i][i + 1] << (-1.0 + eps);
        }
    }
    return A;
}

double sparse_relative_residual(const mat::compressed2D<double>& A,
                                const vec::dense_vector<double>& x,
                                const vec::dense_vector<double>& b) {
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    double res2 = 0.0, b2 = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x(indices[k]);
        double ri = Ax_i - b(i);
        res2 += ri * ri;
        b2 += b(i) * b(i);
    }
    if (b2 == 0.0) return std::sqrt(res2);
    return std::sqrt(res2 / b2);
}

void report(const char* solver, const char* pc, const char* matrix,
            std::size_t n, int iters, double rr, double tol) {
    std::cout << std::left << std::setw(12) << solver
              << std::setw(8) << pc
              << std::setw(14) << matrix
              << "  n=" << std::setw(7) << n
              << "  iters=" << std::setw(6) << iters
              << "  residual=" << std::scientific << std::setprecision(3) << rr
              << "  tol=" << tol
              << (rr < tol ? "  PASS" : "  FAIL")
              << std::defaultfloat << std::endl;
}

} // anonymous namespace

TEST_CASE("BiCGSTAB regression: 2D Laplacian, identity PC", "[regression][itl][bicgstab]") {
    auto k = GENERATE(50, 100);
    std::size_t n = std::size_t(k) * std::size_t(k);

    auto A = generators::laplacian_2d<double>(k, k);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 10000, 1e-10);
    int err = itl::bicgstab(A, x, b, pc, iter);

    double rr = sparse_relative_residual(A, x, b);
    report("BiCGSTAB", "none", "Laplacian-2D", n, iter.iterations(), rr, 1e-8);
    REQUIRE(err == 0);
    REQUIRE(rr < 1e-8);
}

TEST_CASE("BiCGSTAB regression: 1D convection-diffusion, ILU(0) PC", "[regression][itl][bicgstab]") {
    auto n = GENERATE(10000, 50000);

    auto A = make_convdiff_1d(n);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ilu_0<double> pc(A);
    itl::basic_iteration<double> iter(b, 5000, 1e-10);
    int err = itl::bicgstab(A, x, b, pc, iter);

    double rr = sparse_relative_residual(A, x, b);
    report("BiCGSTAB", "ILU(0)", "ConvDiff-1D", n, iter.iterations(), rr, 1e-8);
    REQUIRE(err == 0);
    REQUIRE(rr < 1e-8);
}

TEST_CASE("BiCGSTAB regression: 2D Laplacian, Jacobi PC", "[regression][itl][bicgstab]") {
    auto k = GENERATE(100, 158);
    std::size_t n = std::size_t(k) * std::size_t(k);

    auto A = generators::laplacian_2d<double>(k, k);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 10000, 1e-10);
    int err = itl::bicgstab(A, x, b, pc, iter);

    double rr = sparse_relative_residual(A, x, b);
    report("BiCGSTAB", "Jacobi", "Laplacian-2D", n, iter.iterations(), rr, 1e-8);
    REQUIRE(err == 0);
    REQUIRE(rr < 1e-8);
}
