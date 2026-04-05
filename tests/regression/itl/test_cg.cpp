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
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/generators/laplacian.hpp>
#include <mtl/generators/poisson.hpp>

using namespace mtl;

namespace {

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
    std::cout << std::left << std::setw(8) << solver
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

TEST_CASE("CG regression: 2D Laplacian, identity PC", "[regression][itl][cg]") {
    auto k = GENERATE(100, 224);
    std::size_t n = std::size_t(k) * std::size_t(k);

    auto A = generators::laplacian_2d<double>(k, k);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 5000, 1e-10);
    int err = itl::cg(A, x, b, pc, iter);

    double rr = sparse_relative_residual(A, x, b);
    report("CG", "none", "Laplacian-2D", n, iter.iterations(), rr, 1e-8);
    REQUIRE(err == 0);
    REQUIRE(rr < 1e-8);
}

TEST_CASE("CG regression: 2D Laplacian, Jacobi PC", "[regression][itl][cg]") {
    auto k = GENERATE(100, 224);
    std::size_t n = std::size_t(k) * std::size_t(k);

    auto A = generators::laplacian_2d<double>(k, k);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 5000, 1e-10);
    int err = itl::cg(A, x, b, pc, iter);

    double rr = sparse_relative_residual(A, x, b);
    report("CG", "Jacobi", "Laplacian-2D", n, iter.iterations(), rr, 1e-8);
    REQUIRE(err == 0);
    REQUIRE(rr < 1e-8);
}

TEST_CASE("CG regression: 2D Poisson, IC(0) PC", "[regression][itl][cg]") {
    auto k = GENERATE(50, 100);
    std::size_t n = std::size_t(k) * std::size_t(k);

    auto A = generators::poisson2d_dirichlet<double>(k, k);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ic_0<double> pc(A);
    itl::basic_iteration<double> iter(b, 5000, 1e-10);
    int err = itl::cg(A, x, b, pc, iter);

    double rr = sparse_relative_residual(A, x, b);
    report("CG", "IC(0)", "Poisson-2D", n, iter.iterations(), rr, 1e-8);
    REQUIRE(err == 0);
    REQUIRE(rr < 1e-8);
}

TEST_CASE("CG regression: 1D Laplacian 10K DOF", "[regression][itl][cg]") {
    std::size_t n = 10000;

    auto A = generators::laplacian_1d<double>(n);
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 50000, 1e-10);
    int err = itl::cg(A, x, b, pc, iter);

    double rr = sparse_relative_residual(A, x, b);
    report("CG", "Jacobi", "Laplacian-1D", n, iter.iterations(), rr, 1e-6);
    REQUIRE(err == 0);
    REQUIRE(rr < 1e-6);
}
