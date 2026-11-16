// Regression tier for the Krylov solvers that had none (issue 866): cgs, qmr,
// tfqmr, minres, idr_s and bicgstab_ell, at 4900-10000 unknowns. Each solve
// is judged by its true relative residual ||b - A x|| / ||b||, not the
// solver's own estimate.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/itl.hpp>
#include <mtl/generators/laplacian.hpp>

using namespace mtl;

namespace {

using csr = mat::compressed2D<double>;
using vec_t  = vec::dense_vector<double>;

/// The 5-point convection-diffusion operator on a k x k grid, convection
/// eps in both directions: nonsymmetric, and ILU(0) is not exact on it (on a
/// 1D tridiagonal operator ILU(0) is an exact LU, and every solver finished
/// in one step).
csr convdiff_2d(std::size_t k, double eps = 0.3) {
    const std::size_t n = k * k;
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t r = i * k + j;
                ins[r][r] << 4.0;
                if (j > 0)     ins[r][r - 1] << (-1.0 - eps);
                if (j + 1 < k) ins[r][r + 1] << (-1.0 + eps);
                if (i > 0)     ins[r][r - k] << (-1.0 - eps);
                if (i + 1 < k) ins[r][r + k] << (-1.0 + eps);
            }
    }
    return A;
}

double true_relres(const csr& A, const vec_t& x, const vec_t& b) {
    const auto& rp = A.ref_major(); const auto& ci = A.ref_minor(); const auto& d = A.ref_data();
    double r2 = 0.0, b2 = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) ax += d[k] * x(ci[k]);
        r2 += (b(i) - ax) * (b(i) - ax);
        b2 += b(i) * b(i);
    }
    return std::sqrt(r2 / b2);
}

void report(const char* solver, const char* pc, const char* matrix, std::size_t n, int iters,
            double rr) {
    std::cout << std::left << std::setw(14) << solver << std::setw(8) << pc << std::setw(14) << matrix
              << "  n=" << std::setw(7) << n << "  iters=" << std::setw(6) << iters
              << "  residual=" << std::scientific << std::setprecision(3) << rr << std::defaultfloat
              << std::endl;
}

/// Solve with `solve(A, x, b, pc, iter)` from x = 0 at `rtol` within
/// `max_iter`; require convergence, at most `max_iter` iterations, and a
/// true residual below `bound`.
template <typename Solve, typename PC>
void run_at(const char* name, const char* pcname, const char* matrix, const csr& A, const PC& pc,
            Solve solve, int max_iter, double rtol, double bound) {
    const std::size_t n = A.num_rows();
    vec_t b(n, 1.0), x(n, 0.0);
    itl::basic_iteration<double> iter(b, max_iter, rtol);
    const int err = solve(A, x, b, pc, iter);
    const double rr = true_relres(A, x, b);
    report(name, pcname, matrix, n, iter.iterations(), rr);
    CHECK(err == 0);
    CHECK(rr < bound);
}

/// Solve with `solve(A, x, b, pc, iter)` from x = 0 at rtol 1e-10; require
/// convergence and a true residual below `bound` (1e-8 unless stated).
template <typename Solve, typename PC>
void run(const char* name, const char* pcname, const char* matrix, const csr& A, const PC& pc,
         Solve solve, int max_iter, double bound = 1e-8) {
    const std::size_t n = A.num_rows();
    vec_t b(n, 1.0), x(n, 0.0);
    itl::basic_iteration<double> iter(b, max_iter, 1e-10);
    const int err = solve(A, x, b, pc, iter);
    const double rr = true_relres(A, x, b);
    report(name, pcname, matrix, n, iter.iterations(), rr);
    CHECK(err == 0);
    CHECK(rr < bound);
}

} // namespace

TEST_CASE("Krylov family regression: 2D convection-diffusion, ILU(0) PC", "[regression][itl][krylov]") {
    const csr A = convdiff_2d(100);   // n = 10000
    const itl::pc::ilu_0<double> pc(A);
    // CGS reported 1e-10 here at a true 7.6e-8 before it checked the true
    // residual (issue 873).
    run("CGS", "ILU(0)", "ConvDiff-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::cgs(M, x, b, P, it); }, 5000);
    run("QMR", "ILU(0)", "ConvDiff-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::qmr(M, x, b, P, it); }, 5000);
    run("TFQMR", "ILU(0)", "ConvDiff-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::tfqmr(M, x, b, P, it); }, 5000);
    run("IDR(4)", "ILU(0)", "ConvDiff-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::idr_s(M, x, b, P, it, 4); }, 5000);
    run("BiCGSTAB(2)", "ILU(0)", "ConvDiff-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::bicgstab_ell(M, x, b, P, it, 2); },
        5000);
}

TEST_CASE("Krylov family regression: strongly nonnormal convection-diffusion, no PC", "[regression][itl][krylov]") {
    // eps = 0.9: BiCGSTAB reported convergence at a true residual of 0.2, and
    // BiCGSTAB(2) at 7e-3, before they checked the true residual (issue 873).
    const csr A = convdiff_2d(100, 0.9);   // n = 10000
    const itl::pc::identity<csr> pc(A);
    run("BiCGSTAB", "none", "ConvDiff-0.9", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::bicgstab(M, x, b, P, it); }, 5000);
    run("BiCGSTAB(2)", "none", "ConvDiff-0.9", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::bicgstab_ell(M, x, b, P, it, 2); },
        5000);
    run("IDR(4)", "none", "ConvDiff-0.9", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::idr_s(M, x, b, P, it, 4); }, 5000);
    run("GMRES(30)", "none", "ConvDiff-0.9", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::gmres(M, x, b, P, it, 30); }, 5000);
}

TEST_CASE("Krylov family regression: IDR(s) across s at tight tolerances", "[regression][itl][krylov]") {
    // The prototype IDR(s) took 729 (s = 2) and 1612 (s = 8) iterations at
    // rtol 1e-13 here, and stopped at true residuals 5-35x the tolerance; on
    // the 200^2 grid s = 2, 4, 8 ran to 20000 iterations (issue 874).
    const csr A = convdiff_2d(200);   // n = 40000
    const itl::pc::ilu_0<double> pc(A);
    for (const std::size_t s : {1u, 2u, 4u, 8u, 16u}) {
        const std::string name = "IDR(" + std::to_string(s) + ")";
        run_at(name.c_str(), "ILU(0)", "ConvDiff-2D", A, pc,
               [s](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::idr_s(M, x, b, P, it, s); },
               400, 1e-13, 1.5e-13);
    }
    const csr B = convdiff_2d(100, 0.9);
    const itl::pc::identity<csr> none(B);
    for (const std::size_t s : {1u, 2u, 4u, 8u}) {
        const std::string name = "IDR(" + std::to_string(s) + ")";
        run_at(name.c_str(), "none", "ConvDiff-0.9", B, none,
               [s](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::idr_s(M, x, b, P, it, s); },
               800, 1e-13, 1.5e-13);
    }
}

TEST_CASE("Krylov family regression: 2D Laplacian, Jacobi PC", "[regression][itl][krylov]") {
    const std::size_t k = 70, n = k * k;
    const csr A = generators::laplacian_2d<double>(k, k);
    const itl::pc::diagonal<csr> pc(A);
    run("MINRES", "Jacobi", "Laplacian-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::minres(M, x, b, P, it); }, 20000);
    run("CGS", "Jacobi", "Laplacian-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::cgs(M, x, b, P, it); }, 20000);
    run("TFQMR", "Jacobi", "Laplacian-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::tfqmr(M, x, b, P, it); }, 20000);
    run("IDR(4)", "Jacobi", "Laplacian-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::idr_s(M, x, b, P, it, 4); }, 20000);
    run("BiCGSTAB(4)", "Jacobi", "Laplacian-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::bicgstab_ell(M, x, b, P, it, 4); },
        20000);
    (void)n;
}

TEST_CASE("Krylov family regression: 2D Laplacian, no PC", "[regression][itl][krylov]") {
    const std::size_t k = 70;
    const csr A = generators::laplacian_2d<double>(k, k);
    const itl::pc::identity<csr> pc(A);
    run("MINRES", "none", "Laplacian-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::minres(M, x, b, P, it); }, 20000);
    run("QMR", "none", "Laplacian-2D", A, pc,
        [](auto& M, auto& x, auto& b, auto& P, auto& it) { return itl::qmr(M, x, b, P, it); }, 20000);
}
