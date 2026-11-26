// multigrid_poisson.cpp -- Multigrid for 1D Poisson
//
// This example demonstrates multigrid methods for solving -u'' = f on [0,1]:
//
//   1. Build a geometric multigrid hierarchy, coarsening n -> (n - 1) / 2
//      down to 3 points, where an exact coarse solve costs nothing
//   2. Show V-cycle convergence as a stand-alone solver
//   3. Show that the cycle count does not grow with n -- and that a fixed
//      3-level hierarchy, whose coarse grid grows with n, loses that
//   4. Use the V-cycle as a CG preconditioner, which needs a symmetric
//      V-cycle: symmetric Gauss-Seidel and an exact coarse solve
//   5. Compare with CG alone
//
// The 1D Poisson equation with Dirichlet BCs is a classic benchmark for
// multigrid. With a full hierarchy, a V-cycle costs O(n) and the number of
// cycles does not depend on n, so the whole solve is O(n).

#include <mtl/mtl.hpp>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace mtl;
using csr = mat::compressed2D<double>;
using vec_t  = vec::dense_vector<double>;

/// The 1D Poisson matrix for n interior points: the Laplacian tridiag(-1, 2,
/// -1) from the generators, scaled by 1/h^2 = (n + 1)^2.
static csr poisson(std::size_t n) {
    csr A = generators::laplacian_1d<double>(n);
    const double h2_inv = static_cast<double>((n + 1) * (n + 1));
    for (auto& v : A.ref_data()) v *= h2_inv;
    return A;
}

static double relative_residual(const csr& A, const vec_t& x, const vec_t& b) {
    auto Ax = A * x;
    vec_t r(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) r(i) = b(i) - Ax(i);
    return two_norm(r) / two_norm(b);
}

enum class smoothing { gauss_seidel, symmetric_gauss_seidel };
enum class coarse { gauss_seidel_100, exact };

/// A V-cycle hierarchy for n0 = 2^k - 1 points. `levels` = 0 coarsens until
/// the grid has 3 points; otherwise the hierarchy has that many levels.
static itl::mg::multigrid<double> make_multigrid(std::size_t n0, std::size_t levels,
                                                 smoothing sm, coarse cs) {
    std::vector<csr> A, R, P;
    std::size_t n = n0;
    A.push_back(poisson(n));
    while ((levels == 0 && n > 3) || (levels != 0 && A.size() < levels)) {
        const std::size_t nc = (n - 1) / 2;
        R.push_back(itl::mg::make_restriction_1d(n));
        P.push_back(itl::mg::make_prolongation_1d(nc));
        A.push_back(poisson(nc));
        n = nc;
    }

    std::function<void(vec_t&, const vec_t&)> coarse_solver;
    const csr& Ac = A.back();
    if (cs == coarse::exact) {
        // A dense LU of the coarsest matrix, factored once: on 3 points it is
        // a 3 x 3 solve. An exact solve is symmetric, as CG needs.
        mat::dense2D<double> LU(Ac.num_rows(), Ac.num_cols());
        for (std::size_t i = 0; i < Ac.num_rows(); ++i)
            for (std::size_t j = 0; j < Ac.num_cols(); ++j) LU(i, j) = Ac(i, j);
        std::vector<std::size_t> pivot;
        lu_factor(LU, pivot);
        coarse_solver = [LU, pivot](vec_t& x, const vec_t& b) { lu_solve(LU, pivot, x, b); };
    } else {
        // 100 forward Gauss-Seidel sweeps. Not exact: their convergence factor
        // on m points is cos^2(pi / (m + 1)), 0.96 on 15 points, so 100 sweeps
        // leave about 2% of the error there, and far more on a larger grid.
        coarse_solver = [Ac](vec_t& x, const vec_t& b) {
            itl::smoother::gauss_seidel<csr> gs(Ac);
            for (int i = 0; i < 100; ++i) gs(x, b);
        };
    }

    if (sm == smoothing::symmetric_gauss_seidel) {
        auto factory = [](const csr& M) { return itl::smoother::symmetric_gauss_seidel<csr>(M); };
        return itl::mg::multigrid<double>(A, R, P, factory, coarse_solver, 2, 2);
    }
    auto factory = [](const csr& M) { return itl::smoother::gauss_seidel<csr>(M); };
    return itl::mg::multigrid<double>(A, R, P, factory, coarse_solver, 2, 2);
}

/// V-cycles from x = 0 until ||b - A x|| / ||b|| < tol; 0 if not in max_cycles.
static int vcycles_to(const itl::mg::multigrid<double>& mg, const csr& A, const vec_t& b,
                      double tol, int max_cycles) {
    vec_t x(b.size(), 0.0);
    for (int c = 1; c <= max_cycles; ++c) {
        mg.vcycle(x, b);
        if (relative_residual(A, x, b) < tol) return c;
    }
    return 0;
}

/// MG-preconditioned CG iterations to tol; 0 if not converged in max_iter.
static int pcg_iterations(const itl::mg::multigrid<double>& mg, const csr& A, const vec_t& b,
                          double tol, int max_iter) {
    vec_t x(b.size(), 0.0);
    itl::basic_iteration<double> iter(b, max_iter, tol);
    const int info = itl::cg(A, x, b, mg, iter);
    return info == 0 ? static_cast<int>(iter.iterations()) : 0;
}

/// ||B - B^T||_F / ||B||_F for the preconditioner B: x -> mg.solve(x),
/// built column by column from unit vectors.
static double asymmetry(const itl::mg::multigrid<double>& mg, std::size_t n) {
    mat::dense2D<double> B(n, n);
    vec_t e(n, 0.0), col(n);
    for (std::size_t j = 0; j < n; ++j) {
        e(j) = 1.0;
        mg.solve(col, e);
        for (std::size_t i = 0; i < n; ++i) B(i, j) = col(i);
        e(j) = 0.0;
    }
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            diff += (B(i, j) - B(j, i)) * (B(i, j) - B(j, i));
            norm += B(i, j) * B(i, j);
        }
    return std::sqrt(diff / norm);
}

static std::string count(int c, int cap) {
    return c > 0 ? std::to_string(c) : "not in " + std::to_string(cap);
}

int main() {
    std::cout << "=== Multigrid for 1D Poisson Equation ===\n\n";
    const double tol = 1e-10;

    // -- 1. The hierarchy ------------------------------------------------
    const std::size_t n0 = 63;
    std::cout << "Hierarchy for n = " << n0 << ":";
    for (std::size_t n = n0; ; n = (n - 1) / 2) {
        std::cout << " " << n;
        if (n <= 3) break;
        std::cout << " ->";
    }
    std::cout << "\nSmoother: symmetric Gauss-Seidel, 2 sweeps before and after the\n";
    std::cout << "coarse correction. Coarse solve: dense LU of the 3 x 3 matrix.\n\n";

    const csr A0 = poisson(n0);
    const vec_t b(n0, 1.0);   // constant forcing f(x) = 1
    auto mg = make_multigrid(n0, 0, smoothing::symmetric_gauss_seidel, coarse::exact);

    // -- 2. V-cycle as stand-alone solver --------------------------------
    std::cout << "--- V-cycle as stand-alone solver ---\n";
    std::cout << std::setw(8) << "Cycle" << std::setw(16) << "||r|| / ||b||" << "\n";
    std::cout << std::string(24, '-') << "\n";
    {
        vec_t x(n0, 0.0);
        double prev = 1.0;
        for (int cycle = 1; cycle <= 15; ++cycle) {
            mg.vcycle(x, b);
            const double rel = relative_residual(A0, x, b);
            std::cout << std::setw(8) << cycle << std::scientific << std::setprecision(3)
                      << std::setw(16) << rel << "   (x " << std::fixed << std::setprecision(3)
                      << rel / prev << ")\n";
            prev = rel;
            if (rel < tol) break;
        }
        std::cout << "Each cycle cuts the residual by a roughly constant factor.\n\n";
    }

    // -- 3. Cost does not grow with n: the full hierarchy ----------------
    // 1e-8, not 1e-10: cond(A) grows like n^2, and at n = 4095 rounding
    // stops the residual near 1e-10, so the last digits would cost cycles
    // that say nothing about multigrid.
    const double sweep_tol = 1e-8;
    std::cout << "--- Cycles to 1e-8 as n grows ---\n";
    std::cout << "3 levels: the example's original setup, forward Gauss-Seidel smoothing and\n";
    std::cout << "100 forward Gauss-Seidel sweeps as the coarse solve. Full: coarsen to 3\n";
    std::cout << "points, symmetric Gauss-Seidel, exact coarse solve. (1e-8 rather than\n";
    std::cout << "1e-10, because at n = 4095 rounding stops the residual near 1e-10.)\n\n";
    std::cout << std::setw(7) << "n" << std::setw(18) << "3 levels: coarse"
              << std::setw(14) << "V-cycles" << std::setw(12) << "MG-PCG"
              << std::setw(16) << "full: levels" << std::setw(11) << "V-cycles"
              << std::setw(10) << "MG-PCG" << "\n";
    for (std::size_t n : {63, 255, 1023, 4095}) {
        const csr A = poisson(n);
        const vec_t bn(n, 1.0);
        auto three = make_multigrid(n, 3, smoothing::gauss_seidel, coarse::gauss_seidel_100);
        auto full = make_multigrid(n, 0, smoothing::symmetric_gauss_seidel, coarse::exact);
        std::size_t full_levels = 1;
        for (std::size_t m = n; m > 3; m = (m - 1) / 2) ++full_levels;
        std::cout << std::setw(7) << n << std::setw(18) << (n - 1) / 4 << std::setw(14)
                  << count(vcycles_to(three, A, bn, sweep_tol, 100), 100) << std::setw(12)
                  << count(pcg_iterations(three, A, bn, sweep_tol, 500), 500) << std::setw(16) << full_levels
                  << std::setw(11) << count(vcycles_to(full, A, bn, sweep_tol, 100), 100) << std::setw(10)
                  << count(pcg_iterations(full, A, bn, sweep_tol, 500), 500) << "\n";
    }
    std::cout << "With 3 fixed levels the coarse grid grows with n, and 100 sweeps stop\n";
    std::cout << "being anything like a solve. Recursing to 3 points keeps the coarse solve\n";
    std::cout << "exact and free, and the cycle count flat: that is multigrid's O(n).\n\n";

    // -- 4. The V-cycle as a CG preconditioner: it must be symmetric ------
    std::cout << "--- A CG preconditioner must be symmetric ---\n";
    std::cout << "CG needs a symmetric positive definite preconditioner B. A V-cycle is\n";
    std::cout << "symmetric only if its post-smoother is the adjoint of its pre-smoother and\n";
    std::cout << "its coarse solve is symmetric. Forward Gauss-Seidel before and after is not;\n";
    std::cout << "symmetric Gauss-Seidel (a forward and a backward sweep) is. Measured at\n";
    std::cout << "n = " << n0 << ", 3 levels:\n\n";
    std::cout << std::setw(26) << "smoother" << std::setw(20) << "coarse solve" << std::setw(20)
              << "||B - B^T||/||B||" << std::setw(10) << "MG-PCG" << "\n";
    for (auto sm : {smoothing::gauss_seidel, smoothing::symmetric_gauss_seidel})
        for (auto cs : {coarse::gauss_seidel_100, coarse::exact}) {
            auto m = make_multigrid(n0, 3, sm, cs);
            std::cout << std::setw(26) << (sm == smoothing::gauss_seidel ? "gauss_seidel" : "symmetric_gauss_seidel")
                      << std::setw(20) << (cs == coarse::exact ? "exact (LU)" : "100 GS sweeps")
                      << std::scientific << std::setprecision(1) << std::setw(20) << asymmetry(m, n0)
                      << std::setw(10) << count(pcg_iterations(m, A0, b, tol, 500), 500) << "\n";
        }
    std::cout << "Only the symmetric smoother with the exact coarse solve gives a symmetric\n";
    std::cout << "B (to rounding). PCG may still converge with the others, but CG's theory\n";
    std::cout << "does not cover them, and on a harder problem they can stall it.\n\n";

    // -- 5. CG without preconditioning -----------------------------------
    std::cout << "--- CG without preconditioning ---\n";
    auto plain_cg = [&](const vec_t& rhs) {
        vec_t x(n0, 0.0);
        itl::pc::identity<csr> pc(A0);
        itl::basic_iteration<double> iter(rhs, 500, tol);
        itl::cg(A0, x, rhs, pc, iter);
        return std::pair<std::size_t, double>(iter.iterations(), relative_residual(A0, x, rhs));
    };
    {
        const auto [its, rel] = plain_cg(b);
        std::cout << "f = 1:       " << its << " iterations, relative residual " << std::scientific
                  << std::setprecision(2) << rel << "\n";
        vec_t b2(n0);
        for (std::size_t i = 0; i < n0; ++i) b2(i) = static_cast<double>(i + 1) / static_cast<double>(n0 + 1);
        const auto [its2, rel2] = plain_cg(b2);
        std::cout << "f = x:       " << its2 << " iterations, relative residual " << rel2 << "\n";
        std::cout << "f = 1 is symmetric about the midpoint, so it has no component along the\n";
        std::cout << "antisymmetric half of the eigenvectors, and CG finishes in (n + 1) / 2 = "
                  << (n0 + 1) / 2 << "\n";
        std::cout << "steps. A right-hand side without that symmetry takes up to n = " << n0
                  << ". Either way CG's\n";
        std::cout << "count grows with n; MG-PCG's above does not.\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
