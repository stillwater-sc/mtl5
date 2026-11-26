// convection_diffusion.cpp - When CG Fails: Non-Symmetric Systems
//
// This example demonstrates:
//   1. How convection creates a non-symmetric matrix
//   2. Why CG fails on non-symmetric systems (it requires SPD)
//   3. Why BiCGSTAB succeeds where CG fails
//   4. Three iteration controllers: basic, cyclic, noisy
//
// Physics: 1D convection-diffusion equation:
//   -epsilon * u''(x) + u'(x) = f(x) on [0,1], u(0)=u(1)=0
// The first derivative u'(x) creates asymmetry in the discrete operator.
// Small epsilon means convection-dominated flow (harder to solve).

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#include "../common/replay.hpp"
#include "../common/trace.hpp"

using namespace mtl;

/// Build convection-diffusion matrix with central differences
mat::dense2D<double> build_conv_diff(std::size_t n, double epsilon) {
    double h = 1.0 / (n + 1);
    mat::dense2D<double> A(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        // Diffusion: -epsilon * u'' -> epsilon * (2u_i - u_{i-1} - u_{i+1}) / h^2
        A(i, i) = 2.0 * epsilon / (h * h);
        if (i > 0)     A(i, i-1) = -epsilon / (h * h) - 1.0 / (2.0 * h);  // + upwind
        if (i + 1 < n) A(i, i+1) = -epsilon / (h * h) + 1.0 / (2.0 * h);  // + downwind
    }
    return A;
}

/// The exact solution of -eps u'' + u' = 1, u(0) = u(1) = 0: a straight line
/// u = x that bends down to zero in a boundary layer of width ~eps at x = 1.
double exact_solution(double x, double eps) {
    return x - (std::exp((x - 1.0) / eps) - std::exp(-1.0 / eps)) / (1.0 - std::exp(-1.0 / eps));
}

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "3B convection-diffusion");

    std::cout << "=============================================================\n";
    std::cout << " Phase 3B: Convection-Diffusion - CG vs BiCGSTAB\n";
    std::cout << "=============================================================\n\n";

    const std::size_t n = 30;
    const double h = 1.0 / (n + 1);

    // Build RHS: simple constant forcing f=1
    vec::dense_vector<double> b(n, 1.0);

    // -- Regime 1: Diffusion-dominated (epsilon = 1.0) --------------------
    {
        double eps = 1.0;
        std::cout << "=== Regime 1: Diffusion-dominated (epsilon = " << eps << ") ===\n";
        std::cout << "When epsilon is large, the diffusion term -eps*u'' dominates.\n";
        std::cout << "The matrix is 'nearly symmetric' - CG may still work.\n\n";

        auto A = build_conv_diff(n, eps);

        // Show asymmetry: compare A(1,0) vs A(0,1)
        std::cout << "A(0,1) = " << std::setprecision(4) << A(0, 1)
                  << ",  A(1,0) = " << A(1, 0) << "\n";
        std::cout << "Asymmetry ratio: |A(0,1)-A(1,0)| / |A(0,1)+A(1,0)| = "
                  << std::abs(A(0,1) - A(1,0)) / std::abs(A(0,1) + A(1,0)) << "\n\n";

        // Try CG (will likely converge but shouldn't be trusted)
        vec::dense_vector<double> x_cg(n, 0.0);
        itl::pc::identity<mat::dense2D<double>> I(A);
        itl::basic_iteration<double> iter_cg(b, 500, 1.0e-8);
        int info_cg = itl::cg(A, x_cg, b, I, iter_cg);
        std::cout << "CG:      " << iter_cg.iterations() << " iterations, "
                  << "error code = " << info_cg << "\n";

        // BiCGSTAB (correct choice for non-symmetric)
        vec::dense_vector<double> x_bi(n, 0.0);
        itl::basic_iteration<double> iter_bi(b, 500, 1.0e-8);
        int info_bi = itl::bicgstab(A, x_bi, b, I, iter_bi);
        std::cout << "BiCGSTAB: " << iter_bi.iterations() << " iterations, "
                  << "error code = " << info_bi << "\n\n";
    }

    // -- Regime 2: Convection-dominated (epsilon = 0.01) ------------------
    {
        double eps = 0.01;
        std::cout << "=== Regime 2: Convection-dominated (epsilon = " << eps << ") ===\n";
        std::cout << "When epsilon is small, the convection term u' dominates.\n";
        std::cout << "The matrix is strongly non-symmetric - CG should fail.\n\n";

        auto A = build_conv_diff(n, eps);

        std::cout << "A(0,1) = " << std::setprecision(4) << A(0, 1)
                  << ",  A(1,0) = " << A(1, 0) << "\n";
        std::cout << "Asymmetry ratio: "
                  << std::abs(A(0,1) - A(1,0)) / std::abs(A(0,1) + A(1,0)) << "\n\n";

        // CG on strongly non-symmetric system
        vec::dense_vector<double> x_cg(n, 0.0);
        itl::pc::identity<mat::dense2D<double>> I(A);
        itl::basic_iteration<double> iter_cg(b, 500, 1.0e-8);
        int info_cg = itl::cg(A, x_cg, b, I, iter_cg);
        std::cout << "CG:       " << iter_cg.iterations() << " iterations, "
                  << "error code = " << info_cg;
        if (info_cg != 0) std::cout << " (FAILED - expected for non-SPD)";
        std::cout << "\n";

        // BiCGSTAB with diagonal preconditioner
        vec::dense_vector<double> x_bi(n, 0.0);
        itl::pc::diagonal<mat::dense2D<double>> diag_pc(A);
        itl::basic_iteration<double> iter_bi(b, 500, 1.0e-8);
        int info_bi = itl::bicgstab(A, x_bi, b, diag_pc, iter_bi);
        std::cout << "BiCGSTAB: " << iter_bi.iterations() << " iterations, "
                  << "error code = " << info_bi << "\n";

        // Verify solution: compute residual ||b - Ax||
        auto r = A * x_bi;
        double res_norm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            res_norm += (b(i) - r(i)) * (b(i) - r(i));
        res_norm = std::sqrt(res_norm);
        std::cout << "BiCGSTAB residual ||b - Ax||_2 = " << std::scientific << res_norm << "\n\n";
    }

    // -- Iteration Controllers Demo ---------------------------------------
    std::cout << "=== Iteration Controller Comparison ===\n";
    std::cout << "MTL5 provides three controllers with different verbosity:\n\n";

    double eps = 0.1;
    auto A = build_conv_diff(n, eps);
    itl::pc::identity<mat::dense2D<double>> I(A);

    // 1. basic_iteration - silent
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::basic_iteration<double> iter(b, 200, 1.0e-8);
        itl::bicgstab(A, x, b, I, iter);
        std::cout << "basic_iteration (silent): converged in "
                  << iter.iterations() << " iterations\n";
    }

    // 2. cyclic_iteration - prints every N steps
    {
        std::cout << "\ncyclic_iteration (every 5 steps):\n";
        vec::dense_vector<double> x(n, 0.0);
        itl::cyclic_iteration<double> iter(b, 200, 1.0e-8, 5);
        itl::bicgstab(A, x, b, I, iter);
        std::cout << "\n";
    }

    // 3. noisy_iteration - prints every step
    {
        std::cout << "noisy_iteration (every step):\n";
        vec::dense_vector<double> x(n, 0.0);
        itl::noisy_iteration<double> iter(b, 200, 1.0e-8);
        itl::bicgstab(A, x, b, I, iter);
        std::cout << "\n";
    }

    // -- Across epsilon -----------------------------------------------------
    // The same comparison as epsilon shrinks, both solvers unpreconditioned.
    // The cell Peclet number h / (2 eps) passing 1 is where central
    // differences start to oscillate next to the boundary layer.
    std::cout << "=== Across epsilon (no preconditioner) ===\n";
    std::cout << std::setw(8) << "epsilon" << std::setw(12) << "cell Pe" << std::setw(16)
              << "CG iters/code" << std::setw(22) << "BiCGSTAB iters/code" << '\n';
    const std::vector<double> sweep = {1.0, 0.1, 0.03, 0.01};
    auto runs = example_trace::value::array();
    for (double e : sweep) {
        auto Ae = build_conv_diff(n, e);
        itl::pc::identity<mat::dense2D<double>> Ie(Ae);
        vec::dense_vector<double> xc(n, 0.0), xb(n, 0.0);
        itl::basic_iteration<double> ic(b, 500, 1.0e-8), ib(b, 500, 1.0e-8);
        const int cc = itl::cg(Ae, xc, b, Ie, ic);
        const int cb = itl::bicgstab(Ae, xb, b, Ie, ib);
        const double pe = h / (2.0 * e);
        std::cout << std::fixed << std::setw(8) << std::setprecision(2) << e << std::setw(12)
                  << std::setprecision(3) << pe << std::setw(12) << ic.iterations() << " / " << cc
                  << std::setw(18) << ib.iterations() << " / " << cb << '\n';
        if (!trace) continue;

        // Replay: every BiCGSTAB step, and CG's first K steps (CG runs to the
        // 500-step cap, so its tail adds nothing to the picture).
        const int kb = ib.iterations();
        const int K = std::min(500, std::max(40, 2 * kb));
        auto run = example_trace::value::object();
        run["epsilon"] = e;
        run["cell_peclet"] = example_trace::value::rounded(pe, 6);
        auto history = [&](const char* name, int steps, int total, int code, double final_res, auto solve) {
            auto o = example_trace::value::object();
            o["name"] = name;
            o["iterations"] = total;
            o["code"] = code;
            o["final_residual"] = example_trace::value::rounded(final_res, 6);
            auto it = example_trace::value::array(), rs = example_trace::value::array();
            for (int k = 0; k <= steps; ++k) {
                auto x = example_trace::iterate_after(b, k, solve);
                it.push_back(example_trace::value::numbers(example_trace::to_std(x), 4));
                rs.push_back(example_trace::value::rounded(example_trace::true_residual(Ae, x, b), 5));
            }
            o["iterates"] = it;
            o["residual"] = rs;
            return o;
        };
        auto solvers = example_trace::value::array();
        solvers.push_back(history("cg", K, ic.iterations(), cc, example_trace::true_residual(Ae, xc, b),
                                  [&](auto& x, auto& it) { itl::cg(Ae, x, b, Ie, it); }));
        solvers.push_back(history("bicgstab", kb, kb, cb, example_trace::true_residual(Ae, xb, b),
                                  [&](auto& x, auto& it) { itl::bicgstab(Ae, x, b, Ie, it); }));
        run["solvers"] = solvers;
        std::vector<double> xs(n), ue(n), fine_x, fine_u;
        for (std::size_t i = 0; i < n; ++i) {
            xs[i] = (i + 1) * h;
            ue[i] = exact_solution(xs[i], e);
        }
        for (int i = 0; i <= 300; ++i) {
            fine_x.push_back(i / 300.0);
            fine_u.push_back(exact_solution(i / 300.0, e));
        }
        run["exact_on_grid"] = example_trace::value::numbers(ue, 6);
        run["exact_x"] = example_trace::value::numbers(fine_x, 5);
        run["exact_u"] = example_trace::value::numbers(fine_u, 5);
        runs.push_back(run);
    }
    std::cout << "\n";

    if (trace) {
        trace["n"] = n;
        trace["tolerance"] = 1.0e-8;
        std::vector<double> xs(n);
        for (std::size_t i = 0; i < n; ++i) xs[i] = (i + 1) * h;
        trace["x"] = example_trace::value::numbers(xs, 6);
        trace["runs"] = runs;
    }

    // -- Commentary -------------------------------------------------------
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. CG requires the matrix to be SPD (Symmetric Positive Definite).\n";
    std::cout << "   On non-symmetric systems it may diverge or give wrong answers.\n";
    std::cout << "2. BiCGSTAB handles non-symmetric systems by using two sequences\n";
    std::cout << "   of residuals. It's the go-to solver for general sparse systems.\n";
    std::cout << "3. The Peclet number Pe = |velocity|*h / (2*epsilon) determines\n";
    std::cout << "   whether the problem is diffusion- or convection-dominated.\n";
    std::cout << "4. Choose your iteration controller based on debugging needs:\n";
    std::cout << "   - basic: production code (no overhead)\n";
    std::cout << "   - cyclic: monitoring long runs\n";
    std::cout << "   - noisy: debugging convergence issues\n";

    return trace.close();   // EXIT_FAILURE if a requested trace could not be written
}
