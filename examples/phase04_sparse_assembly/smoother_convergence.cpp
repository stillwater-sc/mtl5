// smoother_convergence.cpp - Stationary Iterative Methods
//
// This example demonstrates:
//   1. Jacobi, Gauss-Seidel, and SOR as standalone solvers
//   2. How SOR(omega=1) reduces to Gauss-Seidel
//   3. The effect of relaxation parameter omega on convergence
//   4. Why these methods are slow but useful as smoothers in multigrid
//
// We solve the 1D Poisson equation -u'' = f on [0,1] and track
// the relative residual ||b - Ax||_2 / ||b||_2 at each sweep.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include "../common/replay.hpp"
#include "../common/trace.hpp"

using namespace mtl;

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "4B smoothers as solvers");

    std::cout << "=============================================================\n";
    std::cout << " Phase 4B: Smoother Convergence - Jacobi / GS / SOR\n";
    std::cout << "=============================================================\n\n";

    // -- Build 1D Poisson system (sparse) ---------------------------------
    const std::size_t n = 50;
    const double h = 1.0 / (n + 1);

    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A, 3);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0 / (h * h);
            if (i > 0)     ins[i][i-1] << -1.0 / (h * h);
            if (i + 1 < n) ins[i][i+1] << -1.0 / (h * h);
        }
    }

    // RHS: f(x) = 1 (constant)
    vec::dense_vector<double> b(n, 1.0);
    double b_norm = two_norm(b);

    // Helper: compute relative residual
    auto rel_residual = [&](const vec::dense_vector<double>& x) {
        auto r = A * x;
        vec::dense_vector<double> res(n);
        for (std::size_t i = 0; i < n; ++i)
            res(i) = b(i) - r(i);
        return two_norm(res) / b_norm;
    };

    const int max_sweeps = 100;
    const int print_interval = 10;

    // -- Jacobi -----------------------------------------------------------
    std::cout << "--- Jacobi Method ---\n";
    std::cout << "Updates all x_i simultaneously using old values.\n";
    std::cout << "Each sweep requires one matrix-vector-like pass.\n\n";

    itl::smoother::jacobi<mat::compressed2D<double>> jac(A);
    vec::dense_vector<double> x_jac(n, 0.0);
    std::vector<double> hist_jac;

    for (int sweep = 0; sweep <= max_sweeps; ++sweep) {
        double rr = rel_residual(x_jac);
        hist_jac.push_back(rr);
        if (sweep % print_interval == 0)
            std::cout << "  sweep " << std::setw(3) << sweep
                      << "  rel_resid = " << std::scientific << rr << "\n";
        if (sweep < max_sweeps) jac(x_jac, b);
    }

    // -- Gauss-Seidel -----------------------------------------------------
    std::cout << "\n--- Gauss-Seidel Method ---\n";
    std::cout << "Updates x_i in-place, immediately using newest values.\n";
    std::cout << "Converges ~2x faster than Jacobi for 1D Poisson.\n\n";

    itl::smoother::gauss_seidel<mat::compressed2D<double>> gs(A);
    vec::dense_vector<double> x_gs(n, 0.0);
    std::vector<double> hist_gs;

    for (int sweep = 0; sweep <= max_sweeps; ++sweep) {
        double rr = rel_residual(x_gs);
        hist_gs.push_back(rr);
        if (sweep % print_interval == 0)
            std::cout << "  sweep " << std::setw(3) << sweep
                      << "  rel_resid = " << std::scientific << rr << "\n";
        if (sweep < max_sweeps) gs(x_gs, b);
    }

    // -- SOR with multiple omega values -----------------------------------
    std::cout << "\n--- SOR (Successive Over-Relaxation) ---\n";
    std::cout << "x_new = omega * GS_update + (1-omega) * x_old\n";
    std::cout << "omega = 1.0 is Gauss-Seidel. Optimal omega for 1D Poisson:\n";
    double omega_opt = 2.0 / (1.0 + std::sin(std::numbers::pi * h));
    std::cout << "  omega_opt = 2/(1 + sin(pi*h)) = " << std::fixed
              << std::setprecision(4) << omega_opt << "\n\n";

    std::vector<double> omegas = {0.5, 1.0, 1.2, 1.5, omega_opt};
    std::vector<std::string> omega_names = {"0.50", "1.00", "1.20", "1.50", "opt "};
    std::vector<std::vector<double>> hist_sor(omegas.size());

    for (std::size_t w = 0; w < omegas.size(); ++w) {
        itl::smoother::sor<mat::compressed2D<double>> sor_smoother(A, omegas[w]);
        vec::dense_vector<double> x_sor(n, 0.0);

        for (int sweep = 0; sweep <= max_sweeps; ++sweep) {
            hist_sor[w].push_back(rel_residual(x_sor));
            if (sweep < max_sweeps) sor_smoother(x_sor, b);
        }
    }

    // Print SOR comparison table
    std::cout << std::setw(6) << "Sweep";
    for (std::size_t w = 0; w < omegas.size(); ++w)
        std::cout << std::setw(14) << ("w=" + omega_names[w]);
    std::cout << "\n" << std::string(6 + 14 * omegas.size(), '-') << "\n";

    for (int sweep = 0; sweep <= max_sweeps; sweep += print_interval) {
        std::cout << std::setw(6) << sweep;
        for (std::size_t w = 0; w < omegas.size(); ++w)
            std::cout << std::scientific << std::setw(14) << std::setprecision(3)
                      << hist_sor[w][sweep];
        std::cout << "\n";
    }

    // -- Final Summary ----------------------------------------------------
    std::cout << "\n--- Convergence after " << max_sweeps << " sweeps ---\n";
    std::cout << std::setw(20) << "Method" << std::setw(16) << "Rel Residual" << "\n";
    std::cout << std::string(36, '-') << "\n";
    std::cout << std::setw(20) << "Jacobi"
              << std::scientific << std::setw(16) << hist_jac.back() << "\n";
    std::cout << std::setw(20) << "Gauss-Seidel"
              << std::scientific << std::setw(16) << hist_gs.back() << "\n";
    for (std::size_t w = 0; w < omegas.size(); ++w)
        std::cout << std::setw(20) << ("SOR(w=" + omega_names[w] + ")")
                  << std::scientific << std::setw(16) << hist_sor[w].back() << "\n";

    // -- The omega sweep -----------------------------------------------------
    // Residual after max_sweeps SOR sweeps for omega across (0, 2).
    std::cout << "\n--- SOR residual after " << max_sweeps << " sweeps, across omega ---\n";
    std::vector<double> sweep_omegas, sweep_resid;
    for (int k = 1; k <= 39; ++k) sweep_omegas.push_back(0.05 * k);   // 0.05 .. 1.95
    sweep_omegas.push_back(omega_opt);
    std::sort(sweep_omegas.begin(), sweep_omegas.end());
    double best_omega = 0.0, best_resid = 1.0e300;
    for (double w : sweep_omegas) {
        itl::smoother::sor<mat::compressed2D<double>> sw(A, w);
        vec::dense_vector<double> xw(n, 0.0);
        for (int sweep = 0; sweep < max_sweeps; ++sweep) sw(xw, b);
        const double r = rel_residual(xw);
        sweep_resid.push_back(r);
        if (r < best_resid) { best_resid = r; best_omega = w; }
    }
    for (std::size_t k = 0; k < sweep_omegas.size(); k += 4)
        std::cout << "  omega = " << std::fixed << std::setprecision(2) << sweep_omegas[k]
                  << "  rel_resid = " << std::scientific << std::setprecision(3) << sweep_resid[k] << "\n";
    std::cout << "  smallest residual at omega = " << std::fixed << std::setprecision(4) << best_omega
              << " (theory: " << omega_opt << ")\n";

    // -- Why smoothers: what each sweep does to the error ----------------------
    // The discrete solution of -u'' = 1 is u_i = x_i (1 - x_i) / 2 exactly.
    // Start from it plus a smooth error (mode 1) and an oscillatory one
    // (mode 40 of 50), and watch the two modes' amplitudes sweep by sweep.
    std::cout << "\n--- What a sweep does to smooth and oscillatory error ---\n";
    const double pi = std::numbers::pi;
    const int lo_mode = 1, hi_mode = 40;
    vec::dense_vector<double> u_exact(n), x0(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double xi = (i + 1) * h;
        u_exact(i) = xi * (1.0 - xi) / 2.0;
        x0(i) = u_exact(i) + 0.02 * std::sin(lo_mode * pi * xi) + 0.02 * std::sin(hi_mode * pi * xi);
    }
    auto amplitude = [&](const vec::dense_vector<double>& e, int mode) {
        double a = 0.0;
        for (std::size_t i = 0; i < n; ++i) a += e(i) * std::sin(mode * pi * (i + 1) * h);
        return 2.0 * a / (n + 1);
    };
    std::cout << "  residual of the exact discrete solution: " << std::scientific << std::setprecision(1)
              << rel_residual(u_exact) << "\n";
    std::cout << std::setw(8) << "sweep" << std::setw(16) << "Jacobi mode 1" << std::setw(16) << "Jacobi mode 40"
              << std::setw(14) << "GS mode 1" << std::setw(14) << "GS mode 40\n";
    struct smoothing { std::vector<std::vector<double>> errors; std::vector<double> lo, hi; };
    smoothing sm_jac, sm_gs;
    vec::dense_vector<double> xj = x0, xg = x0;
    for (int sweep = 0; sweep <= max_sweeps; ++sweep) {
        vec::dense_vector<double> ej(xj - u_exact), eg(xg - u_exact);
        sm_jac.errors.push_back(example_trace::to_std(ej));
        sm_gs.errors.push_back(example_trace::to_std(eg));
        sm_jac.lo.push_back(amplitude(ej, lo_mode));
        sm_jac.hi.push_back(amplitude(ej, hi_mode));
        sm_gs.lo.push_back(amplitude(eg, lo_mode));
        sm_gs.hi.push_back(amplitude(eg, hi_mode));
        if (sweep == 0 || sweep == 1 || sweep == 2 || sweep == 5 || sweep == 10 || sweep == 30 || sweep == 100)
            std::cout << std::setw(8) << sweep << std::scientific << std::setprecision(2)
                      << std::setw(16) << sm_jac.lo.back() << std::setw(16) << sm_jac.hi.back()
                      << std::setw(14) << sm_gs.lo.back() << std::setw(14) << sm_gs.hi.back() << "\n";
        if (sweep < max_sweeps) {
            jac(xj, b);
            gs(xg, b);
        }
    }
    std::cout << "  The oscillatory mode is gone in a few sweeps; the smooth one barely moves.\n";
    std::cout << "  A coarser grid sees the smooth error as oscillatory: that is multigrid.\n";

    if (trace) {
        std::vector<double> xs(n);
        for (std::size_t i = 0; i < n; ++i) xs[i] = (i + 1) * h;
        trace["n"] = n;
        trace["sweeps"] = max_sweeps;
        trace["x"] = example_trace::value::numbers(xs, 6);
        auto hist = example_trace::value::array();
        auto add = [&](const std::string& name, double omega, const std::vector<double>& r) {
            auto o = example_trace::value::object();
            o["name"] = name;
            o["omega"] = example_trace::value::rounded(omega, 6);
            o["residual"] = example_trace::value::numbers(r, 5);
            hist.push_back(o);
        };
        add("Jacobi", 0.0, hist_jac);
        add("Gauss-Seidel", 1.0, hist_gs);
        for (std::size_t w = 0; w < omegas.size(); ++w)
            add("SOR", omegas[w], hist_sor[w]);
        trace["residuals"] = hist;
        trace["omega_opt"] = example_trace::value::rounded(omega_opt, 8);
        trace["omega_sweep"] = example_trace::value::numbers(sweep_omegas, 6);
        trace["omega_sweep_residual"] = example_trace::value::numbers(sweep_resid, 5);
        trace["best_omega"] = example_trace::value::rounded(best_omega, 6);
        auto& smooth = trace["smoothing"];
        smooth["modes"] = example_trace::value::numbers(std::vector<int>{lo_mode, hi_mode});
        for (auto [name, sm] : {std::pair<const char*, smoothing*>{"jacobi", &sm_jac}, {"gauss_seidel", &sm_gs}}) {
            auto o = example_trace::value::object();
            auto errs = example_trace::value::array();
            for (const auto& e : sm->errors) errs.push_back(example_trace::value::numbers(e, 3));
            o["errors"] = errs;
            o["low_mode"] = example_trace::value::numbers(sm->lo, 4);
            o["high_mode"] = example_trace::value::numbers(sm->hi, 4);
            smooth[name] = o;
        }
    }

    // -- Commentary -------------------------------------------------------
    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. Jacobi is the slowest: spectral radius rho(M_J) close to 1.\n";
    std::cout << "2. Gauss-Seidel ~ 2x faster: rho(M_GS) = rho(M_J)^2.\n";
    std::cout << "3. SOR with optimal omega is dramatically faster.\n";
    std::cout << "   For 1D Poisson: optimal omega = 2/(1+sin(pi*h)).\n";
    std::cout << "4. omega > 2 always diverges (spectral radius > 1).\n";
    std::cout << "5. These methods are too slow as standalone solvers\n";
    std::cout << "   (convergence rate depends on mesh size h), but they\n";
    std::cout << "   effectively smooth high-frequency error components,\n";
    std::cout << "   making them ideal as smoothers in multigrid methods.\n";

    return trace.close();   // EXIT_FAILURE if a requested trace could not be written
}
