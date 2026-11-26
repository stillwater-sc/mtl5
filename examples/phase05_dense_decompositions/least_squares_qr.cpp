// least_squares_qr.cpp - Least Squares Curve Fitting
//
// This example demonstrates:
//   1. Polynomial curve fitting as a least-squares problem
//   2. Why the normal equations (V^T*V) become ill-conditioned
//   3. Why QR factorization is numerically superior
//   4. How condition number affects solution accuracy
//
// We fit y = sin(x) with polynomial basis on [0, pi] and compare
// QR vs normal equations at increasing polynomial degrees.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <vector>

#include "../common/replay.hpp"
#include "../common/trace.hpp"

using namespace mtl;

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "5A least squares via QR");

    std::cout << "=============================================================\n";
    std::cout << " Phase 5A: Least Squares Curve Fitting - QR vs Normal Eqns\n";
    std::cout << "=============================================================\n\n";

    // -- Generate noisy data ----------------------------------------------
    const std::size_t m = 20;  // data points
    const double pi = std::numbers::pi;

    // Data: y_i = sin(x_i) + noise (using deterministic "noise")
    std::cout << "Data: " << m << " points from y = sin(x) on [0, pi]\n\n";

    vec::dense_vector<double> x_data(m), y_data(m);
    for (std::size_t i = 0; i < m; ++i) {
        x_data(i) = pi * i / (m - 1);
        // Small deterministic perturbation simulating noise
        double noise = 0.02 * std::sin(7.0 * i + 3.0);
        y_data(i) = std::sin(x_data(i)) + noise;
    }

    // -- Try polynomial fits at increasing degree -------------------------
    std::cout << "--- Fitting polynomials of degree p: y = c0 + c1*x + ... + cp*x^p ---\n\n";

    // One fit of degree p by both methods. The printed degrees are 3, 5, 8 and
    // 12; the trace (--trace) records every degree from 1 to m - 2.
    struct fit {
        std::size_t p;
        int chol_info;
        vec::dense_vector<double> c_normal, c_qr;
        double res_normal, res_qr, cond_VtV_eig, cond_V_svd;
    };
    auto fit_degree = [&](std::size_t p) {
        const std::size_t ncols = p + 1;  // number of coefficients
        fit f{p, 0, vec::dense_vector<double>(ncols, 0.0), vec::dense_vector<double>(ncols, 0.0), 0.0, 0.0, 0.0, 0.0};

        // Build Vandermonde matrix: V(i,j) = x_i^j
        mat::dense2D<double> V(m, ncols);
        for (std::size_t i = 0; i < m; ++i) {
            double xi_pow = 1.0;
            for (std::size_t j = 0; j < ncols; ++j) {
                V(i, j) = xi_pow;
                xi_pow *= x_data(i);
            }
        }
        auto residual = [&](const vec::dense_vector<double>& c) {
            auto Vc = V * c;
            double r = 0.0;
            for (std::size_t i = 0; i < m; ++i) r += (y_data(i) - Vc(i)) * (y_data(i) - Vc(i));
            return std::sqrt(r);
        };

        // -- Method 1: Normal Equations (V^T * V * c = V^T * y) ----------
        // This squares the condition number: cond(V^T*V) = cond(V)^2
        auto VtV = trans(V) * V;   // ncols x ncols SPD
        auto Vty = trans(V) * y_data;  // ncols x 1
        mat::dense2D<double> VtV_copy(ncols, ncols);
        for (std::size_t i = 0; i < ncols; ++i)
            for (std::size_t j = 0; j < ncols; ++j)
                VtV_copy(i, j) = VtV(i, j);
        f.chol_info = cholesky_factor(VtV_copy);
        if (f.chol_info == 0) {
            cholesky_solve(VtV_copy, f.c_normal, Vty);
            f.res_normal = residual(f.c_normal);
        }

        // -- Method 2: QR Factorization ----------------------------------
        // Numerically stable: works directly with V, no squaring
        mat::dense2D<double> V_qr(m, ncols);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < ncols; ++j)
                V_qr(i, j) = V(i, j);
        vec::dense_vector<double> tau;
        qr_factor(V_qr, tau);
        qr_solve(V_qr, tau, f.c_qr, y_data);
        f.res_qr = residual(f.c_qr);

        // -- Condition numbers -------------------------------------------
        // cond(V^T*V) from the eigenvalues of the computed V^T*V: once it
        // passes about 1e16 its smallest eigenvalue is rounding noise.
        auto VtV_fresh = trans(V) * V;
        mat::dense2D<double> VtV_eig(ncols, ncols);
        for (std::size_t i = 0; i < ncols; ++i)
            for (std::size_t j = 0; j < ncols; ++j)
                VtV_eig(i, j) = VtV_fresh(i, j);
        auto eigs = eigenvalue_symmetric(VtV_eig);
        double lambda_min = std::abs(eigs(0));
        double lambda_max = std::abs(eigs(ncols - 1));
        f.cond_VtV_eig = (lambda_min > 1e-300) ? lambda_max / lambda_min : 1e300;
        // cond(V) from V's singular values, which one-sided Jacobi computes
        // accurately well past the point where V^T*V has lost them.
        mat::dense2D<double> U, S, W;
        svd(V, U, S, W);
        double smax = 0.0, smin = 1e300;
        for (std::size_t j = 0; j < ncols; ++j) {
            smax = std::max(smax, S(j, j));
            smin = std::min(smin, S(j, j));
        }
        f.cond_V_svd = smax / smin;
        return f;
    };

    std::vector<std::size_t> degrees = {3, 5, 8, 12};

    for (auto p : degrees) {
        std::size_t ncols = p + 1;  // number of coefficients
        if (ncols > m) break;

        std::cout << "======= Degree " << p << " (Vandermonde " << m << "x" << ncols << ") =======\n";
        const fit f = fit_degree(p);
        if (f.chol_info == 0) {
            std::cout << "  Normal eqns (Cholesky): residual = "
                      << std::scientific << f.res_normal << "\n";
        } else {
            std::cout << "  Normal eqns (Cholesky): FAILED (not SPD, step "
                      << f.chol_info << ")\n";
        }
        std::cout << "  QR factorization:       residual = "
                  << std::scientific << f.res_qr << "\n";
        std::cout << "  cond(V^T*V) = " << std::scientific << f.cond_VtV_eig << "\n";
        std::cout << "  cond(V)     ~ " << std::scientific << std::sqrt(f.cond_VtV_eig) << "\n";

        // Show coefficients
        std::cout << "  QR coefficients: [";
        for (std::size_t j = 0; j < std::min(ncols, std::size_t(5)); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << f.c_qr(j);
        }
        if (ncols > 5) std::cout << ", ...";
        std::cout << "]\n\n";
    }

    // -- Trace for the docs-site animation (only with --trace <file>) -----
    if (trace) {
        trace["m"] = m;
        trace["x_data"] = example_trace::value::numbers(example_trace::to_std(x_data), 8);
        trace["y_data"] = example_trace::value::numbers(example_trace::to_std(y_data), 8);
        std::vector<double> xf;
        for (int i = 0; i <= 120; ++i) xf.push_back(pi * i / 120.0);
        trace["x_fine"] = example_trace::value::numbers(xf, 6);
        auto eval = [&](const vec::dense_vector<double>& c) {
            std::vector<double> yv;
            for (double xv : xf) {
                double y = 0.0;
                for (std::size_t j = c.size(); j-- > 0;) y = y * xv + c(j);   // Horner
                yv.push_back(y);
            }
            return yv;
        };
        auto fits = example_trace::value::array();
        for (std::size_t p = 1; p + 2 <= m; ++p) {
            const fit f = fit_degree(p);
            auto o = example_trace::value::object();
            o["degree"] = p;
            o["cholesky_info"] = f.chol_info;
            o["residual_normal"] = f.chol_info == 0 ? example_trace::value::rounded(f.res_normal, 6) : example_trace::value();
            o["residual_qr"] = example_trace::value::rounded(f.res_qr, 6);
            o["cond_VtV_eig"] = example_trace::value::rounded(f.cond_VtV_eig, 4);
            o["cond_V_svd"] = example_trace::value::rounded(f.cond_V_svd, 4);
            o["curve_normal"] = f.chol_info == 0 ? example_trace::value::numbers(eval(f.c_normal), 5) : example_trace::value();
            o["curve_qr"] = example_trace::value::numbers(eval(f.c_qr), 5);
            fits.push_back(o);
        }
        trace["fits"] = fits;
    }

    // -- Commentary -------------------------------------------------------
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. The Vandermonde matrix V has condition number that grows\n";
    std::cout << "   exponentially with polynomial degree. The normal equations\n";
    std::cout << "   square this: cond(V^T*V) = cond(V)^2.\n";
    std::cout << "2. QR factorization works directly on V, avoiding the squaring.\n";
    std::cout << "   It gives the same least-squares solution with much better\n";
    std::cout << "   numerical stability.\n";
    std::cout << "3. Rule of thumb (Trefethen & Bau):\n";
    std::cout << "   - If cond(A) ~ 10^k, expect to lose k digits of accuracy.\n";
    std::cout << "   - Normal equations: lose 2k digits (squared condition).\n";
    std::cout << "   - QR: lose only k digits.\n";
    std::cout << "4. In practice, prefer QR for least-squares. Only use normal\n";
    std::cout << "   equations when A is well-conditioned AND speed is critical.\n";

    return trace.close();   // EXIT_FAILURE if a requested trace could not be written
}
