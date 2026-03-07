// phase5a_least_squares_qr.cpp - Least Squares Curve Fitting
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

using namespace mtl;

int main() {
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

    std::vector<std::size_t> degrees = {3, 5, 8, 12};

    for (auto p : degrees) {
        std::size_t ncols = p + 1;  // number of coefficients
        if (ncols > m) break;

        std::cout << "======= Degree " << p << " (Vandermonde " << m << "x" << ncols << ") =======\n";

        // Build Vandermonde matrix: V(i,j) = x_i^j
        mat::dense2D<double> V(m, ncols);
        for (std::size_t i = 0; i < m; ++i) {
            double xi_pow = 1.0;
            for (std::size_t j = 0; j < ncols; ++j) {
                V(i, j) = xi_pow;
                xi_pow *= x_data(i);
            }
        }

        // -- Method 1: Normal Equations (V^T * V * c = V^T * y) ----------
        // This squares the condition number: cond(V^T*V) = cond(V)^2
        auto VtV = trans(V) * V;   // ncols x ncols SPD
        auto Vty = trans(V) * y_data;  // ncols x 1

        // Try Cholesky on normal equations
        mat::dense2D<double> VtV_copy(ncols, ncols);
        for (std::size_t i = 0; i < ncols; ++i)
            for (std::size_t j = 0; j < ncols; ++j)
                VtV_copy(i, j) = VtV(i, j);

        vec::dense_vector<double> c_normal(ncols, 0.0);
        int chol_info = cholesky_factor(VtV_copy);

        if (chol_info == 0) {
            cholesky_solve(VtV_copy, c_normal, Vty);
            // Compute residual
            auto Vc = V * c_normal;
            double res_normal = 0.0;
            for (std::size_t i = 0; i < m; ++i)
                res_normal += (y_data(i) - Vc(i)) * (y_data(i) - Vc(i));
            res_normal = std::sqrt(res_normal);
            std::cout << "  Normal eqns (Cholesky): residual = "
                      << std::scientific << res_normal << "\n";
        } else {
            std::cout << "  Normal eqns (Cholesky): FAILED (not SPD, step "
                      << chol_info << ")\n";
        }

        // -- Method 2: QR Factorization ----------------------------------
        // Numerically stable: works directly with V, no squaring
        mat::dense2D<double> V_qr(m, ncols);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < ncols; ++j)
                V_qr(i, j) = V(i, j);

        vec::dense_vector<double> tau;
        qr_factor(V_qr, tau);

        vec::dense_vector<double> c_qr(ncols, 0.0);
        qr_solve(V_qr, tau, c_qr, y_data);

        // Compute residual
        auto Vc_qr = V * c_qr;
        double res_qr = 0.0;
        for (std::size_t i = 0; i < m; ++i)
            res_qr += (y_data(i) - Vc_qr(i)) * (y_data(i) - Vc_qr(i));
        res_qr = std::sqrt(res_qr);

        std::cout << "  QR factorization:       residual = "
                  << std::scientific << res_qr << "\n";

        // -- Show condition number of V^T*V ------------------------------
        // Approximate via ratio of max/min eigenvalue
        auto VtV_fresh = trans(V) * V;
        mat::dense2D<double> VtV_eig(ncols, ncols);
        for (std::size_t i = 0; i < ncols; ++i)
            for (std::size_t j = 0; j < ncols; ++j)
                VtV_eig(i, j) = VtV_fresh(i, j);

        auto eigs = eigenvalue_symmetric(VtV_eig);
        double lambda_min = std::abs(eigs(0));
        double lambda_max = std::abs(eigs(ncols - 1));
        double cond_VtV = (lambda_min > 1e-300) ? lambda_max / lambda_min : 1e300;

        std::cout << "  cond(V^T*V) = " << std::scientific << cond_VtV << "\n";
        std::cout << "  cond(V)     ~ " << std::scientific << std::sqrt(cond_VtV) << "\n";

        // Show coefficients
        std::cout << "  QR coefficients: [";
        for (std::size_t j = 0; j < std::min(ncols, std::size_t(5)); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << c_qr(j);
        }
        if (ncols > 5) std::cout << ", ...";
        std::cout << "]\n\n";
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

    return EXIT_SUCCESS;
}
