// vibrating_string.cpp - Vibrating String Eigenvalue Problem
//
// This example demonstrates:
//   1. Eigenvalues of the 1D Laplacian = vibration frequencies
//   2. Analytical vs numerical eigenvalues comparison
//   3. Kronecker product to build 2D Laplacian from 1D
//   4. 2D eigenvalues = sums of 1D eigenvalue pairs
//   5. General eigenvalue solver for non-symmetric matrices
//   6. SVD singular values and their relation to eigenvalues
//
// Physics: the eigenvalues of the discrete Laplacian correspond
// to standing wave frequencies of a vibrating string.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <complex>
#include <algorithm>
#include <vector>

#include "../common/replay.hpp"
#include "../common/trace.hpp"

using namespace mtl;

/// The 1D Laplacian made non-symmetric the way convection does it (example
/// 3B): subdiagonal -(1 + d), diagonal 2, superdiagonal -(1 - d).
mtl::mat::dense2D<double> skewed_laplacian(std::size_t n, double d) {
    mtl::mat::dense2D<double> B(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        B(i, i) = 2.0;
        if (i > 0)     B(i, i - 1) = -(1.0 + d);
        if (i + 1 < n) B(i, i + 1) = -(1.0 - d);
    }
    return B;
}

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "6A vibrating string");

    std::cout << "=============================================================\n";
    std::cout << " Phase 6A: Vibrating String - Eigenvalues + Kronecker\n";
    std::cout << "=============================================================\n\n";

    const double pi = std::numbers::pi;

    // ======================================================================
    // Part 1: 1D Laplacian Eigenvalues
    // ======================================================================
    std::cout << "=== Part 1: 1D Laplacian Eigenvalues ===\n\n";

    const std::size_t n = 8;
    std::cout << "Building " << n << "x" << n << " tridiagonal 1D Laplacian:\n";
    std::cout << "T(i,i) = 2, T(i,i+/-1) = -1\n\n";

    // Build 1D Laplacian (no h^2 scaling - pure combinatorial Laplacian)
    mat::dense2D<double> T(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        T(i, i) = 2.0;
        if (i > 0)     T(i, i-1) = -1.0;
        if (i + 1 < n) T(i, i+1) = -1.0;
    }

    auto eigs_1d = eigenvalue_symmetric(T);

    std::cout << std::setw(6) << "k"
              << std::setw(16) << "Computed"
              << std::setw(16) << "Analytical"
              << std::setw(18) << "Error" << "\n";
    std::cout << std::string(56, '-') << "\n";

    // Analytical: lambda_k = 4 * sin^2(k*pi / (2*(n+1))), k = 1..n
    for (std::size_t k = 0; k < n; ++k) {
        double analytical = 4.0 * std::pow(std::sin((k + 1) * pi / (2.0 * (n + 1))), 2);
        double computed = eigs_1d(k);
        double error = std::abs(computed - analytical);
        std::cout << std::setw(6) << k + 1
                  << std::fixed << std::setprecision(10)
                  << std::setw(16) << computed
                  << std::setw(16) << analytical
                  << std::scientific << std::setw(18) << error << "\n";
    }

    std::cout << "\nPhysical interpretation: eigenvalue k corresponds to the k-th\n";
    std::cout << "harmonic of a vibrating string. Frequency ~ sqrt(lambda_k).\n\n";

    // ======================================================================
    // Part 2: 2D Laplacian via Kronecker Product
    // ======================================================================
    std::cout << "=== Part 2: 2D Laplacian via Kronecker Product ===\n\n";

    const std::size_t n2d = 4;  // small for manageable output
    const std::size_t N = n2d * n2d;  // total unknowns

    // Build 1D Laplacian for 2D construction
    mat::dense2D<double> T2(n2d, n2d);
    for (std::size_t i = 0; i < n2d; ++i) {
        T2(i, i) = 2.0;
        if (i > 0)      T2(i, i-1) = -1.0;
        if (i + 1 < n2d) T2(i, i+1) = -1.0;
    }

    // Identity matrix
    mat::identity2D<double> I(n2d);

    std::cout << "2D Laplacian = kron(I, T) + kron(T, I)\n";
    std::cout << "Size: " << N << " x " << N << "\n\n";

    // L2D = kron(I, T) + kron(T, I)
    auto kI_T = kron(I, T2);  // operates on x-direction
    auto kT_I = kron(T2, I);  // operates on y-direction
    mat::dense2D<double> L2D(N, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            L2D(i, j) = kI_T(i, j) + kT_I(i, j);

    auto eigs_2d = eigenvalue_symmetric(L2D);

    // 1D eigenvalues for n2d
    auto eigs_1d_small = eigenvalue_symmetric(T2);

    // Verify: 2D eigenvalues = lambda_i + lambda_j for all pairs
    std::cout << "Verification: 2D eigenvalues = lambda_i + lambda_j\n\n";
    std::cout << std::setw(6) << "k"
              << std::setw(14) << "Computed 2D"
              << std::setw(12) << "(i,j)"
              << std::setw(14) << "l_i + l_j"
              << std::setw(14) << "Error" << "\n";
    std::cout << std::string(60, '-') << "\n";

    // Build sorted list of all lambda_i + lambda_j pairs
    std::vector<std::pair<double, std::pair<int,int>>> pairs;
    for (std::size_t i = 0; i < n2d; ++i)
        for (std::size_t j = 0; j < n2d; ++j)
            pairs.push_back({eigs_1d_small(i) + eigs_1d_small(j), {(int)i+1, (int)j+1}});
    std::sort(pairs.begin(), pairs.end());

    for (std::size_t k = 0; k < N; ++k) {
        double err = std::abs(eigs_2d(k) - pairs[k].first);
        std::cout << std::setw(6) << k + 1
                  << std::fixed << std::setprecision(6)
                  << std::setw(14) << eigs_2d(k)
                  << "    (" << pairs[k].second.first << "," << pairs[k].second.second << ")"
                  << std::setw(14) << pairs[k].first
                  << std::scientific << std::setw(14) << err << "\n";
    }

    // ======================================================================
    // Part 3: General Eigenvalues (Non-Symmetric Matrix)
    // ======================================================================
    std::cout << "\n=== Part 3: Non-Symmetric Matrix -> Complex Eigenvalues ===\n\n";

    mat::dense2D<double> B(n2d, n2d);
    for (std::size_t i = 0; i < n2d; ++i)
        for (std::size_t j = 0; j < n2d; ++j)
            B(i, j) = T2(i, j);
    // Add asymmetric perturbation
    B(0, n2d - 1) = 0.5;

    std::cout << "Perturbed matrix (T with B(0," << n2d-1 << ") = 0.5):\n";
    auto eigs_gen = eigenvalue(B);

    std::cout << std::setw(6) << "k"
              << std::setw(16) << "Real"
              << std::setw(16) << "Imag" << "\n";
    std::cout << std::string(38, '-') << "\n";
    for (std::size_t k = 0; k < n2d; ++k) {
        std::cout << std::setw(6) << k + 1
                  << std::fixed << std::setprecision(8)
                  << std::setw(16) << eigs_gen(k).real()
                  << std::setw(16) << eigs_gen(k).imag() << "\n";
    }

    std::cout << "\nOne asymmetric entry is not enough: these eigenvalues are still real.\n";
    std::cout << "Make the whole operator lopsided instead, as convection does (example 3B):\n";
    std::cout << "subdiagonal -(1+d), superdiagonal -(1-d). The eigenvalues are\n";
    std::cout << "2 - 2 sqrt(1 - d^2) cos(k pi / (n+1)): real for d < 1, complex for d > 1.\n\n";
    std::cout << std::setw(8) << "d" << std::setw(18) << "max |Im lambda|" << std::setw(18)
              << "theory" << "\n";
    std::vector<double> deltas;
    for (int k = 0; k <= 15; ++k) deltas.push_back(0.1 * k);
    std::vector<std::vector<std::complex<double>>> swept;
    for (double d : deltas) {
        auto ev = eigenvalue(skewed_laplacian(n, d));
        std::vector<std::complex<double>> v(ev.size());
        double im = 0.0;
        for (std::size_t k = 0; k < ev.size(); ++k) {
            v[k] = ev(k);
            im = std::max(im, std::abs(ev(k).imag()));
        }
        swept.push_back(v);
        const double theory = d > 1.0 ? 2.0 * std::sqrt(d * d - 1.0) * std::cos(pi / (n + 1)) : 0.0;
        std::cout << std::fixed << std::setprecision(2) << std::setw(8) << d << std::scientific
                  << std::setprecision(3) << std::setw(18) << im << std::setw(18) << theory << "\n";
    }
    std::cout << "At d = 1 the superdiagonal is zero: the matrix is triangular, and all\n";
    std::cout << "eight eigenvalues are exactly 2, where the real ones meet before they split.\n";

    // ======================================================================
    // Part 4: SVD and Eigenvalue Connection
    // ======================================================================
    std::cout << "\n=== Part 4: SVD Singular Values vs Eigenvalues ===\n\n";

    std::cout << "For symmetric A: singular values = |eigenvalues|\n\n";

    mat::dense2D<double> U, S, V;
    svd(T2, U, S, V);

    // Collect and sort singular values
    std::vector<double> svals;
    for (std::size_t i = 0; i < n2d; ++i)
        svals.push_back(S(i, i));
    std::sort(svals.begin(), svals.end());

    std::cout << std::setw(6) << "k"
              << std::setw(16) << "|eigenvalue|"
              << std::setw(16) << "sing. value"
              << std::setw(16) << "Error" << "\n";
    std::cout << std::string(54, '-') << "\n";

    for (std::size_t k = 0; k < n2d; ++k) {
        double abs_eig = std::abs(eigs_1d_small(k));
        double err = std::abs(abs_eig - svals[k]);
        std::cout << std::setw(6) << k + 1
                  << std::fixed << std::setprecision(8)
                  << std::setw(16) << abs_eig
                  << std::setw(16) << svals[k]
                  << std::scientific << std::setw(16) << err << "\n";
    }

    // -- Commentary -------------------------------------------------------
    if (trace) {
        trace["n"] = n;
        // The string: T's eigenvalues and eigenvectors (column k = mode k).
        auto [lam, Q] = eigen_symmetric(T);
        auto modes = example_trace::value::array();
        std::vector<double> analytic(n);
        for (std::size_t k = 0; k < n; ++k) {
            std::vector<double> v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = Q(i, k);
            if (v[0] < 0) for (double& x : v) x = -x;   // a sign convention for drawing
            modes.push_back(example_trace::value::numbers(v, 6));
            analytic[k] = 4.0 * std::pow(std::sin((k + 1) * pi / (2.0 * (n + 1))), 2);
        }
        trace["eigenvalues"] = example_trace::value::numbers(example_trace::to_std(lam), 12);
        trace["analytic"] = example_trace::value::numbers(analytic, 12);
        trace["modes"] = modes;
        // The membrane: the Kronecker Laplacian's eigenpairs on the n2d x n2d grid.
        auto [lam2, Q2] = eigen_symmetric(L2D);
        auto modes2 = example_trace::value::array();
        for (std::size_t k = 0; k < N; ++k) {
            std::vector<double> v(N);
            for (std::size_t i = 0; i < N; ++i) v[i] = Q2(i, k);
            modes2.push_back(example_trace::value::numbers(v, 5));
        }
        trace["n2d"] = n2d;
        trace["membrane_eigenvalues"] = example_trace::value::numbers(example_trace::to_std(lam2), 10);
        trace["membrane_modes"] = modes2;
        // The lopsided operator's eigenvalues across d.
        auto sweep = example_trace::value::array();
        for (std::size_t q = 0; q < deltas.size(); ++q) {
            auto o = example_trace::value::object();
            o["d"] = example_trace::value::rounded(deltas[q], 6);
            std::vector<double> re, im;
            for (auto z : swept[q]) {
                re.push_back(z.real());
                im.push_back(z.imag());
            }
            o["re"] = example_trace::value::numbers(re, 8);
            o["im"] = example_trace::value::numbers(im, 8);
            sweep.push_back(o);
        }
        trace["skew_sweep"] = sweep;
    }

    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. The 1D Laplacian eigenvalues have closed-form expressions.\n";
    std::cout << "   They correspond to vibration modes of a discretized string.\n";
    std::cout << "2. Kronecker product builds higher-dimensional operators from 1D.\n";
    std::cout << "   This 'tensor product' structure is key to spectral methods.\n";
    std::cout << "3. Non-symmetric matrices can have complex eigenvalue pairs, as the\n";
    std::cout << "   lopsided Laplacian does once d > 1; one asymmetric entry need not.\n";
    std::cout << "   Use eigenvalue() (general QR) instead of eigenvalue_symmetric().\n";
    std::cout << "4. For symmetric matrices: singular values = |eigenvalues|.\n";
    std::cout << "   This connection is fundamental in numerical linear algebra.\n";

    return trace.close();   // EXIT_FAILURE if a requested trace could not be written
}
