// phase6a_vibrating_string.cpp - Vibrating String Eigenvalue Problem
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

using namespace mtl;

int main() {
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
              << std::setw(16) << "Error" << "\n";
    std::cout << std::string(54, '-') << "\n";

    // Analytical: lambda_k = 4 * sin^2(k*pi / (2*(n+1))), k = 1..n
    for (std::size_t k = 0; k < n; ++k) {
        double analytical = 4.0 * std::pow(std::sin((k + 1) * pi / (2.0 * (n + 1))), 2);
        double computed = eigs_1d(k);
        double error = std::abs(computed - analytical);
        std::cout << std::setw(6) << k + 1
                  << std::fixed << std::setprecision(10)
                  << std::setw(16) << computed
                  << std::setw(16) << analytical
                  << std::scientific << std::setw(16) << error << "\n";
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
              << std::setw(14) << "Error" << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (std::size_t k = 0; k < n2d; ++k) {
        double abs_eig = std::abs(eigs_1d_small(k));
        double err = std::abs(abs_eig - svals[k]);
        std::cout << std::setw(6) << k + 1
                  << std::fixed << std::setprecision(8)
                  << std::setw(16) << abs_eig
                  << std::setw(16) << svals[k]
                  << std::scientific << std::setw(14) << err << "\n";
    }

    // -- Commentary -------------------------------------------------------
    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. The 1D Laplacian eigenvalues have closed-form expressions.\n";
    std::cout << "   They correspond to vibration modes of a discretized string.\n";
    std::cout << "2. Kronecker product builds higher-dimensional operators from 1D.\n";
    std::cout << "   This 'tensor product' structure is key to spectral methods.\n";
    std::cout << "3. Non-symmetric perturbations create complex eigenvalue pairs.\n";
    std::cout << "   Use eigenvalue() (general QR) instead of eigenvalue_symmetric().\n";
    std::cout << "4. For symmetric matrices: singular values = |eigenvalues|.\n";
    std::cout << "   This connection is fundamental in numerical linear algebra.\n";

    return EXIT_SUCCESS;
}
