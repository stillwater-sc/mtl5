// pca_svd.cpp - PCA via SVD: Data Dimensionality Reduction
//
// This example demonstrates:
//   1. Principal Component Analysis (PCA) as an SVD problem
//   2. How SVD decomposes data into orthogonal components
//   3. The connection between SVD singular values and covariance eigenvalues
//   4. Dimensionality reduction via truncated reconstruction
//   5. Variance explained by each principal component
//
// We create a synthetic 8x3 dataset with known correlation structure,
// perform PCA via SVD, and reconstruct from fewer components.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mtl;

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 6B: PCA via SVD - Data Dimensionality Reduction\n";
    std::cout << "=============================================================\n\n";

    // -- Generate synthetic data ------------------------------------------
    // 8 samples, 3 features. Feature 3 ~= Feature 1 + noise (correlated).
    const std::size_t m = 8;  // samples
    const std::size_t p = 3;  // features

    std::cout << "Dataset: " << m << " samples x " << p << " features\n";
    std::cout << "Feature 3 is designed to correlate with Feature 1.\n\n";

    mat::dense2D<double> X(m, p);
    // Feature 1: roughly [1,2,3,...,8]
    // Feature 2: roughly independent
    // Feature 3: Feature 1 + small noise (correlated)
    double raw_data[8][3] = {
        {1.0, 4.2, 1.1},
        {2.1, 3.8, 2.3},
        {3.0, 4.5, 2.8},
        {4.2, 3.1, 4.4},
        {5.1, 4.8, 5.0},
        {5.8, 3.3, 6.1},
        {7.0, 4.1, 6.8},
        {8.1, 3.6, 8.3}
    };

    std::cout << "Raw data X:\n";
    for (std::size_t i = 0; i < m; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < p; ++j) {
            X(i, j) = raw_data[i][j];
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(1) << X(i, j);
        }
        std::cout << "]\n";
    }

    // -- Step 1: Center the data ------------------------------------------
    std::cout << "\n--- Step 1: Center the data (subtract column means) ---\n";

    vec::dense_vector<double> col_means(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < m; ++i)
            col_means(j) += X(i, j);
        col_means(j) /= m;
    }

    std::cout << "Column means: [";
    for (std::size_t j = 0; j < p; ++j) {
        if (j > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3) << col_means(j);
    }
    std::cout << "]\n";

    mat::dense2D<double> Xc(m, p);  // centered data
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < p; ++j)
            Xc(i, j) = X(i, j) - col_means(j);

    // -- Step 2: SVD of centered data -------------------------------------
    std::cout << "\n--- Step 2: SVD of centered data ---\n";
    std::cout << "X_centered = U * S * V^T\n\n";

    mat::dense2D<double> U, S, V;
    svd(Xc, U, S, V);

    // Print singular values
    std::size_t rank = std::min(m, p);
    std::cout << "Singular values:\n";
    double total_var = 0.0;
    std::vector<double> svals;
    for (std::size_t k = 0; k < rank; ++k) {
        svals.push_back(S(k, k));
        total_var += S(k, k) * S(k, k);
    }

    for (std::size_t k = 0; k < rank; ++k) {
        double var_pct = 100.0 * svals[k] * svals[k] / total_var;
        std::cout << "  sigma_" << k+1 << " = " << std::fixed << std::setprecision(4)
                  << svals[k] << "  (variance explained: "
                  << std::setprecision(1) << var_pct << "%)\n";
    }

    // Print V (principal component directions)
    std::cout << "\nPrincipal components V (columns = directions):\n";
    for (std::size_t i = 0; i < p; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < rank; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << V(i, j);
        }
        std::cout << "]\n";
    }

    // -- Step 3: Compare with covariance eigendecomposition ---------------
    std::cout << "\n--- Step 3: Verify against covariance eigendecomposition ---\n";
    std::cout << "Covariance C = X^T * X / (n-1)\n";
    std::cout << "Eigenvalues of C should equal sigma_i^2 / (n-1)\n\n";

    // Compute covariance matrix
    auto XtX = trans(Xc) * Xc;
    mat::dense2D<double> C(p, p);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            C(i, j) = XtX(i, j) / (m - 1);

    auto cov_eigs = eigenvalue_symmetric(C);

    std::cout << std::setw(6) << "k"
              << std::setw(18) << "sigma^2/(n-1)"
              << std::setw(18) << "cov eigenvalue"
              << std::setw(14) << "Error" << "\n";
    std::cout << std::string(56, '-') << "\n";

    // Eigenvalues are sorted ascending, singular values descending
    for (std::size_t k = 0; k < rank; ++k) {
        double svd_var = svals[rank - 1 - k] * svals[rank - 1 - k] / (m - 1);
        double cov_var = cov_eigs(k);
        double err = std::abs(svd_var - cov_var);
        std::cout << std::setw(6) << k + 1
                  << std::fixed << std::setprecision(6)
                  << std::setw(18) << svd_var
                  << std::setw(18) << cov_var
                  << std::scientific << std::setw(14) << err << "\n";
    }

    // -- Step 4: Dimensionality reduction ---------------------------------
    std::cout << "\n--- Step 4: Reduce from 3D to 2D ---\n";
    std::cout << "Project onto first 2 principal components.\n\n";

    // Project: X_reduced = Xc * V[:, :2]  (m x 2)
    mat::dense2D<double> V2(p, 2);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            V2(i, j) = V(i, j);

    auto X_reduced = Xc * V2;

    std::cout << "Projected data (2D):\n";
    for (std::size_t i = 0; i < m; ++i) {
        std::cout << "  sample " << i << ": ("
                  << std::fixed << std::setprecision(3) << X_reduced(i, 0)
                  << ", " << X_reduced(i, 1) << ")\n";
    }

    // -- Step 5: Reconstruct and measure error ----------------------------
    std::cout << "\n--- Step 5: Reconstruct from 2 components ---\n";

    // Reconstruct: X_approx = X_reduced * V2^T + means
    auto X_recon_centered = X_reduced * trans(V2);
    mat::dense2D<double> X_approx(m, p);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < p; ++j)
            X_approx(i, j) = X_recon_centered(i, j) + col_means(j);

    // Compute reconstruction error
    double recon_err = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < p; ++j)
            recon_err += (X(i, j) - X_approx(i, j)) * (X(i, j) - X_approx(i, j));
    recon_err = std::sqrt(recon_err);

    double total_norm = frobenius_norm(Xc);
    double rel_err = recon_err / total_norm;

    std::cout << "Reconstruction error ||X - X_approx||_F = "
              << std::scientific << recon_err << "\n";
    std::cout << "Relative error: " << std::fixed << std::setprecision(2)
              << 100.0 * rel_err << "%\n";
    std::cout << "Variance captured: "
              << std::setprecision(1)
              << 100.0 * (svals[0]*svals[0] + svals[1]*svals[1]) / total_var << "%\n\n";

    std::cout << "Sample reconstruction comparison:\n";
    std::cout << std::setw(8) << "Sample"
              << std::setw(24) << "Original"
              << std::setw(24) << "Reconstructed" << "\n";
    std::cout << std::string(56, '-') << "\n";
    for (std::size_t i = 0; i < m; i += 2) {
        std::cout << std::setw(8) << i << "  (";
        for (std::size_t j = 0; j < p; ++j) {
            if (j > 0) std::cout << ",";
            std::cout << std::fixed << std::setprecision(1) << std::setw(5) << X(i, j);
        }
        std::cout << ")  (";
        for (std::size_t j = 0; j < p; ++j) {
            if (j > 0) std::cout << ",";
            std::cout << std::fixed << std::setprecision(1) << std::setw(5) << X_approx(i, j);
        }
        std::cout << ")\n";
    }

    // -- Commentary -------------------------------------------------------
    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. PCA finds orthogonal directions of maximum variance.\n";
    std::cout << "   The k-th principal component captures the k-th most\n";
    std::cout << "   variance in the data.\n";
    std::cout << "2. SVD is the numerically stable way to compute PCA.\n";
    std::cout << "   Forming X^T*X squares the condition number - avoid it.\n";
    std::cout << "3. Singular values relate to covariance eigenvalues:\n";
    std::cout << "   variance_k = sigma_k^2 / (n-1)\n";
    std::cout << "4. Truncating to k components gives the best rank-k\n";
    std::cout << "   approximation (Eckart-Young theorem).\n";

    return EXIT_SUCCESS;
}
