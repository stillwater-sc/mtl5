// solve_three_ways.cpp - Solving Ax = b: LU vs Cholesky vs QR
//
// This example demonstrates:
//   1. When to use each direct factorization method
//   2. Cholesky is 2x faster but requires SPD matrices
//   3. LU with pivoting handles any non-singular system
//   4. QR is the most numerically stable
//   5. Why you should never compute inv(A)*b in practice
//
// We solve the same system three ways, then show what happens
// when the matrix is not SPD (Cholesky correctly fails).

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mtl;

void print_vector(const std::string& name, const vec::dense_vector<double>& v) {
    std::cout << "  " << name << " = [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(8) << v(i);
    }
    std::cout << "]\n";
}

double residual_norm(const mat::dense2D<double>& A,
                     const vec::dense_vector<double>& x,
                     const vec::dense_vector<double>& b) {
    auto Ax = A * x;
    vec::dense_vector<double> r(b.size());
    for (std::size_t i = 0; i < b.size(); ++i)
        r(i) = b(i) - Ax(i);
    return two_norm(r);
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 5B: Solving Ax=b - LU vs Cholesky vs QR\n";
    std::cout << "=============================================================\n\n";

    const std::size_t n = 4;

    // ======================================================================
    // Part 1: SPD System
    // ======================================================================
    std::cout << "=== Part 1: SPD (Symmetric Positive Definite) System ===\n\n";

    // Build a well-conditioned SPD matrix (Hilbert-like but better conditioned)
    mat::dense2D<double> A_spd(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A_spd(i, j) = 1.0 / (1.0 + std::abs(static_cast<double>(i) - static_cast<double>(j)));
        }
        A_spd(i, i) += 1.0;  // boost diagonal for better conditioning
    }

    // Known solution x_true = [1, 2, 3, 4], compute b = A * x_true
    vec::dense_vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b_spd = A_spd * x_true;

    std::cout << "Matrix A (4x4 SPD):\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < n; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << A_spd(i, j);
        }
        std::cout << "]\n";
    }
    print_vector("b", b_spd);
    std::cout << "\n";

    // -- Method 1: LU factorization --------------------------------------
    {
        std::cout << "--- LU factorization (most general) ---\n";
        mat::dense2D<double> A_copy(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A_copy(i, j) = A_spd(i, j);

        std::vector<std::size_t> pivot;
        int info = lu_factor(A_copy, pivot);
        std::cout << "  lu_factor return code: " << info << "\n";

        vec::dense_vector<double> x(n);
        lu_solve(A_copy, pivot, x, b_spd);
        print_vector("x_lu", x);
        std::cout << "  ||b - Ax|| = " << std::scientific
                  << residual_norm(A_spd, x, b_spd) << "\n\n";
    }

    // -- Method 2: Cholesky factorization --------------------------------
    {
        std::cout << "--- Cholesky factorization (SPD only, 2x faster) ---\n";
        mat::dense2D<double> A_copy(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A_copy(i, j) = A_spd(i, j);

        int info = cholesky_factor(A_copy);
        std::cout << "  cholesky_factor return code: " << info << "\n";

        vec::dense_vector<double> x(n);
        cholesky_solve(A_copy, x, b_spd);
        print_vector("x_chol", x);
        std::cout << "  ||b - Ax|| = " << std::scientific
                  << residual_norm(A_spd, x, b_spd) << "\n\n";
    }

    // -- Method 3: QR factorization --------------------------------------
    {
        std::cout << "--- QR factorization (most stable) ---\n";
        mat::dense2D<double> A_copy(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A_copy(i, j) = A_spd(i, j);

        vec::dense_vector<double> tau;
        int info = qr_factor(A_copy, tau);
        std::cout << "  qr_factor return code: " << info << "\n";

        vec::dense_vector<double> x(n);
        qr_solve(A_copy, tau, x, b_spd);
        print_vector("x_qr", x);
        std::cout << "  ||b - Ax|| = " << std::scientific
                  << residual_norm(A_spd, x, b_spd) << "\n\n";
    }

    // -- Method 4: inv(A) * b (never do this!) ---------------------------
    {
        std::cout << "--- inv(A) * b (DO NOT do this in practice) ---\n";
        std::cout << "  Computing the full inverse is O(n^3) and numerically\n";
        std::cout << "  inferior to factorization-based solves.\n";
        auto A_inv = inv(A_spd);
        auto x_inv = A_inv * b_spd;
        vec::dense_vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x(i) = x_inv(i);
        print_vector("x_inv", x);
        std::cout << "  ||b - Ax|| = " << std::scientific
                  << residual_norm(A_spd, x, b_spd) << "\n\n";
    }

    // ======================================================================
    // Part 2: Non-SPD System
    // ======================================================================
    std::cout << "=== Part 2: Non-SPD System (Cholesky should fail) ===\n\n";

    // Make a symmetric INDEFINITE matrix (has negative eigenvalues)
    mat::dense2D<double> A_indef(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A_indef(i, j) = A_spd(i, j);
    // Reduce diagonal to make it indefinite (symmetric but not positive definite)
    A_indef(0, 0) = 0.1;
    A_indef(1, 1) = 0.1;

    auto b_indef = A_indef * x_true;

    std::cout << "Perturbed matrix (A(0,0)=0.1, A(1,1)=0.1 - symmetric indefinite):\n";
    std::cout << "  Diagonal now too small relative to off-diagonal entries.\n\n";

    // LU: works
    {
        std::cout << "--- LU: works on any non-singular matrix ---\n";
        mat::dense2D<double> A_copy(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A_copy(i, j) = A_indef(i, j);

        std::vector<std::size_t> pivot;
        int info = lu_factor(A_copy, pivot);
        vec::dense_vector<double> x(n);
        lu_solve(A_copy, pivot, x, b_indef);
        print_vector("x_lu", x);
        std::cout << "  info = " << info << ", ||b - Ax|| = " << std::scientific
                  << residual_norm(A_indef, x, b_indef) << "\n\n";
    }

    // Cholesky: fails
    {
        std::cout << "--- Cholesky: FAILS on indefinite matrix ---\n";
        mat::dense2D<double> A_copy(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A_copy(i, j) = A_indef(i, j);

        int info = cholesky_factor(A_copy);
        std::cout << "  cholesky_factor return code: " << info;
        if (info != 0)
            std::cout << " (failed at step " << info << " - matrix is not SPD)";
        std::cout << "\n\n";
    }

    // QR: works
    {
        std::cout << "--- QR: works on any non-singular matrix ---\n";
        mat::dense2D<double> A_copy(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A_copy(i, j) = A_indef(i, j);

        vec::dense_vector<double> tau;
        qr_factor(A_copy, tau);
        vec::dense_vector<double> x(n);
        qr_solve(A_copy, tau, x, b_indef);
        print_vector("x_qr", x);
        std::cout << "  ||b - Ax|| = " << std::scientific
                  << residual_norm(A_indef, x, b_indef) << "\n\n";
    }

    // ======================================================================
    // Part 3: Preconditioner Preview (ILU(0) and IC(0) on sparse SPD)
    // ======================================================================
    std::cout << "=== Part 3: Preconditioner Preview (Sparse SPD) ===\n\n";

    const std::size_t ns = 20;
    double hs = 1.0 / (ns + 1);
    mat::compressed2D<double> A_sparse(ns, ns);
    {
        mat::inserter<mat::compressed2D<double>> ins(A_sparse, 3);
        for (std::size_t i = 0; i < ns; ++i) {
            ins[i][i] << 2.0 / (hs * hs);
            if (i > 0)      ins[i][i-1] << -1.0 / (hs * hs);
            if (i + 1 < ns) ins[i][i+1] << -1.0 / (hs * hs);
        }
    }

    vec::dense_vector<double> b_sparse(ns, 1.0);

    // ILU(0) preconditioned BiCGSTAB
    {
        vec::dense_vector<double> x(ns, 0.0);
        itl::pc::ilu_0<double> ilu(A_sparse);
        itl::basic_iteration<double> iter(b_sparse, 500, 1.0e-10);
        itl::bicgstab(A_sparse, x, b_sparse, ilu, iter);
        std::cout << "  ILU(0) + BiCGSTAB: " << iter.iterations() << " iterations\n";
    }

    // IC(0) preconditioned CG
    {
        vec::dense_vector<double> x(ns, 0.0);
        itl::pc::ic_0<double> ic(A_sparse);
        itl::basic_iteration<double> iter(b_sparse, 500, 1.0e-10);
        itl::cg(A_sparse, x, b_sparse, ic, iter);
        std::cout << "  IC(0) + CG:        " << iter.iterations() << " iterations\n";
    }

    // Diagonal preconditioned CG (for comparison)
    {
        vec::dense_vector<double> x(ns, 0.0);
        itl::pc::diagonal<mat::compressed2D<double>> diag(A_sparse);
        itl::basic_iteration<double> iter(b_sparse, 500, 1.0e-10);
        itl::cg(A_sparse, x, b_sparse, diag, iter);
        std::cout << "  Diagonal + CG:     " << iter.iterations() << " iterations\n\n";
    }

    // -- Commentary -------------------------------------------------------
    std::cout << "=== Decision Tree for Choosing a Factorization ===\n";
    std::cout << "                  Is A symmetric?\n";
    std::cout << "                  /           \\\n";
    std::cout << "                Yes            No\n";
    std::cout << "               /                \\\n";
    std::cout << "          Is A positive      Use LU (general)\n";
    std::cout << "          definite?          or QR (stable)\n";
    std::cout << "          /        \\\n";
    std::cout << "        Yes         No\n";
    std::cout << "       /              \\\n";
    std::cout << "  Cholesky         LDL^T or LU\n";
    std::cout << "  (fastest)        (symmetric indef.)\n\n";
    std::cout << "For iterative methods on sparse systems:\n";
    std::cout << "  SPD -> CG + IC(0)    Non-symmetric -> BiCGSTAB + ILU(0)\n";

    return EXIT_SUCCESS;
}
