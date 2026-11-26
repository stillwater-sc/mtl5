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
#include <string>
#include <utility>
#include <vector>

#include "../common/trace.hpp"

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


// -- Step-by-step states, for the docs-site animation (--trace) -------------
// MTL5's factorizations return their factors, not their intermediate working
// matrices. Each intermediate state is determined by the factors, so it is
// rebuilt from them here: the animation shows the elimination MTL5 did.

using grid = std::vector<std::vector<double>>;

grid to_grid(const mat::dense2D<double>& M) {
    grid g(M.num_rows(), std::vector<double>(M.num_cols()));
    for (std::size_t i = 0; i < M.num_rows(); ++i)
        for (std::size_t j = 0; j < M.num_cols(); ++j) g[i][j] = M(i, j);
    return g;
}

// LU with partial pivoting, after k steps, in the row order the algorithm
// holds at that point: rows < k hold U (with L's multipliers left of the
// diagonal), rows >= k hold the multipliers and the Schur complement.
// lu_factor's final rows are in the order after all n swaps; the swaps
// pivot[k..n-1] are undone, last first, to get the order after k swaps.
grid lu_state(const mat::dense2D<double>& LU, const std::vector<std::size_t>& pivot,
              const mat::dense2D<double>& A, std::size_t k) {
    const std::size_t n = A.num_rows();
    std::vector<std::size_t> row(n);   // row[i] = original row in final position i
    for (std::size_t i = 0; i < n; ++i) row[i] = i;
    for (std::size_t s = 0; s < n; ++s) std::swap(row[s], row[pivot[s]]);
    grid W(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (i < k || j < k) {
                W[i][j] = LU(i, j);                              // U rows, L multipliers
            } else {
                double s2 = A(row[i], j);                        // Schur complement
                for (std::size_t p = 0; p < k; ++p) s2 -= LU(i, p) * LU(p, j);
                W[i][j] = s2;
            }
        }
    for (std::size_t s = n; s-- > k;) std::swap(W[s], W[pivot[s]]);
    return W;
}

// Cholesky after k columns: L's columns < k in the lower triangle, and the
// Schur complement A22 - L21 L21^T in the trailing block (lower half shown).
grid cholesky_state(const mat::dense2D<double>& L, const mat::dense2D<double>& A, std::size_t k) {
    const std::size_t n = A.num_rows();
    grid W(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            if (j < k) {
                W[i][j] = L(i, j);
            } else {
                double s2 = A(i, j);
                for (std::size_t p = 0; p < k; ++p) s2 -= L(i, p) * L(j, p);
                W[i][j] = s2;
            }
        }
    return W;
}

// QR after k Householder reflections: H_{k-1} ... H_0 A, with each H_j built
// from the reflector qr_factor stored (v = [1; QR(j+1:, j)], tau(j)).
grid qr_state(const mat::dense2D<double>& QR, const vec::dense_vector<double>& tau,
              const mat::dense2D<double>& A, std::size_t k) {
    const std::size_t n = A.num_rows();
    grid W = to_grid(A);
    for (std::size_t j = 0; j < k; ++j) {
        std::vector<double> v(n, 0.0);
        v[j] = 1.0;
        for (std::size_t i = j + 1; i < n; ++i) v[i] = QR(i, j);
        for (std::size_t c = 0; c < n; ++c) {
            double d = 0.0;
            for (std::size_t i = j; i < n; ++i) d += v[i] * W[i][c];
            for (std::size_t i = j; i < n; ++i) W[i][c] -= tau(j) * v[i] * d;
        }
    }
    return W;
}

// Textbook flop counts of step k on an n x n matrix (m = n - k rows left).
double lu_flops(std::size_t n, std::size_t k) {
    const double m = static_cast<double>(n - k);
    return (m - 1) + 2.0 * (m - 1) * (m - 1);                 // multipliers + rank-1 update
}
double cholesky_flops(std::size_t n, std::size_t k) {
    const double m = static_cast<double>(n - k);
    return 1 + (m - 1) + (m - 1) * m;                         // sqrt + divisions + half update
}
double qr_flops(std::size_t n, std::size_t k) {
    const double m = static_cast<double>(n - k);
    return 3.0 * m + 4.0 * m * (m - 1);                       // reflector + apply to m - 1 columns
}

example_trace::value grid_value(const grid& g) {
    auto rows = example_trace::value::array();
    for (const auto& r : g) rows.push_back(example_trace::value::numbers(r, 6));
    return rows;
}

// Every state of the three factorizations of A, for one part of the example.
example_trace::value factor_states(const char* name, const mat::dense2D<double>& A) {
    const std::size_t n = A.num_rows();
    auto part = example_trace::value::object();
    part["name"] = name;
    part["matrix"] = grid_value(to_grid(A));

    mat::dense2D<double> LU(A);
    std::vector<std::size_t> pivot;
    const int lu_info = lu_factor(LU, pivot);
    mat::dense2D<double> L(A);
    const int chol_info = cholesky_factor(L);
    mat::dense2D<double> QR(A);
    vec::dense_vector<double> tau;
    const int qr_info = qr_factor(QR, tau);

    // Cholesky stops at the failing pivot: its columns before it are valid.
    const std::size_t chol_steps = chol_info == 0 ? n : static_cast<std::size_t>(chol_info - 1);
    auto method = [&](const char* m, int info, std::size_t steps, auto state, auto flops) {
        auto o = example_trace::value::object();
        o["method"] = m;
        o["info"] = info;
        auto states = example_trace::value::array();
        auto cost = example_trace::value::array();
        double total = 0.0;
        for (std::size_t k = 0; k <= steps; ++k) {
            states.push_back(grid_value(state(k)));
            cost.push_back(total);
            if (k < steps) total += flops(n, k);
        }
        o["states"] = states;
        o["flops"] = cost;
        return o;
    };
    auto methods = example_trace::value::array();
    auto lu = method("LU", lu_info, n, [&](std::size_t k) { return lu_state(LU, pivot, A, k); }, lu_flops);
    lu["pivot"] = example_trace::value::numbers(pivot);
    methods.push_back(lu);
    auto ch = method("Cholesky", chol_info, chol_steps, [&](std::size_t k) { return cholesky_state(L, A, k); }, cholesky_flops);
    if (chol_info != 0) {
        // The pivot that failed: the Schur complement's diagonal entry.
        ch["failed_pivot"] = example_trace::value::rounded(cholesky_state(L, A, chol_steps)[chol_steps][chol_steps], 8);
    }
    methods.push_back(ch);
    methods.push_back(method("QR", qr_info, n, [&](std::size_t k) { return qr_state(QR, tau, A, k); }, qr_flops));
    part["methods"] = methods;
    return part;
}

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "5B solving Ax = b three ways");

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

    if (trace) {
        trace["n"] = n;
        auto parts = example_trace::value::array();
        parts.push_back(factor_states("SPD", A_spd));
        parts.push_back(factor_states("indefinite", A_indef));
        trace["parts"] = parts;
        // The same counts summed for larger n: the leading terms 2n^3/3,
        // n^3/3 and 4n^3/3 take over from the lower-order ones.
        auto table = example_trace::value::array();
        for (std::size_t m : {4, 10, 100, 1000}) {
            double l = 0, c = 0, q = 0;
            for (std::size_t k = 0; k < m; ++k) {
                l += lu_flops(m, k);
                c += cholesky_flops(m, k);
                q += qr_flops(m, k);
            }
            auto row = example_trace::value::object();
            row["n"] = m;
            row["lu"] = l;
            row["cholesky"] = c;
            row["qr"] = q;
            table.push_back(row);
        }
        trace["flops_by_n"] = table;
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

    return trace.close();   // EXIT_FAILURE if a requested trace could not be written
}
