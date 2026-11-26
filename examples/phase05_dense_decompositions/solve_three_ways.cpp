// solve_three_ways.cpp - Solving Ax = b: LU vs Cholesky vs QR
//
// This example demonstrates:
//   1. When to use each direct factorization method
//   2. Cholesky is 2x faster but requires SPD matrices
//   3. LU with pivoting handles any non-singular system
//   4. QR is backward stable without pivoting, at twice LU's cost
//   5. Why you should never compute inv(A)*b in practice
//
// We solve the same system three ways, then show what happens
// when the matrix is not SPD (Cholesky correctly fails), and then
// sweep the Hilbert matrix, where inv(A)*b falls behind the three
// factorizations. The test matrices come from mtl::generators, whose
// properties are documented: lehmer (SPD, well conditioned), wilkinson
// (symmetric indefinite), hilbert (SPD, badly conditioned) and
// laplacian_1d.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
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

// lehmer and hilbert are implicit generators: they compute each entry on
// demand and store nothing. The factorizations work in place, so they need
// the matrix copied into a dense2D first.
template <typename Generator>
mat::dense2D<double> dense_copy(const Generator& G) {
    mat::dense2D<double> A(G.num_rows(), G.num_cols());
    for (std::size_t i = 0; i < G.num_rows(); ++i)
        for (std::size_t j = 0; j < G.num_cols(); ++j) A(i, j) = G(i, j);
    return A;
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

    // The Lehmer matrix, L(i, j) = (min(i, j) + 1) / (max(i, j) + 1): SPD,
    // with a tridiagonal inverse, and well conditioned.
    const mat::dense2D<double> A_spd = dense_copy(generators::lehmer<double>(n));

    // Known solution x_true = [1, 2, 3, 4], compute b = A * x_true
    vec::dense_vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b_spd = A_spd * x_true;

    std::cout << "Matrix A = lehmer(4), SPD, cond_2(A) = " << std::fixed << std::setprecision(2)
              << condition_number(A_spd) << ":\n";
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
        std::cout << "  Forming the inverse takes about 2n^3 flops, three times LU's\n";
        std::cout << "  2n^3/3. On this well-conditioned matrix it gives the same digits;\n";
        std::cout << "  Part 3 shows where it does not.\n";
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

    // Wilkinson's W4+ (MATLAB's wilkinson(4)): symmetric tridiagonal, with
    // 1.5, 0.5, 0.5, 1.5 on the diagonal and 1 beside it. Symmetric, but
    // with one negative eigenvalue, so it is not positive definite.
    const mat::dense2D<double> A_indef = generators::wilkinson<double>(n);

    auto b_indef = A_indef * x_true;

    std::cout << "Matrix A = wilkinson(4), symmetric tridiagonal:\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < n; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << A_indef(i, j);
        }
        std::cout << "]\n";
    }
    {
        auto ev = eigenvalue_symmetric(A_indef);
        std::vector<double> lam(ev.size());
        for (std::size_t i = 0; i < ev.size(); ++i) lam[i] = ev(i);
        std::sort(lam.begin(), lam.end());
        std::size_t negative = 0;
        std::cout << "  eigenvalues:";
        for (double l : lam) {
            std::cout << " " << std::fixed << std::setprecision(4) << l;
            if (l < 0.0) ++negative;
        }
        std::cout << "\n  " << negative << " negative eigenvalue: symmetric indefinite.\n\n";
    }

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
    // Part 3: The Hilbert Matrix -- Why Not inv(A) * b
    // ======================================================================
    std::cout << "=== Part 3: The Hilbert Matrix, n = 4 to 15 ===\n\n";
    std::cout << "H(i, j) = 1 / (i + j + 1) is SPD and badly conditioned. Solve H x = b\n";
    std::cout << "for x = (1, ..., 1). The backward error ||b - H x^|| / (||H|| ||x^||)\n";
    std::cout << "(infinity norms) measures how well each method solved the system it\n";
    std::cout << "was given; the forward error ||x^ - x|| / ||x|| how close it came to x.\n\n";
    std::cout << std::setw(4) << "n" << std::setw(11) << "cond_2(H)" << "  backward: "
              << std::setw(9) << "LU" << std::setw(10) << "Cholesky" << std::setw(9) << "QR"
              << std::setw(10) << "inv(H)*b" << "   forward: " << std::setw(9) << "LU"
              << std::setw(10) << "inv(H)*b" << "\n";
    auto hilbert_rows = example_trace::value::array();
    int chol_first_fail = 0;
    double inv_bw_first = 0.0, inv_bw_max = 0.0, cond_first_fail = 0.0, cond_last_ok = 0.0;
    std::size_t inv_bw_max_n = 0, lu_within = 0, lu_rows = 0;
    for (std::size_t m = 4; m <= 15; ++m) {
        const mat::dense2D<double> H = dense_copy(generators::hilbert<double>(m));
        const vec::dense_vector<double> ones(m, 1.0);
        const auto b = vec::dense_vector<double>(H * ones);
        const double normH = infinity_norm(H);
        auto backward = [&](const vec::dense_vector<double>& x) {
            auto Hx = H * x;
            double r = 0.0;
            for (std::size_t i = 0; i < m; ++i) r = std::max(r, std::abs(b(i) - Hx(i)));
            return r / (normH * infinity_norm(x));
        };
        auto forward = [&](const vec::dense_vector<double>& x) {
            double e = 0.0;
            for (std::size_t i = 0; i < m; ++i) e = std::max(e, std::abs(x(i) - 1.0));
            return e;   // ||x|| = 1 in the infinity norm
        };

        vec::dense_vector<double> x_lu(m), x_chol(m), x_qr(m), x_inv(m);
        mat::dense2D<double> F(H);
        std::vector<std::size_t> pivot;
        lu_factor(F, pivot);
        lu_solve(F, pivot, x_lu, b);
        mat::dense2D<double> C(H);
        const int chol_info = cholesky_factor(C);
        if (chol_info == 0) cholesky_solve(C, x_chol, b);
        mat::dense2D<double> Q(H);
        vec::dense_vector<double> tau;
        qr_factor(Q, tau);
        qr_solve(Q, tau, x_qr, b);
        auto xi = inv(H) * b;
        for (std::size_t i = 0; i < m; ++i) x_inv(i) = xi(i);

        const double cond = condition_number(H);
        const double bw_lu = backward(x_lu), bw_qr = backward(x_qr), bw_inv = backward(x_inv);
        const double bw_chol = chol_info == 0 ? backward(x_chol) : 0.0;
        if (m == 4) inv_bw_first = bw_inv;
        if (bw_inv > inv_bw_max) { inv_bw_max = bw_inv; inv_bw_max_n = m; }
        if (chol_info == 0 && chol_first_fail == 0) cond_last_ok = cond;
        if (chol_info != 0 && chol_first_fail == 0) {
            chol_first_fail = static_cast<int>(m);
            cond_first_fail = cond;
        }
        // Where the SVD can still resolve cond(H), LU's forward error stays
        // below cond(H) * eps, the sensitivity of the problem itself.
        if (cond < 1e16) {
            ++lu_rows;
            if (forward(x_lu) <= cond * std::numeric_limits<double>::epsilon()) ++lu_within;
        }
        std::cout << std::setw(4) << m << std::scientific << std::setprecision(2) << std::setw(11) << cond
                  << "            " << std::setw(9) << bw_lu;
        if (chol_info == 0) std::cout << std::setw(10) << bw_chol;
        else                std::cout << std::setw(10) << ("fails@" + std::to_string(chol_info));
        std::cout << std::setw(9) << bw_qr << std::setw(10) << bw_inv << "            "
                  << std::setw(9) << forward(x_lu) << std::setw(10) << forward(x_inv) << "\n";

        auto row = example_trace::value::object();
        row["n"] = m;
        row["cond"] = example_trace::value::rounded(cond, 6);
        row["cholesky_info"] = chol_info;
        auto method = [&](const char* name, const vec::dense_vector<double>& x, bool ok) {
            auto o = example_trace::value::object();
            o["method"] = name;
            o["backward"] = ok ? example_trace::value::rounded(backward(x), 6) : example_trace::value();
            o["forward"] = ok ? example_trace::value::rounded(forward(x), 6) : example_trace::value();
            return o;
        };
        auto ms = example_trace::value::array();
        ms.push_back(method("LU", x_lu, true));
        ms.push_back(method("Cholesky", x_chol, chol_info == 0));
        ms.push_back(method("QR", x_qr, true));
        ms.push_back(method("inv(H) * b", x_inv, true));
        row["methods"] = ms;
        hilbert_rows.push_back(row);
    }
    std::cout << std::scientific << std::setprecision(1);
    std::cout << "\nThe three factorizations stay backward stable: their backward error\n";
    std::cout << "is a small multiple of eps = 2.2e-16 at every n. inv(H) * b is not:\n";
    std::cout << "its backward error rises from " << inv_bw_first << " at n = 4 to " << inv_bw_max
              << " at n = " << inv_bw_max_n << ".\n";
    std::cout << "The forward errors grow with cond(H) for every method, because the\n";
    std::cout << "problem itself is that sensitive. LU's stays below cond(H) * eps at\n";
    std::cout << lu_within << " of the " << lu_rows << " sizes where cond(H) < 1e16; inv(H) * b's is larger.\n";
    std::cout << "Cholesky first fails at n = " << chol_first_fail << ", though H is SPD. Past\n";
    std::cout << "cond(H) = 1 / eps = 4.5e15, H is positive definite only up to rounding,\n";
    std::cout << "and a pivot can come out non-positive. At n = " << chol_first_fail - 1
              << " (cond " << cond_last_ok << ")\n";
    std::cout << "it still succeeds.\n";
    std::cout << "(cond_2 comes from the SVD, which cannot resolve singular values much\n";
    std::cout << "below eps * ||H||: past about 1e16 it is not reliable, and it stops\n";
    std::cout << "growing with n. At n = " << chol_first_fail << " it reads " << cond_first_fail << ".)\n\n";
    if (trace) {
        trace["hilbert"] = hilbert_rows;
        trace["hilbert_cholesky_first_fail"] = chol_first_fail;
    }

    // ======================================================================
    // Part 4: Preconditioner Preview (ILU(0) and IC(0) on sparse SPD)
    // ======================================================================
    std::cout << "=== Part 4: Preconditioner Preview (Sparse SPD) ===\n\n";

    // The 1-D Laplacian tridiag(-1, 2, -1). The usual 1/h^2 scale would not
    // change the iteration counts: the tolerance is relative to ||b||.
    const std::size_t ns = 20;
    const mat::compressed2D<double> A_sparse = generators::laplacian_1d<double>(ns);

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
