// krylov_comparison.cpp -- Krylov Solver Comparison
//
// This example compares the Phase 13 Krylov solvers on sparse systems:
//
//   1. CGS, BiCGSTAB, BiCGSTAB(2) and BiCGSTAB(4) on a non-symmetric 2-D
//      convection-diffusion system, each with the identity, the diagonal
//      (Jacobi) and the ILU(0) preconditioner
//   2. MINRES on a symmetric indefinite system, the discrete Helmholtz
//      operator, against CG and BiCGSTAB
//   3. Why MINRES's preconditioner must be symmetric positive definite
//
// Every solve prints its return code next to its iteration count, so a
// breakdown cannot pass for slow convergence.

#include <mtl/mtl.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace mtl;
using csr   = mat::compressed2D<double>;
using vec_t = vec::dense_vector<double>;

/// Non-symmetric: -div(k grad u) + c du/dx = f on the unit square, on an
/// m x m interior grid, central differences, with diffusion
/// k(x, y) = 1 + 99 x^2. The diagonal varies by a factor of about 100 across
/// the grid, so the diagonal preconditioner has something to do. (With a
/// constant diagonal, as k = 1 gives, Jacobi is M = c I: a Krylov method's
/// iterates do not change under that scaling, and it would show nothing.)
/// The matrix is not tridiagonal, so ILU(0) drops fill and is approximate.
static csr make_convection_diffusion(std::size_t m, double c) {
    const double h = 1.0 / static_cast<double>(m + 1);
    auto k = [](double x) { return 1.0 + 99.0 * x * x; };
    const std::size_t n = m * m;
    csr A(n, n);
    {
        mat::inserter<csr> ins(A, 5);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < m; ++i) {
                const std::size_t r = j * m + i;
                const double x = static_cast<double>(i + 1) * h;
                const double kw = k(x - h / 2), ke = k(x + h / 2), kns = k(x);
                ins[r][r] << kw + ke + 2.0 * kns;
                if (i > 0)     ins[r][r - 1] << -kw - c * h / 2;
                if (i + 1 < m) ins[r][r + 1] << -ke + c * h / 2;
                if (j > 0)     ins[r][r - m] << -kns;
                if (j + 1 < m) ins[r][r + m] << -kns;
            }
    }
    return A;
}

/// Subtract sigma from the diagonal, in place, through the CRS arrays.
static void shift_diagonal(csr& A, double sigma) {
    const auto& starts = A.ref_major();
    const auto& cols = A.ref_minor();
    auto& vals = A.ref_data();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t p = starts[i]; p < starts[i + 1]; ++p)
            if (cols[p] == i) vals[p] -= sigma;
}

static double relative_residual(const csr& A, const vec_t& x, const vec_t& b) {
    auto Ax = A * x;
    vec_t r(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) r(i) = b(i) - Ax(i);
    return two_norm(r) / two_norm(b);
}

/// M = |diag(A)|: symmetric positive definite whenever no diagonal entry is 0.
struct abs_diagonal {
    vec_t inv;
    explicit abs_diagonal(const csr& A) : inv(A.num_rows()) {
        for (std::size_t i = 0; i < A.num_rows(); ++i) inv(i) = 1.0 / std::abs(A(i, i));
    }
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        for (std::size_t i = 0; i < x.size(); ++i) x(i) = inv(i) * b(i);
    }
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const { solve(x, b); }
};

struct row {
    std::string solver, pc;
    int info;
    int iterations;
    double residual;
};

static void print(const std::vector<row>& rows) {
    std::cout << std::left << std::setw(14) << "Solver" << std::setw(18) << "Preconditioner"
              << std::right << std::setw(6) << "info" << std::setw(8) << "Iters"
              << std::setw(14) << "Rel.Resid" << "\n";
    std::cout << std::string(60, '-') << "\n";
    for (const auto& r : rows)
        std::cout << std::left << std::setw(14) << r.solver << std::setw(18) << r.pc << std::right
                  << std::setw(6) << r.info << std::setw(8) << r.iterations << std::scientific
                  << std::setprecision(2) << std::setw(14) << r.residual << "\n";
}

/// Run `solve(x, iter)` from x = 0 and record the outcome.
template <typename Solve>
static row run(const std::string& solver, const std::string& pc, const csr& A, const vec_t& b,
               int max_iter, Solve solve) {
    vec_t x(b.size(), 0.0);
    itl::basic_iteration<double> iter(b, max_iter, 1e-10);
    const int info = solve(x, iter);
    return {solver, pc, info, iter.iterations(), relative_residual(A, x, b)};
}

int main() {
    std::cout << "=== Krylov Solver Comparison ===\n\n";

    // -- 1. Non-symmetric system ------------------------------------------
    const std::size_t m1 = 30, n = m1 * m1;
    const csr A_ns = make_convection_diffusion(m1, 40.0);
    const vec_t b(n, 1.0);
    std::cout << "--- Non-symmetric: 2-D convection-diffusion, k = 1 + 99 x^2, c = 40, "
              << m1 << " x " << m1 << " grid ---\n";
    std::cout << "The diagonal runs from " << std::fixed << std::setprecision(2) << A_ns(0, 0)
              << " to " << A_ns(m1 - 1, m1 - 1) << ", so Jacobi rescales the rows unevenly.\n\n";
    {
        const itl::pc::identity<csr> I(A_ns);
        const itl::pc::diagonal<csr> D(A_ns);
        const itl::pc::ilu_0<double> ILU(A_ns);
        std::vector<row> rows;
        auto each_pc = [&](const std::string& name, auto solver) {
            rows.push_back(run(name, "identity", A_ns, b, 2000, [&](vec_t& x, auto& it) { return solver(x, I, it); }));
            rows.push_back(run(name, "diagonal", A_ns, b, 2000, [&](vec_t& x, auto& it) { return solver(x, D, it); }));
            rows.push_back(run(name, "ILU(0)", A_ns, b, 2000, [&](vec_t& x, auto& it) { return solver(x, ILU, it); }));
        };
        each_pc("CGS", [&](vec_t& x, const auto& M, auto& it) { return itl::cgs(A_ns, x, b, M, it); });
        each_pc("BiCGSTAB", [&](vec_t& x, const auto& M, auto& it) { return itl::bicgstab(A_ns, x, b, M, it); });
        each_pc("BiCGSTAB(2)", [&](vec_t& x, const auto& M, auto& it) { return itl::bicgstab_ell(A_ns, x, b, M, it, 2); });
        each_pc("BiCGSTAB(4)", [&](vec_t& x, const auto& M, auto& it) { return itl::bicgstab_ell(A_ns, x, b, M, it, 4); });
        print(rows);
        std::cout << "info 0 is convergence, 1 the iteration limit, 2 and up a breakdown.\n";
        std::cout << "Jacobi evens out the rows; ILU(0) also captures the coupling between\n";
        std::cout << "neighbours, and cuts the counts further. A higher BiCGSTAB(l) takes fewer,\n";
        std::cout << "though each of its iterations costs more products with A.\n";
        for (const auto& r : rows)
            if (r.info != 0)
                std::cout << r.solver << " with " << r.pc << " failed: info " << r.info << " after "
                          << r.iterations << " iterations, residual " << std::scientific
                          << std::setprecision(1) << r.residual << ".\n";
        std::cout << "CGS squares BiCG's residual polynomial. Where BiCG's residual is erratic,\n";
        std::cout << "CGS's is erratic squared, and on this non-symmetric matrix it grows\n";
        std::cout << "until the recurrence breaks down, unless ILU(0) makes the problem easy.\n";
        std::cout << "BiCGSTAB was designed to smooth exactly that out.\n\n";
    }

    // -- 2. Symmetric indefinite: the discrete Helmholtz operator ---------
    const std::size_t m = 30;
    const double sigma = 1.3;
    csr H = generators::laplacian_2d<double>(m, m);   // SPD
    shift_diagonal(H, sigma);                          // now indefinite
    const vec_t bh(m * m, 1.0);
    std::size_t negative = 0;
    {
        // The 2-D Laplacian's eigenvalues are known in closed form:
        // 4 - 2 cos(i pi / (m + 1)) - 2 cos(j pi / (m + 1)).
        const double pi = 3.14159265358979323846;
        for (std::size_t i = 1; i <= m; ++i)
            for (std::size_t j = 1; j <= m; ++j)
                if (4.0 - 2.0 * std::cos(i * pi / (m + 1)) - 2.0 * std::cos(j * pi / (m + 1)) < sigma) ++negative;
    }
    std::cout << "--- Symmetric indefinite: Helmholtz, laplacian_2d(" << m << ", " << m << ") - "
              << std::fixed << std::setprecision(1) << sigma << " I ---\n";
    std::cout << negative << " of " << m * m << " eigenvalues are negative. CG's theory assumes a\n";
    std::cout << "positive definite matrix; MINRES needs only symmetry, and minimizes\n";
    std::cout << "||b - A x|| over the Krylov space, so its residual never increases.\n\n";
    {
        const itl::pc::identity<csr> I(H);
        std::cout << "True relative residual after k iterations, from x = 0:\n";
        std::cout << std::setw(6) << "k" << std::setw(14) << "CG" << std::setw(14) << "BiCGSTAB"
                  << std::setw(14) << "MINRES" << "\n";
        // ||b|| is the residual at k = 0.
        double cg_max = 1.0, minres_prev = 1.0;
        bool minres_monotone = true;
        for (int k = 5; k <= 140; k += 5) {
            auto capped = [&](auto solve) {
                vec_t x(m * m, 0.0);
                itl::basic_iteration<double> iter(bh, k, 1e-10);
                solve(x, iter);
                return relative_residual(H, x, bh);
            };
            const double r_cg = capped([&](vec_t& x, auto& it) { return itl::cg(H, x, bh, I, it); });
            const double r_bi = capped([&](vec_t& x, auto& it) { return itl::bicgstab(H, x, bh, I, it); });
            const double r_mr = capped([&](vec_t& x, auto& it) { return itl::minres(H, x, bh, I, it); });
            cg_max = std::max(cg_max, r_cg);
            if (r_mr > minres_prev * (1.0 + 1e-12)) minres_monotone = false;
            minres_prev = r_mr;
            if (k % 20 == 0 || k == 5)
                std::cout << std::setw(6) << k << std::scientific << std::setprecision(2) << std::setw(14)
                          << r_cg << std::setw(14) << r_bi << std::setw(14) << r_mr << "\n";
        }
        std::cout << "(Sampled every 5 iterations, every 20th shown.) CG's residual reached\n";
        std::cout << std::setprecision(2) << cg_max << " times ||b|| on the way";
        std::cout << (cg_max > 1.0 ? ", above where it started; " : "; ");
        std::cout << "MINRES's " << (minres_monotone ? "never rose" : "rose too (rounding)") << ".\n";
        std::cout << "CG converges on this matrix anyway, but nothing guarantees that: it can\n";
        std::cout << "break down on an indefinite one. BiCGSTAB ignores the symmetry and pays.\n\n";

        std::vector<row> rows;
        rows.push_back(run("CG", "identity", H, bh, 2000, [&](vec_t& x, auto& it) { return itl::cg(H, x, bh, I, it); }));
        rows.push_back(run("BiCGSTAB", "identity", H, bh, 2000, [&](vec_t& x, auto& it) { return itl::bicgstab(H, x, bh, I, it); }));
        rows.push_back(run("MINRES", "identity", H, bh, 2000, [&](vec_t& x, auto& it) { return itl::minres(H, x, bh, I, it); }));
        print(rows);
        std::cout << "\n";
    }

    // -- 3. MINRES needs an SPD preconditioner ----------------------------
    const std::size_t n3 = 50;
    csr A_si(n3, n3);
    {
        mat::inserter<csr> ins(A_si, 3);
        for (std::size_t i = 0; i < n3; ++i) {
            ins[i][i] << ((i % 2 == 0) ? 4.0 : -3.0);   // alternating diagonal
            if (i > 0)      ins[i][i - 1] << 1.0;
            if (i + 1 < n3) ins[i][i + 1] << 1.0;
        }
    }
    const vec_t b3(n3, 1.0);
    std::cout << "--- MINRES needs a symmetric positive definite preconditioner ---\n";
    std::cout << "tridiag(1, [4, -3, 4, -3, ...], 1), n = " << n3 << ". MINRES runs its Lanczos\n";
    std::cout << "process in the inner product <u, v> = u^T M^{-1} v, which is one only when M\n";
    std::cout << "is SPD. Jacobi's M = diag(A) is indefinite here; M = |diag(A)| is SPD.\n\n";
    {
        const itl::pc::identity<csr> I(A_si);
        const itl::pc::diagonal<csr> D(A_si);
        const abs_diagonal AD(A_si);
        std::vector<row> rows;
        rows.push_back(run("MINRES", "identity", A_si, b3, 500, [&](vec_t& x, auto& it) { return itl::minres(A_si, x, b3, I, it); }));
        rows.push_back(run("MINRES", "diag(A)", A_si, b3, 500, [&](vec_t& x, auto& it) { return itl::minres(A_si, x, b3, D, it); }));
        rows.push_back(run("MINRES", "|diag(A)|", A_si, b3, 500, [&](vec_t& x, auto& it) { return itl::minres(A_si, x, b3, AD, it); }));
        print(rows);
        std::cout << "With diag(A), MINRES stops at once with info 7: r0^T M^{-1} r0 < 0, so M\n";
        std::cout << "is not positive definite (before issue 949 it took the absolute value,\n";
        std::cout << "and broke down with info 2 after 149 iterations, with most of the residual\n";
        std::cout << "left). |diag(A)| is valid, but on this matrix, whose\n";
        std::cout << "eigenvalues sit in two tight clusters, it has nothing to improve.\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
