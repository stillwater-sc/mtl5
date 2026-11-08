#pragma once
// mtl5_bench -- Krylov solvers, preconditioners, smoothers, multigrid and the
// sparse eigensolvers (#721 milestone M5). Inputs are generated, as in M4
// (entries_sparse.hpp supplies the Poisson and convection-diffusion matrices).
//
// CHECKS -- each is one a wrong result cannot pass:
//   Krylov solvers         the solver reports convergence AND the TRUE residual
//                          |b - A x| / |b| (recomputed in long double, not the
//                          solver's recursive estimate) is at most 1e-6
//   preconditioner setup   the preconditioner is CORRECT and USEFUL: with it, a
//                          Krylov solve converges in at most 3/4 of the
//                          iterations it takes unpreconditioned
//   smoothers              one sweep satisfies its defining splitting row by
//                          row, within gamma_k (Higham): Jacobi D x1 = b - (L+U) x0,
//                          Gauss-Seidel (D+L) x1 = b - U x0, SOR
//                          (D/w + L) x1 = b - (U - (1/w - 1) D) x0
//   multigrid              a fixed number of V-cycles reduces the relative
//                          residual below 1e-6 (a broken level or transfer stalls)
//   eigensolvers           shift-invert: converged, and the ANALYTIC smallest
//                          eigenvalues to 1e-10; Lanczos, Arnoldi and power
//                          iteration (run at a fixed, affordable Krylov size that
//                          need not converge on a clustered spectrum): the
//                          Bauer-Fike ENCLOSURE of each returned pair against the
//                          analytic spectrum. The
//                          Laplacian on an nx x ny grid has eigenvalues
//                          (2 - 2cos(i pi/(nx+1))) + (2 - 2cos(j pi/(ny+1))), and
//                          constant-coefficient convection-diffusion is similar to
//                          a symmetric operator with eigenvalues
//                          4 - 2 sqrt(1-c^2) (cos(i pi/(nx+1)) + cos(j pi/(ny+1)))
//                          -- real, though the matrix is not symmetric. Grids are
//                          nx x (nx+1) so that the spectrum has no symmetric
//                          multiplicities a single Krylov space cannot resolve.
//
// WORK MODEL. Krylov solvers, preconditioner setups and eigensolvers are TIMED
// ONLY (n/a): their cost is an iteration count times a per-iteration cost, and
// the count is the interesting number -- it is reported in the check's metric
// (JSON). Smoothers and the multigrid V-cycle have honest flop models (2 flops
// per stored nonzero per sweep) and sit on the bandwidth side of the roofline.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mtl/generators/laplacian.hpp>
#include <mtl/itl/eigen/arnoldi.hpp>
#include <mtl/itl/eigen/lanczos.hpp>
#include <mtl/itl/eigen/power_iteration.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicg.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/bicgstab_ell.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/cgs.hpp>
#include <mtl/itl/krylov/gmres.hpp>
#include <mtl/itl/krylov/idr_s.hpp>
#include <mtl/itl/krylov/minres.hpp>
#include <mtl/itl/krylov/qmr.hpp>
#include <mtl/itl/krylov/tfqmr.hpp>
#include <mtl/itl/mg/multigrid.hpp>
#include <mtl/itl/mg/prolongation.hpp>
#include <mtl/itl/mg/restriction.hpp>
#include <mtl/itl/pc/block_diagonal.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/ildl.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ilut.hpp>
#include <mtl/itl/pc/ssor.hpp>
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include <mtl/itl/smoother/jacobi.hpp>
#include <mtl/itl/smoother/sor.hpp>
#include <mtl/sparse/eigen/shift_invert.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>

#include <benchmarks/platform/entries_sparse.hpp>

namespace mtl::bench::platform::iterative {

using sparse_entries::csr;
using vec_t = vec::dense_vector<double>;
namespace itl = mtl::itl;

inline std::size_t grid_krylov(size_class c) {
    return c == size_class::small ? 32 : c == size_class::medium ? 128 : 256;
}
inline std::size_t grid_eig(size_class c) {
    return c == size_class::small ? 24 : c == size_class::medium ? 64 : 128;
}

// -- generators --------------------------------------------------------------

/// Convection-diffusion on an nx x ny grid with convection strength c.
inline csr convdiff(std::size_t nx, std::size_t ny, double c) {
    const std::size_t n = nx * ny;
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (std::size_t iy = 0; iy < ny; ++iy)
            for (std::size_t ix = 0; ix < nx; ++ix) {
                const std::size_t r = iy * nx + ix;
                ins[r][r] << 4.0;
                if (ix > 0) ins[r][r - 1] << -1.0 - c;
                if (ix + 1 < nx) ins[r][r + 1] << -1.0 + c;
                if (iy > 0) ins[r][r - nx] << -1.0 - c;
                if (iy + 1 < ny) ins[r][r + nx] << -1.0 + c;
            }
    }
    return A;
}

/// Rows scaled by factors in [1, 100]: badly scaled, so a diagonal (Jacobi) or
/// block-diagonal preconditioner has something to repair.
inline csr row_scaled(const csr& A, std::uint64_t seed) {
    xorshift64 g(seed);
    std::vector<double> d(A.num_rows());
    for (auto& v : d) v = std::pow(10.0, 2.0 * g.next_double());
    csr B(A.num_rows(), A.num_cols());
    {
        mat::inserter<csr> ins(B);
        for (std::size_t r = 0; r < A.num_rows(); ++r)
            for (std::size_t q = A.ref_major()[r]; q < A.ref_major()[r + 1]; ++q)
                ins[r][A.ref_minor()[q]] << d[r] * A.ref_data()[q];
    }
    return B;
}

/// |b - A x|_2 / |b|_2 in long double: the TRUE residual, not a recursion's.
inline long double true_relres(const csr& A, const vec_t& x, const vec_t& b) {
    const std::size_t n = A.num_rows();
    std::vector<long double> xl(n), y, ay;
    for (std::size_t i = 0; i < n; ++i) xl[i] = x(i);
    sparse_entries::ref_mult(A, xl, y, ay);
    long double r2 = 0, b2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        r2 += (b(i) - y[i]) * (b(i) - y[i]);
        b2 += (long double)b(i) * b(i);
    }
    return std::sqrt(r2 / b2);
}

// -- Krylov solvers ------------------------------------------------------------

/// Solve to rtol 1e-8 from x = 0. CG and MINRES on the SPD Poisson matrix, the
/// rest on unsymmetric convection-diffusion, with ILU(0) (diagonal for MINRES,
/// which needs an SPD preconditioner).
///
/// Two deliberate exceptions, both findings of this benchmark:
///  * CG is paired with ILU(0), not IC(0). pc::ic_0's apply is QUADRATIC in n
///    (131 ms per apply at n = 16384 against ILU(0)'s 0.14 ms), so CG + IC(0)
///    would take ~9 s at medium and hours at large. On an SPD matrix ILU(0)
///    computes the same incomplete factor (CG takes 35 iterations with either).
///  * TFQMR runs on the SPD Poisson matrix. On unsymmetric convection-diffusion
///    it stalls at a 6-9% residual after thousands of iterations with ANY
///    preconditioner, where BiCGSTAB converges in 13-52; its unit tests use only
///    symmetric matrices, so nothing caught it.
template <int Which>
problem krylov(size_class c) {
    const std::size_t k = grid_krylov(c);
    const bool spd = Which == 0 || Which == 1 || Which == 8;
    struct S {
        csr A;
        vec_t b, x;
        int err = -1, iters = 0;
    };
    auto s = std::make_shared<S>();
    s->A = spd ? sparse_entries::poisson2d(k) : sparse_entries::convdiff2d(k);
    const std::size_t n = s->A.num_rows();
    s->b = vec_of<double>(n, 301 + Which);
    s->x = vec_t(n);
    auto solve = [s](const auto& M) {
        itl::basic_iteration<double> iter(s->b, 20000, 1e-8);
        int e = 0;
        if constexpr (Which == 0) e = itl::cg(s->A, s->x, s->b, M, iter);
        else if constexpr (Which == 1) e = itl::minres(s->A, s->x, s->b, M, iter);
        else if constexpr (Which == 2) e = itl::bicg(s->A, s->x, s->b, M, iter);
        else if constexpr (Which == 3) e = itl::bicgstab(s->A, s->x, s->b, M, iter);
        else if constexpr (Which == 4) e = itl::bicgstab_ell(s->A, s->x, s->b, M, iter, 2);
        else if constexpr (Which == 5) e = itl::cgs(s->A, s->x, s->b, M, iter);
        else if constexpr (Which == 6) e = itl::gmres(s->A, s->x, s->b, M, iter, 30);
        else if constexpr (Which == 7) e = itl::qmr(s->A, s->x, s->b, M, iter);
        else if constexpr (Which == 8) e = itl::tfqmr(s->A, s->x, s->b, M, iter);
        else {
            // IDR(s) at s = 1, 2, 4, 8 (Which 10, 11, 9, 12): s = 1 is close to
            // BiCGSTAB; a larger s needs fewer iterations but costs s + 1 matvecs
            // and O(s n) memory per cycle, so the best s depends on the machine.
            constexpr std::size_t sv = Which == 10 ? 1 : Which == 11 ? 2 : Which == 12 ? 8 : 4;
            e = itl::idr_s(s->A, s->x, s->b, M, iter, sv);
        }
        s->err = e;
        s->iters = iter.iterations();
    };
    std::function<void()> run;
    if constexpr (Which == 1) {
        auto M = std::make_shared<itl::pc::diagonal<csr>>(s->A);
        run = [solve, M] { solve(*M); };
    } else {
        auto M = std::make_shared<itl::pc::ilu_0<double>>(s->A);
        run = [solve, M] { solve(*M); };
    }
    problem p{std::to_string(k) + "^2 grid, n " + std::to_string(n), 0.0, 0.0,
              [s] { for (std::size_t i = 0; i < s->x.size(); ++i) s->x(i) = 0.0; }, run, {}};
    p.check = [s] {
        check_result r;
        r.tolerance = 1e-6;
        r.error = double(true_relres(s->A, s->x, s->b));
        r.what = "converged in " + std::to_string(s->iters) + " iterations; true |b-Ax|/|b|";
        r.ok = s->err == 0 && r.error <= r.tolerance;
        if (s->err != 0) r.what += " (solver reported failure " + std::to_string(s->err) + ")";
        return r;
    };
    return p;
}

// -- preconditioner setups -------------------------------------------------

/// Iterations GMRES(30) (or CG, for SPD) needs to reach rtol 1e-8 with M.
template <bool Spd, typename M>
int iterations_with(const csr& A, const vec_t& b, const M& pc) {
    vec_t x(A.num_rows());
    for (std::size_t i = 0; i < x.size(); ++i) x(i) = 0.0;
    itl::basic_iteration<double> iter(b, 20000, 1e-8);
    const int e = Spd ? itl::cg(A, x, b, pc, iter) : itl::gmres(A, x, b, pc, iter, 30);
    return e == 0 ? iter.iterations() : 1 << 30;
}

/// Time a preconditioner's SETUP (its factorization or extraction); check that
/// it cuts the Krylov iteration count to at most 3/4 of the unpreconditioned one.
template <int Which>
problem pc_setup(size_class c) {
    const std::size_t k = grid_krylov(c);
    // ic_0 and ildl are symmetric: CG on Poisson. diagonal and block_diagonal
    // get a ROW-SCALED matrix, the case they exist for (the plain stencil has a
    // constant diagonal, which a diagonal preconditioner cannot improve).
    constexpr bool spd = Which == 3 || Which == 4;
    // ic_0 and ildl have a quadratic-cost apply (see krylov), and the check runs
    // CG with them, so they are capped at a 64^2 grid.
    const std::size_t kk = spd ? std::min<std::size_t>(k, 64) : k;
    struct S { csr A; vec_t b; };
    auto s = std::make_shared<S>();
    if constexpr (spd) s->A = sparse_entries::poisson2d(kk);
    else if constexpr (Which == 0 || Which == 6) s->A = row_scaled(sparse_entries::convdiff2d(kk), 311);
    else s->A = sparse_entries::convdiff2d(kk);
    s->b = vec_of<double>(s->A.num_rows(), 312);
    auto build = [s] {
        if constexpr (Which == 0) return itl::pc::diagonal<csr>(s->A);
        else if constexpr (Which == 1) return itl::pc::ilu_0<double>(s->A);
        else if constexpr (Which == 2) return itl::pc::ilut<double>(s->A);
        else if constexpr (Which == 3) return itl::pc::ic_0<double>(s->A);
        else if constexpr (Which == 4) return itl::pc::ildl<double>(s->A);
        else if constexpr (Which == 5) return itl::pc::ssor<csr>(s->A);
        else return itl::pc::block_diagonal<csr>(s->A, 4);
    };
    using PC = decltype(build());
    auto slot = std::make_shared<std::unique_ptr<PC>>();
    problem p{std::to_string(kk) + "^2 grid" + (Which == 0 || Which == 6 ? ", rows scaled" : ""), 0.0, 0.0,
              [] {}, [build, slot] { *slot = std::make_unique<PC>(build()); }, {}};
    p.check = [s, slot] {
        // A diagonal preconditioner has an exact answer, b_i / a_ii (one
        // correctly rounded division): check it, since "useful" alone would
        // pass a diagonal that is merely close.
        std::size_t inexact = 0;
        if constexpr (Which == 0) {
            vec_t y(s->b.size());
            (*slot)->solve(y, s->b);
            for (std::size_t i = 0; i < y.size(); ++i) {
                double aii = 0;
                for (std::size_t q = s->A.ref_major()[i]; q < s->A.ref_major()[i + 1]; ++q)
                    if (s->A.ref_minor()[q] == i) aii = s->A.ref_data()[q];
                inexact += std::fabs(y(i) - s->b(i) / aii) > 2 * std::numeric_limits<double>::epsilon() * std::fabs(y(i));
            }
        }
        const int with = iterations_with<spd>(s->A, s->b, **slot);
        const int without = iterations_with<spd>(s->A, s->b, itl::pc::identity<csr>(s->A));
        check_result r;
        r.tolerance = 0.75;
        r.error = double(with) / double(without);
        r.what = std::string(spd ? "CG" : "GMRES(30)") + " iterations with / without: " +
                 std::to_string(with) + " / " + std::to_string(without);
        if (inexact) r.what += "; " + std::to_string(inexact) + " entries differ from b_i / a_ii";
        r.ok = with < (1 << 30) && r.error <= r.tolerance && inexact == 0;
        return r;
    };
    return p;
}

// -- smoothers -----------------------------------------------------------------

/// One forward sweep. The check reconstructs each row's defining splitting
/// from x0 and x1 and bounds its residual by gamma_(nnz+4) times the magnitudes.
template <int Which>
problem smoother_sweep(size_class c) {
    const std::size_t k = sparse_entries::grid_kernel(c);
    const double omega = 1.5;
    struct S { csr A; vec_t b, x0, x; };
    auto s = std::make_shared<S>();
    s->A = sparse_entries::poisson2d(k);
    const std::size_t n = s->A.num_rows();
    s->b = vec_of<double>(n, 321);
    s->x0 = vec_of<double>(n, 322);
    s->x = vec_t(n);
    std::function<void()> run;
    if constexpr (Which == 0) {
        auto sm = std::make_shared<itl::smoother::jacobi<csr>>(s->A);
        run = [s, sm] { (*sm)(s->x, s->b); };
    } else if constexpr (Which == 1) {
        auto sm = std::make_shared<itl::smoother::gauss_seidel<csr>>(s->A);
        run = [s, sm] { (*sm)(s->x, s->b); };
    } else {
        auto sm = std::make_shared<itl::smoother::sor<csr>>(s->A, omega);
        run = [s, sm] { (*sm)(s->x, s->b); };
    }
    problem p{sparse_entries::grid_shape(k, s->A), 2.0 * s->A.nnz(), sparse_entries::sparse_bytes(s->A, 3),
              [s] { s->x = s->x0; }, run, {}};
    p.check = [s, n, omega] {
        error_tracker e(tol<double>(sparse_entries::max_row_nnz(s->A) + 4), "per-row splitting residual");
        const auto& st = s->A.ref_major();
        const auto& ix = s->A.ref_minor();
        const auto& v = s->A.ref_data();
        for (std::size_t i = 0; i < n; ++i) {
            long double lhs = 0, mag = std::fabs((long double)s->b(i));
            for (std::size_t q = st[i]; q < st[i + 1]; ++q) {
                const std::size_t j = ix[q];
                const long double a = v[q];
                long double t;
                if (j == i) {
                    // Jacobi / GS: a_ii x1_i. SOR: a_ii/w x1_i - (1/w - 1) a_ii x0_i.
                    t = Which == 2 ? a / omega * s->x(i) - (1.0L / omega - 1.0L) * a * s->x0(i) : a * s->x(i);
                    mag += std::fabs(a / (Which == 2 ? omega : 1.0) * s->x(i)) +
                           (Which == 2 ? std::fabs((1.0L / omega - 1.0L) * a * s->x0(i)) : 0.0L);
                } else {
                    const bool updated = Which != 0 && j < i;   // GS and SOR read x1 below the diagonal
                    t = a * (updated ? s->x(j) : s->x0(j));
                    mag += std::fabs(t);
                }
                lhs += t;
            }
            e.add(lhs, s->b(i), mag);
        }
        return e.result();
    };
    return p;
}

// -- multigrid -----------------------------------------------------------------

/// 2D transfer as the Kronecker product of the 1D one with itself: rows and
/// columns indexed iy * n + ix, as laplacian_2d orders its unknowns.
inline csr kron_self(const csr& T) {
    const std::size_t r1 = T.num_rows(), c1 = T.num_cols();
    csr K(r1 * r1, c1 * c1);
    {
        mat::inserter<csr> ins(K);
        const auto& st = T.ref_major();
        const auto& ix = T.ref_minor();
        const auto& v = T.ref_data();
        for (std::size_t ry = 0; ry < r1; ++ry)
            for (std::size_t qy = st[ry]; qy < st[ry + 1]; ++qy)
                for (std::size_t rx = 0; rx < r1; ++rx)
                    for (std::size_t qx = st[rx]; qx < st[rx + 1]; ++qx)
                        ins[ry * r1 + rx][ix[qy] * c1 + ix[qx]] << v[qy] * v[qx];
    }
    return K;
}

/// Geometric multigrid on the 2D Poisson problem: k = 2^m - 1 points per side,
/// full-weighting restriction and bilinear prolongation (tensor products of the
/// library's 1D operators), Galerkin coarse operators R A P, Gauss-Seidel
/// V(1,1), a sparse Cholesky solve on the coarsest (7 x 7) grid. Timed: a
/// fixed 10 V-cycles from x = 0.
inline problem multigrid_vcycle(size_class c) {
    const std::size_t k = c == size_class::small ? 63 : c == size_class::medium ? 255 : 511;
    constexpr int cycles = 10;
    using mg_t = itl::mg::multigrid<double>;
    struct S {
        std::unique_ptr<mg_t> mg;
        csr A;
        vec_t b, x;
        double work = 0;
    };
    auto s = std::make_shared<S>();
    std::vector<csr> levels{sparse_entries::poisson2d(k)}, R, P;
    for (std::size_t kk = k; kk > 7; kk = (kk - 1) / 2) {
        R.push_back(kron_self(itl::mg::make_restriction_1d(kk)));
        P.push_back(kron_self(itl::mg::make_prolongation_1d((kk - 1) / 2)));
        levels.push_back(mtl::spgemm(R.back(), mtl::spgemm(levels.back(), P.back())));
    }
    // One cycle: 2 sweeps + a residual on every level but the coarsest (3
    // passes over A_l at 2 flops/nnz), plus a restriction and a prolongation.
    for (std::size_t l = 0; l + 1 < levels.size(); ++l)
        s->work += 3.0 * 2.0 * levels[l].nnz() + 2.0 * R[l].nnz() + 2.0 * P[l].nnz();
    s->work *= cycles;
    auto coarse = std::make_shared<decltype(mtl::sparse::factorization::sparse_cholesky_numeric(
        levels.back(), mtl::sparse::factorization::sparse_cholesky_symbolic(levels.back())))>(
        mtl::sparse::factorization::sparse_cholesky_numeric(
            levels.back(), mtl::sparse::factorization::sparse_cholesky_symbolic(levels.back())));
    s->A = levels.front();
    const std::size_t n = s->A.num_rows();
    const std::size_t nlev = levels.size();
    s->mg = std::make_unique<mg_t>(
        levels, R, P, [](const csr& A) { return itl::smoother::gauss_seidel<csr>(A); },
        [coarse](vec_t& x, const vec_t& b) { coarse->solve(x, b); }, 1, 1);
    s->b = vec_of<double>(n, 331);
    s->x = vec_t(n);
    problem p{std::to_string(k) + "^2 grid, " + std::to_string(nlev) + " levels, " + std::to_string(cycles) +
                  " V(1,1)",
              s->work, sparse_entries::sparse_bytes(s->A, 3) * 3.0 * cycles * 4.0 / 3.0,
              [s] { for (std::size_t i = 0; i < s->x.size(); ++i) s->x(i) = 0.0; },
              [s] { for (int cyc = 0; cyc < cycles; ++cyc) s->mg->vcycle(s->x, s->b); }, {}};
    p.check = [s] {
        check_result r;
        r.tolerance = 1e-6;
        r.error = double(true_relres(s->A, s->x, s->b));
        r.what = "true |b-Ax|/|b| after 10 V-cycles";
        r.ok = r.error <= r.tolerance;
        return r;
    };
    return p;
}

// -- sparse eigensolvers ---------------------------------------------------------

/// The analytic spectrum of the nx x ny Laplacian (s = 1) or convection-
/// diffusion (s = sqrt(1 - c^2)): 4 - 2 s (cos(i pi/(nx+1)) + cos(j pi/(ny+1))).
inline std::vector<double> analytic_spectrum(std::size_t nx, std::size_t ny, double s) {
    std::vector<double> ev;
    ev.reserve(nx * ny);
    const double pi = std::acos(-1.0);
    for (std::size_t i = 1; i <= nx; ++i)
        for (std::size_t j = 1; j <= ny; ++j)
            ev.push_back(4.0 - 2.0 * s * (std::cos(i * pi / (nx + 1)) + std::cos(j * pi / (ny + 1))));
    std::sort(ev.begin(), ev.end());
    return ev;
}

/// Max relative distance of each computed value to the nearest wanted analytic
/// one, and the largest imaginary part relative to the spectrum's scale.
template <typename V>
check_result against_spectrum(const V& got, const std::vector<double>& want, double scale, double tol,
                              const std::string& what) {
    check_result r;
    r.tolerance = tol;
    r.what = what;
    double worst = 0;
    std::vector<double> remaining = want;
    for (std::size_t q = 0; q < got.size(); ++q) {
        const auto z = got(q);
        const double re = double(std::real(std::complex<double>(z)));
        const double im = double(std::imag(std::complex<double>(z)));
        auto it = std::min_element(remaining.begin(), remaining.end(),
                                   [re](double a, double b) { return std::fabs(a - re) < std::fabs(b - re); });
        if (it == remaining.end()) { worst = 1e300; break; }
        // std::max drops a NaN argument, so a non-finite candidate must fail
        // explicitly rather than vanish.
        const double cand = std::max(std::fabs(*it - re), std::fabs(im)) / scale;
        worst = std::isfinite(cand) ? std::max(worst, cand) : 1e300;
        remaining.erase(it);   // each analytic value may be matched once
    }
    r.error = got.size() == want.size() ? worst : 1e300;
    r.ok = r.error <= r.tolerance;
    return r;
}

/// Bauer-Fike enclosure, the check for a Krylov eigensolver that may not have
/// converged: for ANY approximate pair (theta, y), some eigenvalue lies within
/// kappa(V) |A y - theta y| / |y| of theta (kappa = 1 for a symmetric A). A
/// wrong value or vector fails it; an honest unconverged one passes with its
/// error bar. Each pair is matched to its nearest analytic eigenvalue; the
/// metric also reports the distance itself.
template <typename Values, typename Vectors>
check_result enclosure(const csr& A, const Values& vals, const Vectors& Y, const std::vector<double>& spectrum,
                       double kappa, std::size_t wanted, const std::string& what) {
    using std::real;
    using std::imag;
    const std::size_t n = A.num_rows();
    check_result r;
    r.tolerance = 1.0;   // error = distance / (kappa |r| + tiny floor); must be <= 1
    r.what = what;
    double worst = 0, dist_max = 0;
    std::vector<long double> yr(n), yi(n), ar, ai, tmp;
    for (std::size_t q = 0; q < vals.size(); ++q) {
        const std::complex<double> th(vals(q));
        long double y2 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::complex<double> y(Y(i, q));
            yr[i] = y.real();
            yi[i] = y.imag();
            y2 += yr[i] * yr[i] + yi[i] * yi[i];
        }
        sparse_entries::ref_mult(A, yr, ar, tmp);
        sparse_entries::ref_mult(A, yi, ai, tmp);
        long double r2 = 0;
        for (std::size_t i = 0; i < n; ++i) {   // (A y - theta y), complex
            const long double rr = ar[i] - (th.real() * yr[i] - th.imag() * yi[i]);
            const long double ri = ai[i] - (th.real() * yi[i] + th.imag() * yr[i]);
            r2 += rr * rr + ri * ri;
        }
        const double radius = double(kappa * std::sqrt(r2 / y2));
        double d = 1e300;
        for (double lam : spectrum) d = std::min(d, std::abs(std::complex<double>(lam) - th));
        dist_max = std::max(dist_max, d);
        const double floor = 64.0 * std::numeric_limits<double>::epsilon() * spectrum.back();
        // A zero or NaN vector makes the radius NaN; std::max would drop it.
        const double ratio = d / (radius + floor);
        worst = std::isfinite(ratio) ? std::max(worst, ratio) : 1e300;
    }
    r.error = vals.size() == wanted ? worst : 1e300;
    r.what += "; max distance to the analytic spectrum " + std::to_string(dist_max);
    r.ok = r.error <= r.tolerance;
    return r;
}

/// What an enclosure cannot see: a value moved along with its error bar. Two
/// rigorous UPPER bounds close that hole --
///   Lanczos / power iteration (Rayleigh-Ritz on a symmetric A): the i-th
///     largest Ritz value is at most the i-th largest eigenvalue (Cauchy
///     interlacing; for power iteration, i = 1: a Rayleigh quotient <= lambda_max);
///   Arnoldi: Ritz values lie in the numerical range, so Re(theta) is at most
///     lambda_max of the symmetric part (A + A^T)/2 -- here exactly the Laplacian.
/// -- and an accuracy FLOOR against the analytic top eigenvalues: at most `rel`
/// of lambda_max below them (2%: what Krylov dimension 80 attains on a clustered
/// spectrum, where convergence is slow, not wrong).
template <typename Values>
void top_bounds(check_result& r, const Values& vals, std::vector<double> upper, const std::vector<double>& top,
                double rel) {
    std::vector<double> got;
    for (std::size_t q = 0; q < vals.size(); ++q) got.push_back(std::real(std::complex<double>(vals(q))));
    std::sort(got.begin(), got.end(), std::greater<>());
    std::sort(upper.begin(), upper.end(), std::greater<>());
    std::vector<double> want = top;
    std::sort(want.begin(), want.end(), std::greater<>());
    const double scale = want.front();
    bool bad = got.size() != want.size();
    for (double g : got)   // every comparison with NaN is false: reject it outright
        if (!std::isfinite(g)) { bad = true; r.what += "; a Ritz value is not finite"; }
    for (std::size_t i = 0; !bad && i < got.size(); ++i) {
        if (got[i] > upper[i] * (1.0 + 1e-12)) { bad = true; r.what += "; a Ritz value exceeds its upper bound"; }
        else if (want[i] - got[i] > rel * scale) { bad = true; r.what += "; a Ritz value is far below the analytic top"; }
    }
    if (bad) { r.ok = false; r.error = std::max(r.error, 2.0 * r.tolerance + 1.0); }
}

/// Lanczos, 4 largest eigenvalues of the symmetric Laplacian, Krylov dimension
/// 80. The top of a Laplacian's spectrum clusters, so 80 does not converge on
/// the larger grids (nothing affordable does: 320 still has not at 128^2);
/// the enclosure check holds either way.
inline problem lanczos(size_class c) {
    const std::size_t nx = grid_eig(c), ny = nx + 1;
    struct S { csr A; vec_t v0; itl::ritz_pairs<double> r; };
    auto s = std::make_shared<S>();
    s->A = mtl::generators::laplacian_2d<double>(nx, ny);
    s->v0 = vec_of<double>(s->A.num_rows(), 341);
    problem p{std::to_string(nx) + "x" + std::to_string(ny) + " Laplacian, k=4, m=80", 0.0, 0.0, [] {},
              [s] { s->r = itl::lanczos(s->A, s->v0, 4, itl::eigen_which::largest_algebraic, 80); }, {}};
    p.check = [s, nx, ny] {
        const auto all = analytic_spectrum(nx, ny, 1.0);
        auto r = enclosure(s->A, s->r.values, s->r.vectors, all, 1.0, 4,
                           "Bauer-Fike enclosure, interlacing bound, 2% accuracy floor");
        const std::vector<double> top(all.end() - 4, all.end());
        top_bounds(r, s->r.values, top, top, 0.02);
        return r;
    };
    return p;
}

/// Arnoldi, 4 largest-magnitude eigenvalues of an UNSYMMETRIC operator with a
/// known real spectrum, Krylov dimension 80. A = D^-1 S D with D = diag(r^(ix/2)
/// r^(iy/2)), r = (1+c)/(1-c), so kappa(V) <= kappa(D) = r^((nx-1+ny-1)/2);
/// c = 0.01 keeps it near 13 on the largest grid.
inline problem arnoldi(size_class c) {
    const std::size_t nx = grid_eig(c), ny = nx + 1;
    const double cc = 0.01;
    struct S { csr A; vec_t v0; itl::ritz_pairs<std::complex<double>> r; };
    auto s = std::make_shared<S>();
    s->A = convdiff(nx, ny, cc);
    s->v0 = vec_of<double>(s->A.num_rows(), 342);
    problem p{std::to_string(nx) + "x" + std::to_string(ny) + " convection-diffusion, k=4, m=80", 0.0, 0.0,
              [] {}, [s] { s->r = itl::arnoldi(s->A, s->v0, 4, itl::eigen_which::largest_magnitude, 80); }, {}};
    p.check = [s, nx, ny, cc] {
        const double kappa = std::pow((1 + cc) / (1 - cc), 0.5 * double(nx - 1 + ny - 1));
        const auto all = analytic_spectrum(nx, ny, std::sqrt(1.0 - cc * cc));
        auto r = enclosure(s->A, s->r.values, s->r.vectors, all, kappa, 4,
                           "Bauer-Fike enclosure (kappa(V) <= kappa(D)), numerical-range bound, 2% floor");
        const double hmax = analytic_spectrum(nx, ny, 1.0).back();   // lambda_max of (A + A^T)/2
        top_bounds(r, s->r.values, std::vector<double>(4, hmax), std::vector<double>(all.end() - 4, all.end()),
                   0.02);
        return r;
    };
    return p;
}

/// Shift-invert about sigma = 0, Krylov dimension 40: the 4 smallest
/// eigenvalues. Inversion separates them well, and it CONVERGES on every grid
/// (to ~1e-13), so this check requires convergence and the analytic values.
inline problem shift_invert(size_class c) {
    const std::size_t nx = grid_eig(c), ny = nx + 1;
    struct S { csr A; itl::ritz_pairs<std::complex<double>> r; };
    auto s = std::make_shared<S>();
    s->A = mtl::generators::laplacian_2d<double>(nx, ny);
    problem p{std::to_string(nx) + "x" + std::to_string(ny) + " Laplacian, k=4 nearest 0, m=40", 0.0, 0.0,
              [] {}, [s] { s->r = mtl::sparse::sparse_eigs_shift_invert(s->A, 0.0, 4, 40); }, {}};
    p.check = [s, nx, ny] {
        auto all = analytic_spectrum(nx, ny, 1.0);
        std::vector<double> bottom(all.begin(), all.begin() + 4);
        auto r = against_spectrum(s->r.values, bottom, all.front(), 1e-10,
                                  "converged, and values vs analytic smallest, rel to lambda_min");
        if (!s->r.converged) { r.ok = false; r.what += " (did not report convergence)"; }
        return r;
    };
    return p;
}

/// Power iteration, 300 steps: not converged on a clustered Laplacian top, so
/// the check is the Bauer-Fike enclosure of the returned (value, vector).
inline problem power_iteration(size_class c) {
    const std::size_t nx = grid_eig(c), ny = nx + 1;
    struct S { csr A; vec_t v0; itl::eigenpair<double> r; };
    auto s = std::make_shared<S>();
    s->A = mtl::generators::laplacian_2d<double>(nx, ny);
    s->v0 = vec_of<double>(s->A.num_rows(), 343);
    problem p{std::to_string(nx) + "x" + std::to_string(ny) + " Laplacian, 300 steps", 0.0, 0.0, [] {},
              [s] { s->r = itl::power_iteration(s->A, s->v0, 300, 1e-14); }, {}};
    p.check = [s, nx, ny] {
        vec_t val(1);
        val(0) = s->r.value;
        mat::dense2D<double> Y(s->A.num_rows(), 1);
        for (std::size_t i = 0; i < s->A.num_rows(); ++i) Y(i, 0) = s->r.vector(i);
        const auto all = analytic_spectrum(nx, ny, 1.0);
        auto r = enclosure(s->A, val, Y, all, 1.0, 1, "Bauer-Fike enclosure, value <= lambda_max, 10% floor");
        // 300 steps on a clustered top: the Rayleigh quotient is within 10%.
        top_bounds(r, val, {all.back()}, {all.back()}, 0.10);
        return r;
    };
    return p;
}

inline void register_all(registry& r) {
    r.add("krylov", "cg+ilu0", "f64", krylov<0>);
    r.add("krylov", "minres+diag", "f64", krylov<1>);
    r.add("krylov", "bicg+ilu0", "f64", krylov<2>);
    r.add("krylov", "bicgstab+ilu0", "f64", krylov<3>);
    r.add("krylov", "bicgstab_ell+ilu0", "f64", krylov<4>);
    r.add("krylov", "cgs+ilu0", "f64", krylov<5>);
    r.add("krylov", "gmres30+ilu0", "f64", krylov<6>);
    r.add("krylov", "qmr+ilu0", "f64", krylov<7>);
    r.add("krylov", "tfqmr+ilu0", "f64", krylov<8>);
    r.add("krylov", "idr_s1+ilu0", "f64", krylov<10>);
    r.add("krylov", "idr_s2+ilu0", "f64", krylov<11>);
    r.add("krylov", "idr_s4+ilu0", "f64", krylov<9>);
    r.add("krylov", "idr_s8+ilu0", "f64", krylov<12>);

    r.add("pc", "diagonal", "f64", pc_setup<0>);
    r.add("pc", "ilu_0", "f64", pc_setup<1>);
    r.add("pc", "ilut", "f64", pc_setup<2>);
    r.add("pc", "ic_0", "f64", pc_setup<3>);
    r.add("pc", "ildl", "f64", pc_setup<4>);
    r.add("pc", "ssor", "f64", pc_setup<5>);
    r.add("pc", "block_diagonal4", "f64", pc_setup<6>);

    r.add("smoother", "jacobi", "f64", smoother_sweep<0>);
    r.add("smoother", "gauss_seidel", "f64", smoother_sweep<1>);
    r.add("smoother", "sor1.5", "f64", smoother_sweep<2>);

    r.add("mg", "poisson2d_vcycle", "f64", multigrid_vcycle);

    r.add("sparse_eig", "lanczos", "f64", lanczos);
    r.add("sparse_eig", "arnoldi", "f64", arnoldi);
    r.add("sparse_eig", "shift_invert", "f64", shift_invert);
    r.add("sparse_eig", "power_iteration", "f64", power_iteration);
}

} // namespace mtl::bench::platform::iterative
