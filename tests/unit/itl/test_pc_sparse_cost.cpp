// The incomplete factorizations cost O(nnz), not O(n * nnz) or O(n^2)
// (#732, #733):
//   - ic_0 and ildl apply: the L^T back substitution gathers from a transposed
//     copy of L, where it used to scan every later row for each i;
//   - ilu_0 and ilut setup: each row eliminates only its own nonzero columns,
//     where it used to visit every column k < i.
//
// The test times each on the 2D 5-point stencil at n = 64^2 and 256^2, a 16x
// step in n and in nnz. Linear work takes ~16x as long; the old quadratic
// loops took ~256x. The bound is 64x, a factor of 4 from either, so the test
// does not depend on the machine, the build type or a noisy runner. Each time
// is the best of several runs.
//
// The results themselves are unchanged bit for bit -- each change keeps every
// entry's terms and their order -- which the existing preconditioner tests
// (itl_test_pc_accumulator, itl_test_preconditioner_adjoint, the ilu/ic and
// advanced_pc tests) pin. This file adds the residual checks at a size where
// the old loops would have been slow.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>

#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/itl/pc/ildl.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ilut.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

using sparse = mat::compressed2D<double>;
using vec_t  = vec::dense_vector<double>;

/// 2D 5-point stencil on a g x g grid: the Poisson matrix for conv = 0, a
/// nonsymmetric convection-diffusion one otherwise.
sparse stencil(std::size_t g, double conv) {
    const std::size_t n = g * g;
    sparse A(n, n);
    mat::inserter<sparse> ins(A);
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            const std::size_t i = r * g + c;
            ins[i][i] << 4.0;
            if (c > 0)     ins[i][i - 1] << -1.0 - conv;
            if (c + 1 < g) ins[i][i + 1] << -1.0 + conv;
            if (r > 0)     ins[i][i - g] << -1.0;
            if (r + 1 < g) ins[i][i + g] << -1.0;
        }
    return A;
}

template <typename F>
double best_of(int runs, F f) {
    double best = 1e30;
    for (int r = 0; r < runs; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

constexpr std::size_t g_small = 64, g_large = 256;   // n = 4096 and 65536
constexpr double max_ratio = 64.0;                    // linear: ~16; quadratic: ~256

/// Time of one apply of preconditioner P on the Poisson matrix of grid g.
template <typename P>
double apply_time(std::size_t g) {
    const sparse A = stencil(g, 0.0);
    const P M(A);
    vec_t b(g * g, 1.0), x(g * g);
    return best_of(5, [&] { M.solve(x, b); M.adjoint_solve(x, b); });
}

/// Time to construct preconditioner P on the convection-diffusion matrix.
template <typename P>
double setup_time(std::size_t g) {
    const sparse A = stencil(g, 0.4);
    return best_of(3, [&] { const P M(A); (void)M; });
}

/// || A x - b ||_inf / || b ||_inf after x = M^-1 b, for an exact-pattern check.
double relative_residual(const sparse& A, const vec_t& x, const vec_t& b) {
    const auto& st = A.ref_major();
    const auto& ix = A.ref_minor();
    const auto& dv = A.ref_data();
    double r = 0, nb = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double s = 0;
        for (auto p = st[i]; p < st[i + 1]; ++p) s += dv[p] * x(ix[p]);
        r = std::max(r, std::fabs(s - b(i)));
        nb = std::max(nb, std::fabs(b(i)));
    }
    return r / nb;
}

} // namespace

TEST_CASE("ic_0 and ildl apply in O(nnz) (#732)", "[itl][pc][732]") {
    const double ic_ratio = apply_time<itl::pc::ic_0<double>>(g_large) / apply_time<itl::pc::ic_0<double>>(g_small);
    const double ld_ratio = apply_time<itl::pc::ildl<double>>(g_large) / apply_time<itl::pc::ildl<double>>(g_small);
    INFO("apply time ratio for a 16x larger n: ic_0 " << ic_ratio << ", ildl " << ld_ratio);
    CHECK(ic_ratio < max_ratio);
    CHECK(ld_ratio < max_ratio);
}

TEST_CASE("ilu_0 and ilut set up in O(nnz) (#733)", "[itl][pc][733]") {
    const double ilu_ratio = setup_time<itl::pc::ilu_0<double>>(g_large) / setup_time<itl::pc::ilu_0<double>>(g_small);
    const double ilut_ratio = setup_time<itl::pc::ilut<double>>(g_large) / setup_time<itl::pc::ilut<double>>(g_small);
    INFO("setup time ratio for a 16x larger n: ilu_0 " << ilu_ratio << ", ilut " << ilut_ratio);
    CHECK(ilu_ratio < max_ratio);
    CHECK(ilut_ratio < max_ratio);
}

TEST_CASE("the incomplete factorizations are exact on a tridiagonal at large n (#732, #733)",
          "[itl][pc][732][733]") {
    // A tridiagonal has no fill, so IC(0), ILDL, ILU(0) and a no-drop ILUT are
    // exact factorizations: M^-1 b solves A x = b to rounding. At n = 65536 the
    // old back substitution and setup took seconds; this checks the new paths
    // give the exact solve there, through solve() and, A being symmetric,
    // adjoint_solve().
    const std::size_t n = g_large * g_large;
    sparse A(n, n);
    {
        mat::inserter<sparse> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0)     ins[i][i - 1] << -1.0;
            if (i + 1 < n) ins[i][i + 1] << -1.0;
        }
    }
    vec_t b(n), x(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = std::sin(0.01 * double(i)) + 0.5;

    const itl::pc::ic_0<double> ic(A);
    const itl::pc::ildl<double> ld(A);
    const itl::pc::ilu_0<double> ilu(A);
    const itl::pc::ilut<double> ilut(A, 4, 0.0);
    ic.solve(x, b);           CHECK(relative_residual(A, x, b) < 1e-13);
    ic.adjoint_solve(x, b);   CHECK(relative_residual(A, x, b) < 1e-13);
    ld.solve(x, b);           CHECK(relative_residual(A, x, b) < 1e-13);
    ld.adjoint_solve(x, b);   CHECK(relative_residual(A, x, b) < 1e-13);
    ilu.solve(x, b);          CHECK(relative_residual(A, x, b) < 1e-13);
    ilut.solve(x, b);         CHECK(relative_residual(A, x, b) < 1e-13);
}
