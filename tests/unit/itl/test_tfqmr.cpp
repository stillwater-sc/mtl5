#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/tfqmr.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ilu_0.hpp>

#include <cmath>
#include <cstddef>

using namespace mtl;

TEST_CASE("TFQMR: dense 3x3 non-symmetric system", "[itl][krylov][tfqmr]") {
    // Genuinely nonsymmetric (#731: this matrix used to be symmetric).
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 0;
    A(1,0) = -1; A(1,1) = 4; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = -2; A(2,2) = 4;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::tfqmr(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("TFQMR: sparse tridiagonal system", "[itl][krylov][tfqmr]") {
    const std::size_t n = 20;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0)     ins[i][i-1] << -1.0;
            if (i < n - 1) ins[i][i+1] << -1.0;
        }
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 500, 1e-10);

    int err = itl::tfqmr(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

// -- Unsymmetric systems (#731) ------------------------------------------------
//
// TFQMR used to be missing its u recurrence: w stood in for u_{m+1} and for
// u_{m+2}. It stalled at a 5-9% residual on unsymmetric systems with any
// preconditioner, and on SPD ones took 10-16x BiCGSTAB's iterations. Both
// tests above were symmetric, so nothing caught it. These run the 5-point
// convection-diffusion operator from the issue, check the TRUE residual
// b - A x (TFQMR's own convergence test is on the quasi-residual tau), and
// bound the iteration count against BiCGSTAB on the same system, so a
// degraded variant cannot pass by converging slowly.

namespace {

using csr = mat::compressed2D<double>;
using vec_t  = vec::dense_vector<double>;

/// 5-point stencil on a g x g grid, diagonal 4: the issue's convection-
/// diffusion matrix for (west/south, east/north) = (-1.4, -0.6), the
/// Poisson matrix for (-1, -1).
csr stencil(std::size_t g, double ws, double en) {
    csr A(g * g, g * g);
    mat::inserter<csr> ins(A);
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            const std::size_t i = r * g + c;
            ins[i][i] << 4.0;
            if (c > 0)     ins[i][i - 1] << ws;
            if (c + 1 < g) ins[i][i + 1] << en;
            if (r > 0)     ins[i][i - g] << ws;
            if (r + 1 < g) ins[i][i + g] << en;
        }
    return A;
}

vec_t rhs(std::size_t n) {
    vec_t b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = std::sin(0.37 * double(i)) + 0.2;
    return b;
}

/// || b - A x || / || b ||, recomputed.
double true_residual(const csr& A, const vec_t& x, const vec_t& b) {
    vec_t Ax = A * x;
    double r = 0, nb = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        r  += (b(i) - Ax(i)) * (b(i) - Ax(i));
        nb += b(i) * b(i);
    }
    return std::sqrt(r / nb);
}

struct outcome { int err, iters; double resid; };

template <typename PC>
outcome run_tfqmr(const csr& A, const vec_t& b, const PC& M, double rtol) {
    vec_t x(b.size(), 0.0);
    itl::basic_iteration<double> iter(b, 3000, rtol);
    const int err = itl::tfqmr(A, x, b, M, iter);
    return {err, iter.iterations(), true_residual(A, x, b)};
}

template <typename PC>
int bicgstab_iterations(const csr& A, const vec_t& b, const PC& M, double rtol) {
    vec_t x(b.size(), 0.0);
    itl::basic_iteration<double> iter(b, 3000, rtol);
    REQUIRE(itl::bicgstab(A, x, b, M, iter) == 0);
    return iter.iterations();
}

template <typename PC>
void check_against_bicgstab(const csr& A, const PC& M) {
    const vec_t b = rhs(A.num_rows());
    const double rtol = 1e-8;
    const outcome t = run_tfqmr(A, b, M, rtol);
    const int bi = bicgstab_iterations(A, b, M, rtol);
    INFO("TFQMR " << t.iters << " its, true residual " << t.resid << "; BiCGSTAB " << bi << " its");
    REQUIRE(t.err == 0);
    CHECK(t.resid <= 10 * rtol);
    CHECK(t.iters <= 3 * bi);
}

} // namespace

TEST_CASE("TFQMR: unsymmetric convection-diffusion converges with any preconditioner (#731)",
          "[itl][krylov][tfqmr][731]") {
    const csr A = stencil(32, -1.4, -0.6);
    SECTION("identity") { check_against_bicgstab(A, itl::pc::identity<csr>(A)); }
    SECTION("diagonal") { check_against_bicgstab(A, itl::pc::diagonal<csr>(A)); }
    SECTION("ILU(0)")   { check_against_bicgstab(A, itl::pc::ilu_0<double>(A)); }
}

TEST_CASE("TFQMR: SPD Poisson within 3x BiCGSTAB's iterations (#731)", "[itl][krylov][tfqmr][731]") {
    const csr A = stencil(32, -1.0, -1.0);
    SECTION("identity") { check_against_bicgstab(A, itl::pc::identity<csr>(A)); }
    SECTION("ILU(0)")   { check_against_bicgstab(A, itl::pc::ilu_0<double>(A)); }
}

TEST_CASE("TFQMR: the reported convergence is the true residual's (#731)", "[itl][krylov][tfqmr][731]") {
    // Unpreconditioned, the recursively updated residual drifts from b - A x
    // by rounding and the true residual stalls near 2e-7, as CGS's does; the
    // solver checks the true residual when tau converges and restarts from
    // the current x if it has not. So for every tolerance, a reported
    // convergence must be real.
    const csr A = stencil(32, -1.4, -0.6);
    const vec_t b = rhs(A.num_rows());
    const itl::pc::identity<csr> M(A);
    for (const double rtol : {1e-6, 1e-8, 1e-10, 1e-12}) {
        const outcome t = run_tfqmr(A, b, M, rtol);
        INFO("rtol " << rtol << ": " << t.iters << " its, true residual " << t.resid);
        REQUIRE(t.err == 0);
        CHECK(t.resid <= 2 * rtol);
    }
}

TEST_CASE("TFQMR: the Accumulator path solves the unsymmetric system (#731)", "[itl][krylov][tfqmr][731]") {
    const csr A = stencil(16, -1.4, -0.6);
    const vec_t b = rhs(A.num_rows());
    const itl::pc::ilu_0<double> M(A);
    vec_t x(b.size(), 0.0);
    itl::basic_iteration<double> iter(b, 500, 1e-10);
    REQUIRE(itl::tfqmr<long double>(A, x, b, M, iter) == 0);
    CHECK(true_residual(A, x, b) <= 1e-9);
}

TEST_CASE("TFQMR: an iteration limit reached is reported, not hidden by the restart (#731)",
          "[itl][krylov][tfqmr][731]") {
    const csr A = stencil(32, -1.4, -0.6);
    const vec_t b = rhs(A.num_rows());
    vec_t x(b.size(), 0.0);
    itl::basic_iteration<double> iter(b, 10, 1e-12);
    CHECK(itl::tfqmr(A, x, b, itl::pc::identity<csr>(A), iter) == 1);
    CHECK(iter.iterations() == 10);
}
