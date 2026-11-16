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
#include <mtl/itl/krylov/idr_s.hpp>
#include <mtl/itl/pc/ilu_0.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mtl/generators/laplacian.hpp>

using namespace mtl;

TEST_CASE("IDR(s): dense 3x3 system", "[itl][krylov][idr_s]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 4; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 4;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::idr_s(A, x, b, pc, iter, 2);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("IDR(s): sparse tridiagonal with s=4", "[itl][krylov][idr_s]") {
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

    int err = itl::idr_s(A, x, b, pc, iter, 4);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

// -- One omega per cycle (#734) ------------------------------------------------
//
// IDR(s) runs in cycles of s + 1 steps that must share one omega: that is
// what puts their residuals in the same space G_j and gives the dimension
// reduction. The solver used to recompute omega at every step. Its iteration
// counts were then irregular in s, and it diverged on some problems: on the
// unpreconditioned 32x32 Poisson matrix, s = 1 hit a 5000-iteration cap, and
// s = 8 took 242 iterations to s = 4's 147. These pin both: every s
// converges to a true residual within the tolerance, and more shadow space
// does not cost more iterations.

namespace {

using csr   = mat::compressed2D<double>;
using vec_t = vec::dense_vector<double>;

/// 5-point stencil on a g x g grid, diagonal 4: the convection-diffusion
/// matrix of mtl5_bench for (west/south, east/north) = (-1.4, -0.6), the
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

/// Iterations IDR(s) takes to rtol 1e-8 from x = 0; fails the test unless
/// it converges with a true residual within 10x the tolerance.
template <typename PC>
int idr_iterations(const csr& A, const PC& M, std::size_t s) {
    const std::size_t n = A.num_rows();
    vec_t b(n), x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) b(i) = std::sin(0.37 * double(i)) + 0.2;
    itl::basic_iteration<double> iter(b, 3000, 1e-8);
    const int err = itl::idr_s(A, x, b, M, iter, s);

    vec_t Ax = A * x;
    double r = 0, nb = 0;
    for (std::size_t i = 0; i < n; ++i) {
        r  += (b(i) - Ax(i)) * (b(i) - Ax(i));
        nb += b(i) * b(i);
    }
    INFO("s = " << s << ": " << iter.iterations() << " iterations, true residual " << std::sqrt(r / nb));
    REQUIRE(err == 0);
    REQUIRE(std::sqrt(r / nb) <= 1e-7);
    return iter.iterations();
}

/// Iterations for s = 1, 2, 4, 8; each must be at most 10% above the
/// smallest count for any smaller s, and s = 8 must not exceed s = 1.
template <typename PC>
void check_monotone_in_s(const csr& A, const PC& M) {
    std::vector<int> its;
    for (const std::size_t s : {1u, 2u, 4u, 8u}) its.push_back(idr_iterations(A, M, s));
    INFO("iterations at s = 1, 2, 4, 8: " << its[0] << ", " << its[1] << ", " << its[2] << ", " << its[3]);
    CHECK(its[3] <= its[0]);
    for (std::size_t k = 1; k < its.size(); ++k) {
        const int best_before = *std::min_element(its.begin(), its.begin() + static_cast<std::ptrdiff_t>(k));
        CHECK(double(its[k]) <= 1.1 * double(best_before));
    }
}

} // namespace

TEST_CASE("IDR(s): more shadow space does not cost more iterations (#734)", "[itl][krylov][idr_s][734]") {
    SECTION("convection-diffusion, ILU(0)") {
        const csr A = stencil(32, -1.4, -0.6);
        check_monotone_in_s(A, itl::pc::ilu_0<double>(A));
    }
    SECTION("convection-diffusion, unpreconditioned") {
        const csr A = stencil(32, -1.4, -0.6);
        check_monotone_in_s(A, itl::pc::identity<csr>(A));
    }
    SECTION("Poisson, unpreconditioned") {
        // s = 1 hit the iteration cap here, and s = 8 took 1.6x s = 4.
        const csr A = stencil(32, -1.0, -1.0);
        check_monotone_in_s(A, itl::pc::identity<csr>(A));
    }
}

TEST_CASE("IDR(s): s larger than one cycle's worth of steps to converge (#734)", "[itl][krylov][idr_s][734]") {
    // s = 16 at n = 64: the initial s steps and a partial cycle; the cycle
    // structure must stop at convergence mid-cycle.
    const csr A = stencil(8, -1.4, -0.6);
    const int its = idr_iterations(A, itl::pc::identity<csr>(A), 16);
    CHECK(its <= 64);
}

TEST_CASE("IDR(s): float solves the unsymmetric system", "[itl][krylov][idr_s][734]") {
    // The float instantiation, whose breakdown test once compared the pivot
    // with 1e-300 -- zero in float, so a zero pivot was never caught.
    //
    // rtol 1e-4, not tighter: in float the updated residual drifts from
    // b - A x, and the true residual levels off near 1e-4 here. At rtol 1e-5,
    // whether the updated residual gets there depends on code generation:
    // s = 1 and s = 4 ran to the iteration cap with FMA contraction
    // (x86-64-v3) and s = 8 without it, while the rest converged. At 1e-4
    // every s converges in 40-54 iterations, under both.
    using fcsr = mat::compressed2D<float>;
    using fvec = vec::dense_vector<float>;
    const std::size_t g = 16, n = g * g;
    fcsr A(n, n);
    {
        mat::inserter<fcsr> ins(A);
        for (std::size_t r = 0; r < g; ++r)
            for (std::size_t c = 0; c < g; ++c) {
                const std::size_t i = r * g + c;
                ins[i][i] << 4.0f;
                if (c > 0)     ins[i][i - 1] << -1.4f;
                if (c + 1 < g) ins[i][i + 1] << -0.6f;
                if (r > 0)     ins[i][i - g] << -1.4f;
                if (r + 1 < g) ins[i][i + g] << -0.6f;
            }
    }
    fvec b(n, 1.0f), x(n, 0.0f);
    itl::pc::identity<fcsr> M(A);
    for (const std::size_t s : {1u, 2u, 4u, 8u}) {
        for (std::size_t i = 0; i < n; ++i) x(i) = 0.0f;
        itl::basic_iteration<float> iter(b, 500, 1e-4f);
        INFO("s = " << s);
        REQUIRE(itl::idr_s(A, x, b, M, iter, s) == 0);
        fvec Ax = A * x;
        float r = 0, nb = 0;
        for (std::size_t i = 0; i < n; ++i) {
            r  += (b(i) - Ax(i)) * (b(i) - Ax(i));
            nb += b(i) * b(i);
        }
        // The true residual decides convergence (issue 874), so it is within
        // the tolerance; 2x for the test's own float residual.
        CHECK(std::sqrt(r / nb) <= 2e-4f);
    }
}

// -- The biorthogonal IDR(s) (issue 874) ---------------------------------------

namespace {

/// The 5-point convection-diffusion operator on a k x k grid with
/// convection eps (tests/regression/itl/test_krylov_family.cpp); eps = 0.9
/// is strongly nonnormal.
csr convdiff(std::size_t k, double eps) {
    return stencil(k, -1.0 - eps, -1.0 + eps);
}

double true_relres(const csr& A, const vec_t& x, const vec_t& b) {
    vec_t Ax = A * x;
    double r = 0, nb = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        r  += (b(i) - Ax(i)) * (b(i) - Ax(i));
        nb += b(i) * b(i);
    }
    return std::sqrt(r / nb);
}

struct idrs_outcome {
    int err;
    int iterations;
    double true_relres;
};

template <typename PC>
idrs_outcome solve_idrs(const csr& A, const PC& M, double rtol, const itl::idrs_options& opt, int max_iter = 20000) {
    const std::size_t n = A.num_rows();
    const vec_t b(n, 1.0);
    vec_t x(n, 0.0);
    itl::basic_iteration<double> iter(b, max_iter, rtol);
    const int err = itl::idr_s(A, x, b, M, iter, opt);
    return {err, iter.iterations(), true_relres(A, x, b)};
}

/// A controller that records each residual the solver tests.
struct recording : itl::basic_iteration<double> {
    using basic_iteration::basic_iteration;
    std::vector<double> seen;
    template <typename V>
    bool finished(const V& r) { return finished(mtl::two_norm(r)); }
    bool finished(double r) {
        seen.push_back(r);
        return basic_iteration::finished(r);
    }
};

} // namespace

TEST_CASE("IDR(s): every s converges to the true tolerance, also at 1e-13 (issue 874)", "[itl][krylov][idr_s][874]") {
    // The prototype IDR(s) on convection-diffusion with ILU(0) at 1e-13
    // (main at d3ef7b4): s = 2 took 729 iterations, s = 8 took 1612, and the
    // true residual was 5-35x the reported one (IDR(4): 1.3e-12).
    const csr A = convdiff(100, 0.3);
    const itl::pc::ilu_0<double> M(A);
    for (const double rtol : {1e-10, 1e-13}) {
        for (const std::size_t s : {1u, 2u, 4u, 8u, 16u}) {
            itl::idrs_options opt;
            opt.s = s;
            const auto o = solve_idrs(A, M, rtol, opt);
            INFO("rtol " << rtol << ", s = " << s << ": " << o.iterations << " iterations, true " << o.true_relres);
            CHECK(o.err == 0);
            CHECK(o.true_relres <= 1.5 * rtol);
            CHECK(o.iterations <= 150);
        }
    }
}

TEST_CASE("IDR(s): the true residual decides on nonnormal and Poisson systems (issue 874)", "[itl][krylov][idr_s][874]") {
    SECTION("convection-diffusion eps = 0.9, no PC") {
        // The recursive residual drifted: IDR(16) stopped at an updated 9e-11
        // whose true value was 2.5e-9 before residual replacement.
        const csr A = convdiff(100, 0.9);
        const itl::pc::identity<csr> M(A);
        for (const std::size_t s : {1u, 4u, 16u}) {
            itl::idrs_options opt;
            opt.s = s;
            const auto o = solve_idrs(A, M, 1e-10, opt);
            INFO("s = " << s << ": " << o.iterations << " iterations, true " << o.true_relres);
            CHECK(o.err == 0);
            CHECK(o.true_relres <= 1.5e-10);
            CHECK(o.iterations <= 500);
        }
    }
    SECTION("2D Laplacian, no PC") {
        // The prototype's IDR(16) returned converged at a true 3.0e-10 for
        // rtol 1e-10.
        const csr A = generators::laplacian_2d<double>(100, 100);
        const itl::pc::identity<csr> M(A);
        itl::idrs_options opt;
        opt.s = 16;
        const auto o = solve_idrs(A, M, 1e-10, opt);
        CHECK(o.err == 0);
        CHECK(o.true_relres <= 1.5e-10);
    }
}

TEST_CASE("IDR(s): a breakdown restarts, and a degenerate shadow space is redrawn (issue 874)", "[itl][krylov][idr_s][874]") {
    // A = diag(1, 2, ..., n), b = ones, P = e1: after the first step r(0) = 0,
    // and with A diagonal it stays 0, so P^H r = 0 and P^H G = 0 from the
    // second cycle on: a breakdown at a step, then again at the first step
    // of the restart. With P redrawn the solve converges.
    const std::size_t n = 50;
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (std::size_t i = 0; i < n; ++i) ins[i][i] << double(i + 1);
    }
    const itl::pc::identity<csr> M(A);
    const vec_t b(n, 1.0);
    std::vector<vec_t> P(1, vec_t(n, 0.0));
    P[0](0) = 1.0;
    vec_t x(n, 0.0);
    itl::basic_iteration<double> iter(b, 1000, 1e-10);
    CHECK(itl::idr_s(A, x, b, M, iter, itl::idrs_options{}, P) == 0);
    CHECK(true_relres(A, x, b) <= 1.5e-10);
}

TEST_CASE("IDR(s): breakdown codes (issue 874)", "[itl][krylov][idr_s][874]") {
    SECTION("A = 0: P^H G = 0 at the first step, code 2") {
        const csr A(3, 3);
        const itl::pc::identity<csr> M(A);
        const vec_t b(3, 1.0);
        vec_t x(3, 0.0);
        itl::basic_iteration<double> iter(b, 50, 1e-10);
        CHECK(itl::idr_s(A, x, b, M, iter, 1) == 2);
    }
    SECTION("a skew A: t is orthogonal to r, omega = 0, code 3") {
        // A = [[0, 1], [-1, 0]], b = e1, s = 1: after the first step r is
        // nonzero, and t = A r is orthogonal to it.
        csr A(2, 2);
        {
            mat::inserter<csr> ins(A);
            ins[0][1] << 1.0;
            ins[1][0] << -1.0;
        }
        const itl::pc::identity<csr> M(A);
        vec_t b(2, 0.0);
        b(0) = 1.0;
        vec_t x(2, 0.0);
        itl::basic_iteration<double> iter(b, 50, 1e-10);
        CHECK(itl::idr_s(A, x, b, M, iter, 1) == 3);
    }
}

TEST_CASE("IDR(s): options, the shadow space and reproducibility (issue 874)", "[itl][krylov][idr_s][874]") {
    const csr A = stencil(16, -1.4, -0.6);
    const std::size_t n = A.num_rows();
    const itl::pc::identity<csr> M(A);
    const vec_t b(n, 1.0);

    SECTION("a seeded run is bit-reproducible; another seed takes another path") {
        itl::idrs_options opt;
        opt.s = 4;
        opt.seed = 7;
        vec_t x1(n, 0.0), x2(n, 0.0), x3(n, 0.0);
        itl::basic_iteration<double> i1(b, 500, 1e-10), i2(b, 500, 1e-10), i3(b, 500, 1e-10);
        REQUIRE(itl::idr_s(A, x1, b, M, i1, opt) == 0);
        REQUIRE(itl::idr_s(A, x2, b, M, i2, opt) == 0);
        opt.seed = 8;
        REQUIRE(itl::idr_s(A, x3, b, M, i3, opt) == 0);
        bool same = true, differs = false;
        for (std::size_t i = 0; i < n; ++i) {
            same = same && x1(i) == x2(i);
            differs = differs || x1(i) != x3(i);
        }
        CHECK(same);
        CHECK(i1.iterations() == i2.iterations());
        CHECK(differs);
    }
    SECTION("a user-supplied shadow space") {
        // Any full-rank P: here unit vectors and a ramp, orthonormalized inside.
        std::vector<vec_t> P(3, vec_t(n, 0.0));
        P[0](0) = 1.0;
        P[1](n / 2) = 1.0;
        for (std::size_t i = 0; i < n; ++i) P[2](i) = 1.0 + double(i % 5);
        vec_t x(n, 0.0);
        itl::basic_iteration<double> iter(b, 1000, 1e-10);
        CHECK(itl::idr_s(A, x, b, M, iter, itl::idrs_options{}, P) == 0);
        CHECK(true_relres(A, x, b) <= 1.5e-10);
    }
    SECTION("a rank-deficient or wrongly sized shadow space throws") {
        vec_t x(n, 0.0);
        itl::basic_iteration<double> iter(b, 100, 1e-10);
        std::vector<vec_t> P(2, vec_t(n, 1.0));   // two equal vectors
        CHECK_THROWS_AS(itl::idr_s(A, x, b, M, iter, itl::idrs_options{}, P), std::invalid_argument);
        std::vector<vec_t> Q(1, vec_t(n - 1, 1.0));
        CHECK_THROWS_AS(itl::idr_s(A, x, b, M, iter, itl::idrs_options{}, Q), std::invalid_argument);
    }
    SECTION("smoothing gives a monotone residual and the same accuracy") {
        itl::idrs_options opt;
        opt.smoothing = true;
        vec_t x(n, 0.0);
        recording iter(b, 1000, 1e-10);
        REQUIRE(itl::idr_s(A, x, b, M, iter, opt) == 0);
        CHECK(true_relres(A, x, b) <= 1.5e-10);
        // Up to the final true-residual test, each smoothed residual is at
        // most the one before it.
        REQUIRE(iter.seen.size() >= 3);
        for (std::size_t k = 1; k + 1 < iter.seen.size(); ++k) CHECK(iter.seen[k] <= iter.seen[k - 1] * (1 + 1e-12));
    }
    SECTION("kappa = 0.7, and replacement off, still converge") {
        itl::idrs_options opt;
        opt.kappa = 0.7;
        opt.residual_replacement = false;
        vec_t x(n, 0.0);
        itl::basic_iteration<double> iter(b, 1000, 1e-10);
        CHECK(itl::idr_s(A, x, b, M, iter, opt) == 0);
        CHECK(true_relres(A, x, b) <= 1.5e-10);
    }
    SECTION("s is clamped to n; an empty system has converged") {
        const csr B = stencil(2, -1.4, -0.6);   // n = 4
        const vec_t c(4, 1.0);
        vec_t y(4, 0.0);
        itl::basic_iteration<double> iter(c, 100, 1e-12);
        itl::idrs_options opt;
        opt.s = 16;
        CHECK(itl::idr_s(B, y, c, itl::pc::identity<csr>(B), iter, opt) == 0);
        const csr E(0, 0);
        const vec_t e(0);
        vec_t z(0);
        itl::basic_iteration<double> it0(1.0, 10, 1e-10);
        CHECK(itl::idr_s(E, z, e, itl::pc::identity<csr>(E), it0, 4) == 0);
    }
}

TEST_CASE("IDR(s): a complex nonsymmetric system, with a complex shadow space (issue 874)", "[itl][krylov][idr_s][874]") {
    using C = std::complex<double>;
    using ccsr = mat::compressed2D<C>;
    using cvec = vec::dense_vector<C>;
    const std::size_t g = 12, n = g * g;
    ccsr A(n, n);
    {
        // convection-diffusion plus an imaginary shift, 4 + 0.5i on the diagonal
        mat::inserter<ccsr> ins(A);
        for (std::size_t r = 0; r < g; ++r)
            for (std::size_t c = 0; c < g; ++c) {
                const std::size_t i = r * g + c;
                ins[i][i] << C(4.0, 0.5);
                if (c > 0)     ins[i][i - 1] << C(-1.3, 0.0);
                if (c + 1 < g) ins[i][i + 1] << C(-0.7, 0.0);
                if (r > 0)     ins[i][i - g] << C(-1.3, 0.0);
                if (r + 1 < g) ins[i][i + g] << C(-0.7, 0.0);
            }
    }
    cvec b(n), x(n, C(0.0, 0.0));
    for (std::size_t i = 0; i < n; ++i) b(i) = C(1.0 + double(i % 3), double(i % 2));
    const itl::pc::identity<ccsr> M(A);
    for (const std::size_t s : {1u, 4u}) {
        for (std::size_t i = 0; i < n; ++i) x(i) = C(0.0, 0.0);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        INFO("s = " << s);
        REQUIRE(itl::idr_s(A, x, b, M, iter, s) == 0);
        cvec Ax = A * x;
        double r = 0, nb = 0;
        for (std::size_t i = 0; i < n; ++i) {
            r  += std::norm(b(i) - Ax(i));
            nb += std::norm(b(i));
        }
        CHECK(std::sqrt(r / nb) <= 1.5e-10);
    }
}
