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
#include <cstddef>
#include <vector>

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
    for (const std::size_t s : {1u, 4u}) {
        for (std::size_t i = 0; i < n; ++i) x(i) = 0.0f;
        itl::basic_iteration<float> iter(b, 500, 1e-5f);
        INFO("s = " << s);
        REQUIRE(itl::idr_s(A, x, b, M, iter, s) == 0);
        fvec Ax = A * x;
        float r = 0, nb = 0;
        for (std::size_t i = 0; i < n; ++i) {
            r  += (b(i) - Ax(i)) * (b(i) - Ax(i));
            nb += b(i) * b(i);
        }
        CHECK(std::sqrt(r / nb) <= 1e-3f);   // float: the updated residual drifts from b - A x
    }
}
