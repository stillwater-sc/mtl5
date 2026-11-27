#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sstream>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/iteration/noisy_iteration.hpp>
#include <mtl/itl/krylov/cg.hpp>

using namespace mtl;

TEST_CASE("CG converges on 3x3 SPD system with identity PC", "[itl][cg]") {
    // A = {{4,1,0},{1,3,1},{0,1,2}} -- symmetric positive definite
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2;

    // b = {1, 2, 3}
    vec::dense_vector<double> b = {1.0, 2.0, 3.0};

    // x = 0 (initial guess)
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);

    REQUIRE(err == 0);
    REQUIRE(iter.iterations() <= 3); // CG converges in at most n steps for nxn

    // Verify A*x ~= b
    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("CG converges with diagonal (Jacobi) PC on tridiagonal", "[itl][cg]") {
    // 1D Laplacian: tridiagonal with 2 on diagonal, -1 on off-diagonals
    const std::size_t n = 10;
    mat::dense2D<double> A(n, n);
    // Zero out the matrix first
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 2.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -1.0;
    }

    // b = 1
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);

    REQUIRE(err == 0);

    // Verify A*x ~= b
    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("basic_iteration reports max_iter exceeded", "[itl][iteration]") {
    vec::dense_vector<double> r0 = {1.0, 1.0};
    itl::basic_iteration<double> iter(r0, 2, 1e-15); // max 2 iterations, very tight tol

    // Drive with a non-converging residual
    int count = 0;
    double fake_resid = 1.0;
    while (!iter.finished(fake_resid)) {
        ++iter;
        fake_resid *= 0.5; // halve each time, but won't reach 1e-15 in 2 steps
        ++count;
    }

    REQUIRE(iter.error_code() == 1); // max_iter exceeded
    REQUIRE(count == 2);
}

TEST_CASE("noisy_iteration prints to ostream", "[itl][iteration]") {
    std::ostringstream oss;
    vec::dense_vector<double> r0 = {3.0, 4.0}; // norm = 5
    itl::noisy_iteration<double> iter(r0, 5, 1e-15, 0.0, oss);

    // Run a few iterations with fake residuals
    iter.finished(5.0); // initial check
    ++iter;
    iter.finished(2.5);
    ++iter;
    iter.finished(1.0);

    std::string output = oss.str();
    REQUIRE(!output.empty());
    // Should contain "iteration" and "resid"
    REQUIRE(output.find("iteration") != std::string::npos);
    REQUIRE(output.find("resid") != std::string::npos);
}

// Issue 949: a negative rho = r^T M^-1 r (an M that is not positive definite)
// stops CG with code 7, where it ran on before.
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
namespace {
struct given_diagonal_949 {
    vec::dense_vector<double> d;
    template <typename VecX, typename VecB>
    void solve(VecX& z, const VecB& v) const {
        for (std::size_t i = 0; i < d.size(); ++i) z(i) = v(i) / d(i);
    }
};
mat::compressed2D<double> laplacian_949(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0)     ins[i][i - 1] << -1.0;
            if (i + 1 < n) ins[i][i + 1] << -1.0;
        }
    }
    return A;
}
} // namespace

TEST_CASE("cg: a preconditioner that is not positive definite returns 7 (issue 949)", "[itl][cg][949]") {
    const std::size_t n = 50;
    const auto A = laplacian_949(n);

    SECTION("M = diag(4, -3, 4, -3, ...) with b = ones: rho < 0 at once") {
        vec::dense_vector<double> b(n, 1.0), x(n, 0.0), d(n);
        for (std::size_t i = 0; i < n; ++i) d(i) = (i % 2 == 0 ? 4.0 : -3.0);
        given_diagonal_949 M{d};
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        REQUIRE(itl::cg(A, x, b, M, iter) == 7);
        CHECK(iter.iterations() == 1);   // counted, then refused before the step
        for (std::size_t i = 0; i < n; ++i) CHECK(x(i) == 0.0);
    }

    SECTION("found at a later step, x as the last valid step left it") {
        // b = e0: r reaches index k at step k; M is a tiny negative at index 4.
        vec::dense_vector<double> b(n, 0.0), d(n, 4.0), x(n, 0.0), y(n, 0.0);
        b(0) = 1.0;
        d(4) = -1e-6;
        given_diagonal_949 M{d};
        itl::basic_iteration<double> iter(b, 500, 1e-12);
        REQUIRE(itl::cg(A, x, b, M, iter) == 7);
        const int k = iter.iterations();
        REQUIRE(k >= 2);
        itl::basic_iteration<double> limit(b, k - 1, 1e-12);
        REQUIRE(itl::cg(A, y, b, M, limit) == 1);
        for (std::size_t i = 0; i < n; ++i) CHECK(x(i) == y(i));
    }

    SECTION("an SPD M, also at 2^+-300, is unaffected") {
        // At 2^+-600 CG fails on main as well (2, or 1 at the limit): its dot
        // products are not range-scaled as minres's are (#671). Not this check.
        vec::dense_vector<double> b(n, 1.0);
        for (const double s : {1.0, 0x1p300, 0x1p-300}) {
            vec::dense_vector<double> x(n, 0.0), d(n, 4.0 * s);
            given_diagonal_949 M{d};
            itl::basic_iteration<double> iter(b, 500, 1e-10);
            REQUIRE(itl::cg(A, x, b, M, iter) == 0);
        }
    }
}

// Review of #952: the tolerance n u ||r|| ||z|| must neither overflow while
// rho is finite, nor use a finer rounding unit than the accumulator's.
#include <cmath>
#include <utility>
#include <mtl/itl/krylov/spd_check.hpp>
namespace {
// M = [[0, 1], [1, 1/2]], indefinite: M^-1 = [[-1/2, 1], [1, 0]].
struct indefinite_2x2_949 {
    template <typename VecX, typename VecB>
    void solve(VecX& z, const VecB& v) const {
        const double v0 = v(0), v1 = v(1);
        z(0) = -0.5 * v0 + v1;
        z(1) = v0;
    }
};
} // namespace

TEST_CASE("cg: code 7 when ||r|| ||z|| overflows but rho does not (issue 949)", "[itl][cg][949]") {
    // A = I, b = 0, x0 = (-2^512, 0): r = (2^512, 0), z = (-2^511, 2^512),
    // rho = -2^1023 is finite; ||r|| ||z|| is not.
    mat::dense2D<double> A = {{1.0, 0.0}, {0.0, 1.0}};
    vec::dense_vector<double> b(2, 0.0), x(2, 0.0);
    x(0) = -0x1p512;
    const indefinite_2x2_949 M;
    itl::basic_iteration<double> iter(0x1p512, 50, 1e-10);
    REQUIRE(itl::cg(A, x, b, M, iter) == 7);
    CHECK(x(0) == -0x1p512);
    CHECK(x(1) == 0.0);
}

TEST_CASE("not_positive_definite: the tolerance's rounding unit and range (issue 949)", "[itl][949]") {
    namespace d = itl::detail;
    // A dot product of 5 terms of order 1 that rounds to -2^-23: within a
    // float accumulator's rounding, far outside double's.
    const auto norms = [] { return std::pair{std::sqrt(5.0), std::sqrt(3.0)}; };
    CHECK(d::not_positive_definite(-0x1p-23, 5, norms));
    CHECK_FALSE(d::not_positive_definite<float>(-0x1p-23, 5, norms));
    CHECK(d::not_positive_definite<float>(-1.0, 5, norms));     // still found
    CHECK(d::dot_rounding_unit<float, double>() == double(std::numeric_limits<float>::epsilon()));
    CHECK(d::dot_rounding_unit<long double, double>() == std::numeric_limits<double>::epsilon());
    // Norms whose product overflows, around a finite negative d.
    CHECK(d::not_positive_definite(-0x1p1023, 2, [] { return std::pair{0x1p512, 0x1.2p512}; }));
    // Rounding-sized negatives, and positives and NaN, are not a failure.
    CHECK_FALSE(d::not_positive_definite(-1e-17, 5, norms));
    CHECK_FALSE(d::not_positive_definite(1.0, 5, norms));
    CHECK_FALSE(d::not_positive_definite(std::nan(""), 5, norms));
}
