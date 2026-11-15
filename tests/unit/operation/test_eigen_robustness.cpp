// The dense eigensolvers and the SVD on hard input (#853): extreme scaling,
// non-convergence, NaN or infinite entries, non-square matrices, and the
// in-house eigen() on the Schur form.
//
// An explicit Accumulator (double) forces the in-house paths in a LAPACK
// build, so these test MTL5's own code on every configuration.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/eigenvalue_symmetric_jacobi.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;
using cplx = std::complex<double>;

namespace {

mat::dense2D<double> random_matrix(std::size_t n, unsigned seed, double scale = 1.0) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = u(g) * scale;
    return A;
}

mat::dense2D<double> random_symmetric(std::size_t n, unsigned seed) {
    auto A = random_matrix(n, seed);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) A(j, i) = A(i, j);
    return A;
}

std::vector<cplx> sorted(const vec::dense_vector<cplx>& e) {
    std::vector<cplx> v(e.begin(), e.end());
    std::sort(v.begin(), v.end(), [](cplx a, cplx b) {
        return a.real() != b.real() ? a.real() < b.real() : a.imag() < b.imag();
    });
    return v;
}

/// max_k ||A v_k - lambda_k v_k|| / ||A||_max, formed on A / scale so the
/// products stay finite for a matrix near overflow or underflow.
double relative_residual(const mat::dense2D<double>& A, const vec::dense_vector<cplx>& e,
                         const mat::dense2D<cplx>& V, double scale) {
    const std::size_t n = A.num_rows();
    double amax = 0, worst = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) amax = std::max(amax, std::abs(A(i, j) / scale));
    for (std::size_t k = 0; k < n; ++k)
        for (std::size_t i = 0; i < n; ++i) {
            cplx r = -(e(k) / scale) * V(i, k);
            for (std::size_t j = 0; j < n; ++j) r += (A(i, j) / scale) * V(j, k);
            // A NaN would lose std::max and leave worst unchanged.
            if (!std::isfinite(std::abs(r))) return std::numeric_limits<double>::infinity();
            worst = std::max(worst, std::abs(r));
        }
    return worst / amax;
}

} // namespace

TEST_CASE("eigenvalue and eigen: entries near 1e+-170..1e+-250", "[operation][eigen][853]") {
    // eigenvalue() did not scale, as schur() does: at 1e-170 and 1e-200 its
    // eigenvalues were wrong by about 0.7 (relative), at 1e+-250 it threw
    // "failed to converge". The LAPACK path was right, so the paths disagreed.
    const auto A = random_matrix(4, 7);
    const auto ref = sorted(eigenvalue<double>(A));
    for (double s : {1e-170, 1e-200, 1e-250, 1e170, 1e250}) {
        INFO("scale " << s);
        mat::dense2D<double> As(4, 4);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) As(i, j) = A(i, j) * s;
        const auto e = sorted(eigenvalue<double>(As));
        for (std::size_t k = 0; k < 4; ++k)
            CHECK(std::abs(e[k] / s - ref[k]) <= 1e-12 * std::abs(ref[k]) + 1e-13);
        const auto [ev, V] = eigen<double>(As);
        CHECK(relative_residual(As, ev, V, s) <= 1e-12);
        const auto d = sorted(eigenvalue(As));   // the default path, LAPACK where built
        for (std::size_t k = 0; k < 4; ++k)
            CHECK(std::abs(d[k] / s - ref[k]) <= 1e-12 * std::abs(ref[k]) + 1e-13);
    }
}

TEST_CASE("in-house eigen: residual, independence and pairing", "[operation][eigen][853]") {
    // eigen<double> is the in-house path: the Schur form and back-substitution.
    for (std::size_t n : {2u, 5u, 17u, 40u}) {
        INFO("n = " << n);
        const auto A = random_matrix(n, unsigned(n));
        // The default tol, 1e-10, is the deflation tolerance and bounds the
        // residual; a tight one gives a backward-stable decomposition.
        const auto [e, V] = eigen<double>(A, 1e-15);
        CHECK(relative_residual(A, e, V, 1.0) <= 1e-13 * double(n));
        const auto [ed, Vd] = eigen<double>(A);
        CHECK(relative_residual(A, ed, Vd, 1.0) <= 1e-9);
        for (std::size_t k = 0; k < n; ++k) {
            double s = 0;
            for (std::size_t i = 0; i < n; ++i) s += std::norm(V(i, k));
            CHECK(std::abs(std::sqrt(s) - 1.0) <= 1e-13);
        }
        // The same spectrum as eigenvalue() at the same tolerance.
        const auto a = sorted(e), b = sorted(eigenvalue<double>(A, 1e-15));
        for (std::size_t k = 0; k < n; ++k) CHECK(std::abs(a[k] - b[k]) <= 1e-12);
    }
    // A repeated eigenvalue of a diagonalizable matrix: independent vectors.
    mat::dense2D<double> D(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) D(i, j) = 0;
    D(0, 0) = 2; D(1, 1) = 2; D(2, 2) = 2; D(3, 3) = 5;
    const auto [e, V] = eigen<double>(D);
    CHECK(relative_residual(D, e, V, 1.0) <= 1e-14);
    // The three unit vectors for 2 are independent: their Gram matrix G is
    // nonsingular. Pairwise separation would not show it (e1, e2 and
    // (e1 + e2)/sqrt(2) are pairwise apart but dependent); det G does.
    std::vector<std::size_t> twos;
    for (std::size_t k = 0; k < 4; ++k) if (std::abs(e(k) - cplx(2)) < 1e-12) twos.push_back(k);
    REQUIRE(twos.size() == 3);
    cplx G[3][3];
    for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t b = 0; b < 3; ++b) {
            G[a][b] = cplx(0);
            for (std::size_t i = 0; i < 4; ++i) G[a][b] += std::conj(V(i, twos[a])) * V(i, twos[b]);
        }
    const cplx detG = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1])
                    - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
                    + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);
    CHECK(std::abs(detG) > 1e-3);
    // A complex pair: conjugate vectors for conjugate eigenvalues.
    mat::dense2D<double> R{{0, -2}, {2, 0}};
    const auto [er, Vr] = eigen<double>(R);
    CHECK(relative_residual(R, er, Vr, 1.0) <= 1e-14);
    CHECK(std::abs(er(0) - std::conj(er(1))) <= 1e-14);
}

TEST_CASE("non-convergence throws", "[operation][eigen][svd][853]") {
    // The symmetric, Jacobi and SVD iterations returned what they had when
    // they ran out of iterations; eigenvalue() and schur() threw.
    const auto S = random_symmetric(6, 3);
    CHECK_THROWS_AS(eigenvalue_symmetric<double>(S, 1e-15, 1), std::runtime_error);
    CHECK_THROWS_AS(eigen_symmetric<double>(S, 1e-15, 1), std::runtime_error);
    CHECK_THROWS_AS(eigen_symmetric_jacobi(S, std::numeric_limits<double>::epsilon(), 1),
                    std::runtime_error);
    CHECK_THROWS_AS(eigenvalue<double>(random_matrix(6, 4), 1e-15, 1), std::runtime_error);
    CHECK_THROWS_AS(eigen<double>(random_matrix(6, 4), 1e-15, 1), std::runtime_error);
    // The same calls converge with the default limits.
    CHECK(eigenvalue_symmetric<double>(S).size() == 6);
    CHECK(eigen_symmetric_jacobi(S).eigenvalues.size() == 6);
}

TEST_CASE("a NaN or infinite entry is an error", "[operation][eigen][svd][853]") {
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        INFO("bad = " << bad);
        auto A = random_symmetric(4, 5);
        A(1, 2) = A(2, 1) = bad;
        CHECK_THROWS_AS(eigenvalue(A), std::invalid_argument);
        CHECK_THROWS_AS(eigenvalue<double>(A), std::invalid_argument);
        CHECK_THROWS_AS(eigen(A), std::invalid_argument);
        CHECK_THROWS_AS(eigenvalue_symmetric(A), std::invalid_argument);
        CHECK_THROWS_AS(eigenvalue_symmetric<double>(A), std::invalid_argument);
        CHECK_THROWS_AS(eigen_symmetric(A), std::invalid_argument);
        CHECK_THROWS_AS(eigen_symmetric_jacobi(A), std::invalid_argument);
        CHECK_THROWS_AS(svd(A), std::invalid_argument);
        CHECK_THROWS_AS(svd<double>(A), std::invalid_argument);
    }
}

TEST_CASE("a non-square matrix is an error", "[operation][eigen][853]") {
    // These only asserted: under NDEBUG they read past the last column or used
    // the leading block.
    mat::dense2D<double> R(3, 4);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j) R(i, j) = double(i + j);
    CHECK_THROWS_AS(eigenvalue(R), std::invalid_argument);
    CHECK_THROWS_AS(eigenvalue<double>(R), std::invalid_argument);
    CHECK_THROWS_AS(eigen(R), std::invalid_argument);
    CHECK_THROWS_AS(eigenvalue_symmetric(R), std::invalid_argument);
    CHECK_THROWS_AS(eigen_symmetric(R), std::invalid_argument);
    CHECK_THROWS_AS(eigen_symmetric_jacobi(R), std::invalid_argument);
    CHECK_THROWS_AS(hessenberg(R), std::invalid_argument);
    vec::dense_vector<double> tau;
    CHECK_THROWS_AS(hessenberg_factor(R, tau), std::invalid_argument);
}

TEST_CASE("relative_residual sees a NaN", "[operation][eigen][853]") {
    mat::dense2D<double> A{{2, 0}, {0, 3}};
    vec::dense_vector<cplx> e(2);
    e(0) = cplx(2); e(1) = cplx(3);
    mat::dense2D<cplx> V(2, 2);
    V(0, 0) = cplx(1); V(1, 0) = cplx(0); V(0, 1) = cplx(0); V(1, 1) = cplx(std::nan(""));
    CHECK(relative_residual(A, e, V, 1.0) == std::numeric_limits<double>::infinity());
}

TEST_CASE("eigenvalue and eigen at the ends of the range", "[operation][eigen][853]") {
    // The scaling factor 2^1024 for [max()], and 2^1075 for [denorm_min()],
    // overflowed; the exponent does not.
    for (double x : {std::numeric_limits<double>::max(), std::numeric_limits<double>::denorm_min(),
                     -std::numeric_limits<double>::max()}) {
        INFO("x = " << x);
        mat::dense2D<double> A{{x}};
        CHECK(eigenvalue<double>(A)(0) == cplx(x));
        CHECK(eigenvalue(A)(0) == cplx(x));
        CHECK(eigen<double>(A).eigenvalues(0) == cplx(x));
    }
    // A 2x2 near overflow: both eigenvalues finite.
    const double big = std::numeric_limits<double>::max() / 4;
    mat::dense2D<double> B{{big, 0}, {0, -big}};
    const auto eb = sorted(eigenvalue<double>(B));
    CHECK((eb[0] == cplx(-big) && eb[1] == cplx(big)));
}

TEST_CASE("one last sweep that converges is not reported as non-convergence", "[operation][eigen][853]") {
    // One Jacobi rotation diagonalizes a 2x2; one QR sweep deflates [[0,1],[1,0]].
    mat::dense2D<double> J{{2, 1}, {1, 2}};
    const auto rj = eigen_symmetric_jacobi(J, std::numeric_limits<double>::epsilon(), 1);
    CHECK((std::abs(rj.eigenvalues(0) - 1.0) < 1e-15 && std::abs(rj.eigenvalues(1) - 3.0) < 1e-15));
    mat::dense2D<double> F{{0, 1}, {1, 0}};
    const auto ef = eigenvalue_symmetric<double>(F, 1e-15, 1);
    CHECK((std::abs(ef(0) + 1.0) < 1e-15 && std::abs(ef(1) - 1.0) < 1e-15));
    const auto rf = eigen_symmetric<double>(F, 1e-15, 1);
    CHECK((std::abs(rf.eigenvalues(0) + 1.0) < 1e-15 && std::abs(rf.eigenvalues(1) - 1.0) < 1e-15));
}

TEST_CASE("eigen: a small 2x2 block is not floored as if singular", "[operation][eigen][853]") {
    // A quasi-triangular A whose 2x2 block [0 d; -d 0] has entries of 1e-9
    // relative to its 1 elsewhere. Comparing the block's determinant (1e-18
    // scale) with eps * max|a| (linear) floored it, and the third eigenvector
    // came back with a residual near 1.6e-7.
    const double d = 1e-9;
    mat::dense2D<double> R{{0, d, 0}, {-d, 0, 1}, {0, 0, d}};
    const auto [e, V] = eigen<double>(R, 1e-15);
    CHECK(relative_residual(R, e, V, 1.0) <= 1e-14);
}
