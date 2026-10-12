#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/spectral_properties.hpp>
#include <mtl/operation/householder.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randsvd.hpp>
#include <mtl/generators/hilbert.hpp>
#include <mtl/generators/ones.hpp>
#include <mtl/generators/moler.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace mtl;

TEST_CASE("SVD: U*S*V^T reproduces A", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Reconstruct: A_approx = U * S * V^T
    auto SV = S * trans(V);
    auto A_approx = U * SV;

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(A_approx(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-8));
}

TEST_CASE("SVD: U and V are orthogonal", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 2; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 3;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // U^T * U = I
    auto UtU = trans(U) * U;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(UtU(i, j), Catch::Matchers::WithinAbs(expected, 1e-8));
        }

    // V^T * V = I
    auto VtV = trans(V) * V;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(VtV(i, j), Catch::Matchers::WithinAbs(expected, 1e-8));
        }
}

TEST_CASE("SVD: singular values of diagonal matrix", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 5; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 3; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 1;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Singular values should be 5, 3, 1 (in S diagonal)
    std::vector<double> sv = {S(0,0), S(1,1), S(2,2)};
    std::sort(sv.begin(), sv.end());

    REQUIRE_THAT(sv[0], Catch::Matchers::WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(sv[1], Catch::Matchers::WithinAbs(3.0, 1e-8));
    REQUIRE_THAT(sv[2], Catch::Matchers::WithinAbs(5.0, 1e-8));
}

TEST_CASE("SVD: tuple return form", "[operation][svd]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 3; A(0,1) = 0;
    A(1,0) = 0; A(1,1) = 4;

    auto [U, S, V] = svd(A, 1e-12);

    // Reconstruct
    auto A_approx = U * S * trans(V);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            REQUIRE_THAT(A_approx(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-8));
}

// -- Generator-based SVD tests ----------------------------------------

TEST_CASE("SVD recovers prescribed singular values", "[operation][svd][generator]") {
    // Ground truth test: prescribed singular values must be recovered
    std::vector<double> sigma_prescribed = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    auto A = generators::randsvd<double>(6, 6, sigma_prescribed);

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Extract computed singular values and sort
    std::size_t n = 6;
    std::vector<double> sv_computed(n);
    for (std::size_t i = 0; i < n; ++i)
        sv_computed[i] = S(i, i);
    std::sort(sv_computed.begin(), sv_computed.end());

    std::vector<double> sv_expected = sigma_prescribed;
    std::sort(sv_expected.begin(), sv_expected.end());

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(sv_computed[i], Catch::Matchers::WithinAbs(sv_expected[i], 1e-8));
}

TEST_CASE("SVD condition number from randsvd", "[operation][svd][generator]") {
    // Verify sigma_max / sigma_min ~= kappa
    constexpr std::size_t n = 5;
    constexpr double kappa = 50.0;
    auto A = generators::randsvd<double>(n, kappa, 3);

    auto [U, S, V] = svd(A, 1e-12);

    std::vector<double> sv(n);
    for (std::size_t i = 0; i < n; ++i)
        sv[i] = S(i, i);
    std::sort(sv.begin(), sv.end());

    double computed_kappa = sv[n - 1] / sv[0];
    REQUIRE_THAT(computed_kappa, Catch::Matchers::WithinRel(kappa, 1e-6));
}

TEST_CASE("SVD of Hilbert matrix: reconstruction", "[operation][svd][generator]") {
    constexpr std::size_t n = 5;
    generators::hilbert<double> H_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = H_gen(i, j);

    auto [U, S, V] = svd(A, 1e-14);

    // Reconstruct: U*S*V^T should equal A
    auto A_approx = U * S * trans(V);
    double rel_error = frobenius_norm(A_approx - A) / frobenius_norm(A);
    REQUIRE(rel_error < 1e-8);
}

TEST_CASE("SVD of rank-1 ones matrix", "[operation][svd][generator]") {
    constexpr std::size_t n = 4;
    generators::ones<double> J_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = J_gen(i, j);

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Rank-1: only one nonzero singular value = n
    std::vector<double> sv(n);
    for (std::size_t i = 0; i < n; ++i)
        sv[i] = std::abs(S(i, i));
    std::sort(sv.rbegin(), sv.rend());

    REQUIRE_THAT(sv[0], Catch::Matchers::WithinAbs(static_cast<double>(n), 1e-8));
    for (std::size_t i = 1; i < n; ++i)
        REQUIRE_THAT(sv[i], Catch::Matchers::WithinAbs(0.0, 1e-8));
}

TEST_CASE("SVD orthogonality on Moler matrix", "[operation][svd][generator]") {
    // Moler is SPD with eigenvalues clustered near 0 -- stresses SVD
    constexpr std::size_t n = 3;
    auto A = generators::moler<double>(n);

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-14);

    // Verify singular values are non-negative
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(S(i, i) >= -1e-10);

    // Sum of singular values squared = ||A||_F^2 for SPD matrices
    // (singular values = eigenvalues for SPD)
    double sv_sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        sv_sum_sq += S(i, i) * S(i, i);
    double fnorm = frobenius_norm(A);
    REQUIRE_THAT(sv_sum_sq, Catch::Matchers::WithinAbs(fnorm * fnorm, 0.5));
}

// ---------------------------------------------------------------------------
// Regression: #337 -- NaN or badly wrong singular values on ordinary input.
//
// The alternating-QR iteration converged to ~1e-10 and then corrupted its own
// answer, while its convergence test (off-diagonal mass / diagonal mass) kept
// shrinking and so read as converged. Degenerate Householder reflectors turned
// ~30% of symmetric inputs into all-NaN. Replaced by one-sided Jacobi.
//
// The existing cases above use small or specially structured matrices and all
// passed throughout, so the sweeps below deliberately use ordinary random
// input at the sizes that failed.
// ---------------------------------------------------------------------------

TEST_CASE("SVD: no NaN and sigma_max == spectral radius on symmetric input (#337)",
          "[operation][svd][regression]") {
    // For a symmetric matrix sigma_max is the spectral radius by definition,
    // which checks the values without needing a reference implementation.
    std::uint64_t seed = 20260802u;
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((seed >> 11) % 2000001) / 1000000.0 - 1.0;
    };

    for (std::size_t n = 3; n <= 20; ++n) {
        mtl::mat::dense2D<double> A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i; j < n; ++j) { double v = next(); A(i,j) = v; A(j,i) = v; }

        const auto sv = mtl::detail::singular_values(A);
        REQUIRE(sv.size() == n);

        double smax = 0.0;
        for (double s : sv) {
            INFO("n = " << n);
            REQUIRE(std::isfinite(s));          // the all-NaN failure mode
            REQUIRE(s >= -1e-12);
            if (s > smax) smax = s;
        }

        const double sr = mtl::spectral_radius(A);
        INFO("n = " << n << ", sigma_max = " << smax << ", spectral radius = " << sr);
        REQUIRE(std::abs(smax - sr) <= 1e-9 * (sr > 0.0 ? sr : 1.0));
    }
}

TEST_CASE("SVD: reconstruction and orthogonality on random shapes (#337)",
          "[operation][svd][regression]") {
    std::uint64_t seed = 987654321u;
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((seed >> 11) % 2000001) / 1000000.0 - 1.0;
    };

    const std::size_t shapes[][2] = {{4,4}, {8,8}, {12,12}, {7,3}, {3,7}, {10,6}, {6,10}};

    for (const auto& sh : shapes) {
        const std::size_t m = sh[0], n = sh[1];
        mtl::mat::dense2D<double> A(m, n);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) A(i,j) = next();

        mtl::mat::dense2D<double> U, S, V;
        mtl::svd(A, U, S, V);

        INFO("shape " << m << "x" << n);

        // A == U * S * V^T
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < m; ++k)
                    for (std::size_t l = 0; l < n; ++l) acc += U(i,k) * S(k,l) * V(j,l);
                REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(A(i,j), 1e-10));
            }

        // U and V orthogonal, including the completion columns when m > rank
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < m; ++k) acc += U(k,i) * U(k,j);
                REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(i == j ? 1.0 : 0.0, 1e-10));
            }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < n; ++k) acc += V(k,i) * V(k,j);
                REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(i == j ? 1.0 : 0.0, 1e-10));
            }

        // Singular values non-negative and descending
        const std::size_t mn = std::min(m, n);
        for (std::size_t i = 0; i + 1 < mn; ++i) REQUIRE(S(i,i) >= S(i+1,i+1));
        for (std::size_t i = 0; i < mn; ++i)     REQUIRE(S(i,i) >= 0.0);
    }
}

TEST_CASE("SVD: rank-deficient input keeps U orthogonal (#337)",
          "[operation][svd][regression]") {
    // Every column a multiple of the first: rank 1, so U's remaining columns
    // come from the orthonormal completion rather than from the data.
    const std::size_t n = 6;
    mtl::mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i,j) = static_cast<double>((i + 1) * (j + 1));

    mtl::mat::dense2D<double> U, S, V;
    mtl::svd(A, U, S, V);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < n; ++k) acc += U(k,i) * U(k,j);
            REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(i == j ? 1.0 : 0.0, 1e-10));
        }
    REQUIRE(S(0,0) > 1.0);
    for (std::size_t i = 1; i < n; ++i)
        REQUIRE_THAT(S(i,i), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

// ---------------------------------------------------------------------------
// Regression: review findings on the #337 fix itself.
//
// Both were real defects in the one-sided Jacobi implementation, and both were
// missed by the sweeps above -- which is the point of pinning them here.
// ---------------------------------------------------------------------------

TEST_CASE("SVD: extreme scaling does not overflow or underflow the rotations",
          "[operation][svd][regression]") {
    // The Jacobi inner products alpha/beta/gamma are formed from squares. Taken
    // raw they overflow to inf near 1e200 and underflow to zero near 1e-200 --
    // and a zero product silently skipped every rotation, so the columns were
    // never orthogonalized at all.
    const std::size_t n = 4;
    const double base[4][4] = {{4,1,0,0},{1,3,1,0},{0,1,2,1},{0,0,1,1}};

    for (double s : {1e-200, 1e-160, 1.0, 1e160, 1e200}) {
        mtl::mat::dense2D<double> A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) A(i,j) = base[i][j] * s;

        mtl::mat::dense2D<double> U, S, V;
        mtl::svd(A, U, S, V);
        INFO("scale = " << s);

        // Singular values scale linearly and stay finite.
        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(std::isfinite(S(i,i)));
            REQUIRE(S(i,i) >= 0.0);
        }
        REQUIRE(S(0,0) > 0.0);

        // U stays orthogonal -- the symptom when the rotations were skipped.
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < n; ++k) acc += U(k,i) * U(k,j);
                REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(i == j ? 1.0 : 0.0, 1e-10));
            }

        // Reconstruction, measured relatively so it is scale independent.
        double worst = 0.0, anorm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < n; ++k)
                    for (std::size_t l = 0; l < n; ++l) acc += U(i,k) * S(k,l) * V(j,l);
                worst = std::max(worst, std::abs(acc - A(i,j)));
                anorm = std::max(anorm, std::abs(A(i,j)));
            }
        REQUIRE(worst <= 1e-12 * (anorm > 0.0 ? anorm : 1.0));
    }
}

TEST_CASE("SVD: raw and scaled Jacobi inner products agree across the range",
          "[operation][svd][regression]") {
    // The Jacobi core sums each pair's inner products raw and keeps them only
    // when ||b_p||^2 and ||b_q||^2 both land in [sqrt(min)/eps, sqrt(max)/2]:
    // about 2^-459 to 2^511 in double. Otherwise it re-forms them scaled by
    // the pair's largest entry. Scaling A by 2^k is exact, so the singular
    // values must scale by exactly 2^k up to rounding, and V must not move,
    // on both sides of each bound. Too wide a window fails here: without the
    // lower bound the squares at 2^-530 are subnormal, without the upper one
    // ||b_p||^2 * ||b_q||^2 overflows at 2^400 and every rotation is skipped,
    // and with no fallback the squares at 2^520 overflow. The V-only svd is
    // used because it always runs the generic core, LAPACK build or not.
    // Inputs come from mt19937_64 mapped by hand to [-1, 1).
    std::mt19937_64 gen(5701);
    auto u = [&gen] { return double(gen() >> 11) * 0x1.0p-53 * 2.0 - 1.0; };
    const double eps = std::numeric_limits<double>::epsilon();

    // A uniform 12 x 7, and one whose columns span 2^-560 .. 2^-80, so that most
    // pairs mix a column inside the window with one outside it.
    const std::size_t m = 12, n = 7;
    mtl::mat::dense2D<double> A0(m, n), G0(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            A0(i, j) = u();
            G0(i, j) = std::ldexp(u(), -560 + 80 * static_cast<int>(j));
        }

    for (const auto* base : {&A0, &G0}) {
        mtl::vec::dense_vector<double> s0;
        mtl::mat::dense2D<double> V0;
        mtl::svd(*base, s0, V0);
        for (int k : {-560, -530, -500, -300, -235, -225, 0, 250, 260, 400, 520}) {
            if (base == &G0 && k < -460) continue;   // 2^(k-560) would be subnormal: not exact
            mtl::mat::dense2D<double> A(m, n);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < n; ++j) A(i, j) = std::ldexp((*base)(i, j), k);
            mtl::vec::dense_vector<double> s;
            mtl::mat::dense2D<double> V;
            mtl::svd(A, s, V);
            INFO((base == &A0 ? "uniform" : "graded") << ", k = " << k);
            for (std::size_t j = 0; j < n; ++j) {
                const double sj = std::ldexp(s(j), -k);
                REQUIRE(std::abs(sj - s0(j)) <= 100.0 * eps * s0(j));
                double d = 0.0;
                for (std::size_t i = 0; i < n; ++i) d += V(i, j) * V0(i, j);
                REQUIRE(1.0 - std::abs(d) <= 100.0 * eps);
            }
        }
    }
}

TEST_CASE("SVD: basis completion when no canonical vector is dominant",
          "[operation][svd][regression]") {
    // A = I - v*v^T with v = ones/sqrt(m) has rank m-1, and the single missing
    // direction is spread evenly over the coordinates: every canonical vector
    // has residual exactly 1/sqrt(m) against the computed columns. Any fixed
    // acceptance bound above that rejects all of them and leaves a stale
    // identity column in U. 1/sqrt(m) is 0.5 at m=4 and 0.25 at m=16.
    for (std::size_t m : {4u, 9u, 16u}) {
        mtl::mat::dense2D<double> A(m, m);
        const double c = 1.0 / static_cast<double>(m);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j)
                A(i,j) = (i == j ? 1.0 : 0.0) - c;

        mtl::mat::dense2D<double> U, S, V;
        mtl::svd(A, U, S, V);
        INFO("m = " << m << ", residual per candidate = " << 1.0 / std::sqrt((double)m));

        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < m; ++k) acc += U(k,i) * U(k,j);
                REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(i == j ? 1.0 : 0.0, 1e-10));
            }

        // Rank m-1: exactly one zero singular value.
        REQUIRE_THAT(S(m-1, m-1), Catch::Matchers::WithinAbs(0.0, 1e-12));
        REQUIRE(S(m-2, m-2) > 0.5);
    }
}

TEST_CASE("Householder: empty input is well defined", "[operation][householder][regression]") {
    mtl::vec::dense_vector<double> x(0);
    auto [v, beta] = mtl::householder(x);
    REQUIRE(v.size() == 0);
    REQUIRE(beta == 0.0);
}
