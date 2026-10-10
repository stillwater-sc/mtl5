#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/itl/eigen/eigensolvers.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

using namespace mtl;
using cplx = std::complex<double>;

namespace {

// 1D Laplacian (SPD tridiagonal), eigenvalues 2-2cos(k*pi/(n+1)).
mat::dense2D<double> laplacian1d(std::size_t n) {
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 2.0;
        if (i > 0)     A(i, i - 1) = -1.0;
        if (i + 1 < n) A(i, i + 1) = -1.0;
    }
    return A;
}

std::vector<double> laplacian1d_eigs(std::size_t n) {
    std::vector<double> e(n);
    for (std::size_t k = 1; k <= n; ++k)
        e[k - 1] = 2.0 - 2.0 * std::cos(k * 3.14159265358979323846 / (n + 1));
    std::sort(e.begin(), e.end());
    return e;
}

// A matrix-free operator: y = A * x for a fixed dense A, to exercise the
// LinearOperator path with a user type rather than dense2D directly.
struct MatFreeOp {
    const mat::dense2D<double>& A;
    vec::dense_vector<double> operator*(const vec::dense_vector<double>& x) const {
        const std::size_t n = A.num_rows();
        vec::dense_vector<double> y(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) s += A(i, j) * x(j);
            y(i) = s;
        }
        return y;
    }
};

} // namespace

namespace {

// A Rayleigh quotient that fp32 accumulation gets entirely wrong and fp64 gets
// exactly right, so an accumulator test can assert which policy ran rather than
// only that both converged.
//
//   A = diag(2^20, 2^-5, ..., 2^-5, -2^20),  n = 1024,  v0 = ones
//
// v0 normalizes to q = 1/32 exactly, so every product q_i * (A q)_i = d_i / 1024
// is exact and the ONLY rounding is in the sum. In float the running sum sits at
// 2^10 after the first term; each 2^-15 term is below half an ulp of 2^10 (2^-14)
// and is rounded away, and the final -2^10 leaves exactly 0. In double all 1022
// small terms survive: 1022 * 2^-15, which is exact in float too.
constexpr std::size_t kCancelN = 1024;

mat::compressed2D<float> cancelling_diagonal() {
    mat::compressed2D<float> A(kCancelN, kCancelN);
    {
        mat::inserter<mat::compressed2D<float>> ins(A);
        ins[0][0] << std::ldexp(1.0f, 20);
        for (std::size_t i = 1; i + 1 < kCancelN; ++i) ins[i][i] << std::ldexp(1.0f, -5);
        ins[kCancelN - 1][kCancelN - 1] << -std::ldexp(1.0f, 20);
    }
    return A;
}

// q^T A q for the fixture above, computed exactly: (n - 2) * 2^-5 / n.
constexpr double kCancelRayleigh = (kCancelN - 2) * 0.03125 / kCancelN;

} // namespace

TEST_CASE("power_iteration: both accumulator policies converge on a float Laplacian",
          "[itl][eigen][power][accumulator]") {
    // float element type + large n makes the Rayleigh-quotient dot product and
    // Ritz-residual sum long enough for fp32 accumulation error to show up
    // against the known-exact eigenvalue (analytic formula, computed in double).
    const std::size_t n = 500;
    mat::dense2D<float> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 2.0f;
        if (i > 0)     A(i, i - 1) = -1.0f;
        if (i + 1 < n) A(i, i + 1) = -1.0f;
    }
    auto expected = laplacian1d_eigs(n);
    const double lambda_max = expected.back();

    vec::dense_vector<float> v0(n, 1.0f);
    v0(0) = 1.7f;

    // Configuration 1: default fp32 accumulation (Accumulator = float).
    auto naive = itl::power_iteration(A, v0, 20000, 1e-4f);
    // Wide accumulator: fp64 accumulation of the same fp32-element problem.
    // For this matrix, power_iteration's error is dominated by the spectral
    // gap rather than accumulation precision, so naive and wide converge to
    // near-identical error; the accuracy check below allows small slack
    // instead of a strict inequality.
    auto wide = itl::power_iteration<double>(A, v0, 20000, 1e-4);

    const double e_naive = std::abs(static_cast<double>(naive.value) - lambda_max);
    const double e_wide  = std::abs(static_cast<double>(wide.value)  - lambda_max);
    INFO("lambda_max=" << lambda_max
         << " naive.converged=" << naive.converged << " naive.iterations=" << naive.iterations << " e_naive=" << e_naive
         << " wide.converged="  << wide.converged  << " wide.iterations="  << wide.iterations  << " e_wide="  << e_wide);

    CHECK(naive.converged);
    CHECK(wide.converged);
    CHECK(e_naive < 1e-3);
    CHECK(e_wide  < 5e-4);
    CHECK(e_wide <= e_naive * 1.02);  // allow ~2% slack: both configs sit near the residual-tolerance noise floor
    // With 2% slack this cannot tell the two policies apart; the
    // cancelling-Rayleigh-quotient test below does.
}

TEST_CASE("power_iteration: the accumulator policy decides a cancelling Rayleigh quotient",
          "[itl][eigen][power][accumulator]") {
    // max_iter = 1 makes the returned value the first Rayleigh quotient
    // q^T (A q) with q = v0 / ||v0||: one call to dot<Acc, T>. The fixture makes
    // that reduction catastrophic in fp32 (see cancelling_diagonal). Its two
    // +-2^20 eigenvalues mean power iteration would not converge here, which is
    // why this asks for one iteration and does not check `converged`.
    const auto A = cancelling_diagonal();
    const vec::dense_vector<float> v0(kCancelN, 1.0f);

    auto naive = itl::power_iteration(A, v0, 1);
    auto wide  = itl::power_iteration<double>(A, v0, 1);

    const double r_naive = static_cast<double>(naive.value);
    const double r_wide  = static_cast<double>(wide.value);
    INFO("exact=" << kCancelRayleigh << " naive=" << r_naive << " wide=" << r_wide);
    CHECK(r_naive == 0.0);                 // every small term rounded away in fp32
    CHECK(r_wide == kCancelRayleigh);      // exact: the fp64 sum, then one rounding
}

TEST_CASE("power_iteration: dominant eigenpair of SPD Laplacian", "[itl][eigen][power]") {
    const std::size_t n = 20;
    auto A = laplacian1d(n);
    auto expected = laplacian1d_eigs(n);
    double lambda_max = expected.back();

    vec::dense_vector<double> v0(n, 1.0);
    v0(0) = 1.7;  // break symmetry with the null-ish modes
    auto pair = itl::power_iteration(A, v0, 5000, 1e-11);

    REQUIRE(pair.converged);
    REQUIRE_THAT(pair.value, Catch::Matchers::WithinAbs(lambda_max, 1e-6));

    // Residual A v - lambda v is small.
    auto Av = A * pair.vector;
    double res = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double r = Av(i) - pair.value * pair.vector(i);
        res += r * r;
    }
    REQUIRE(std::sqrt(res) < 1e-6);
}

TEST_CASE("lanczos: largest and smallest eigenvalues of SPD Laplacian", "[itl][eigen][lanczos]") {
    const std::size_t n = 30;
    auto A = laplacian1d(n);
    auto expected = laplacian1d_eigs(n);

    vec::dense_vector<double> v0(n);
    for (std::size_t i = 0; i < n; ++i) v0(i) = 1.0 + 0.1 * static_cast<double>(i);

    // Use a full subspace so the extremal Ritz pairs converge deterministically
    // (the default subspace still recovers the values, just not to 1e-8 residual).
    SECTION("largest algebraic") {
        auto r = itl::lanczos(A, v0, 3, itl::eigen_which::largest_algebraic, n);
        REQUIRE(r.values.size() == 3);
        REQUIRE(r.converged);
        // values are ordered largest-first
        REQUIRE_THAT(r.values(0), Catch::Matchers::WithinAbs(expected[n-1], 1e-7));
        REQUIRE_THAT(r.values(1), Catch::Matchers::WithinAbs(expected[n-2], 1e-7));
        REQUIRE_THAT(r.values(2), Catch::Matchers::WithinAbs(expected[n-3], 1e-7));
    }
    SECTION("smallest algebraic") {
        auto r = itl::lanczos(A, v0, 3, itl::eigen_which::smallest_algebraic, n);
        REQUIRE(r.converged);
        REQUIRE_THAT(r.values(0), Catch::Matchers::WithinAbs(expected[0], 1e-7));
        REQUIRE_THAT(r.values(1), Catch::Matchers::WithinAbs(expected[1], 1e-7));
        REQUIRE_THAT(r.values(2), Catch::Matchers::WithinAbs(expected[2], 1e-7));
    }
}

TEST_CASE("lanczos: Ritz vectors satisfy the eigen relation", "[itl][eigen][lanczos]") {
    const std::size_t n = 25;
    auto A = laplacian1d(n);
    vec::dense_vector<double> v0(n, 1.0);
    v0(1) = 2.0;

    auto r = itl::lanczos(A, v0, 4, itl::eigen_which::largest_algebraic);
    REQUIRE(r.values.size() == 4);
    for (std::size_t c = 0; c < 4; ++c) {
        // extract column c
        vec::dense_vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i) y(i) = r.vectors(i, c);
        auto Ay = A * y;
        double res = 0.0, nrm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double ri = Ay(i) - r.values(c) * y(i);
            res += ri * ri; nrm += y(i) * y(i);
        }
        REQUIRE_THAT(std::sqrt(nrm), Catch::Matchers::WithinAbs(1.0, 1e-8));
        REQUIRE(std::sqrt(res) < 1e-6);
    }
}

TEST_CASE("lanczos: works through a matrix-free operator", "[itl][eigen][lanczos]") {
    const std::size_t n = 20;
    auto A = laplacian1d(n);
    auto expected = laplacian1d_eigs(n);
    MatFreeOp op{A};

    vec::dense_vector<double> v0(n, 1.0);
    v0(0) = 1.3;
    auto r = itl::lanczos(op, v0, 2, itl::eigen_which::largest_algebraic);
    REQUIRE(r.converged);
    REQUIRE_THAT(r.values(0), Catch::Matchers::WithinAbs(expected[n-1], 1e-7));
    REQUIRE_THAT(r.values(1), Catch::Matchers::WithinAbs(expected[n-2], 1e-7));
}

TEST_CASE("arnoldi: eigenpairs of a nonsymmetric operator", "[itl][eigen][arnoldi]") {
    // Block-diagonal-ish nonsymmetric matrix with a known real dominant
    // eigenvalue 10 and a complex pair 2 +/- i.
    const std::size_t n = 6;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = 0.0;
    A(0,0) = 10.0;                       // dominant real
    A(1,1) = 2.0; A(1,2) = -1.0;         // 2 +/- i block
    A(2,1) = 1.0; A(2,2) = 2.0;
    A(3,3) = 4.0;
    A(4,4) = -3.0;
    A(5,5) = 1.0;

    vec::dense_vector<double> v0(n, 1.0);
    v0(2) = 0.5;
    auto r = itl::arnoldi(A, v0, 3, itl::eigen_which::largest_magnitude);
    REQUIRE(r.values.size() == 3);
    REQUIRE(r.converged);

    // Dominant Ritz value is 10.
    REQUIRE_THAT(r.values(0).real(), Catch::Matchers::WithinAbs(10.0, 1e-7));
    REQUIRE_THAT(r.values(0).imag(), Catch::Matchers::WithinAbs(0.0, 1e-8));

    // Each Ritz pair satisfies A y = mu y (complex).
    for (std::size_t c = 0; c < 3; ++c) {
        double res = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            cplx Ay(0.0, 0.0);
            for (std::size_t j = 0; j < n; ++j)
                Ay += cplx(A(i, j), 0.0) * r.vectors(j, c);
            Ay -= r.values(c) * r.vectors(i, c);
            res += std::norm(Ay);
        }
        REQUIRE(std::sqrt(res) < 1e-6);
    }
}

TEST_CASE("arnoldi: recovers a complex-conjugate pair", "[itl][eigen][arnoldi]") {
    const std::size_t n = 5;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = 0.0;
    // 3 +/- 2i block plus real eigenvalues 1, -1, 5.
    A(0,0) = 3.0; A(0,1) = -2.0;
    A(1,0) = 2.0; A(1,1) = 3.0;
    A(2,2) = 5.0;
    A(3,3) = 1.0;
    A(4,4) = -1.0;

    vec::dense_vector<double> v0(n, 1.0);
    v0(1) = 0.7;
    auto r = itl::arnoldi(A, v0, 5, itl::eigen_which::largest_magnitude, n);

    // The full spectrum (subspace = n) must contain 3 +/- 2i.
    std::vector<cplx> got(r.values.size());
    for (std::size_t i = 0; i < r.values.size(); ++i) got[i] = r.values(i);
    auto has = [&](cplx target) {
        return std::any_of(got.begin(), got.end(),
                           [&](cplx z){ return std::abs(z - target) < 1e-6; });
    };
    REQUIRE(has(cplx(3, 2)));
    REQUIRE(has(cplx(3, -2)));
    REQUIRE(has(cplx(5, 0)));
}

TEST_CASE("lanczos: both accumulator policies converge on a float Laplacian",
          "[itl][eigen][lanczos][accumulator]") {
    // float element type + moderate n makes the alpha dot product and beta
    // norm long enough for fp32 accumulation error to show up against the
    // known-exact eigenvalues (analytic formula, computed in double).
    const std::size_t n = 300;
    mat::dense2D<float> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 2.0f;
        if (i > 0)     A(i, i - 1) = -1.0f;
        if (i + 1 < n) A(i, i + 1) = -1.0f;
    }
    auto expected = laplacian1d_eigs(n);
    const double lambda_max = expected.back();
    vec::dense_vector<float> v0(n, 1.0f);
    v0(0) = 1.7f;

    // Configuration 1: default fp32 accumulation (Accumulator = float).
    auto naive = itl::lanczos(A, v0, 1, itl::eigen_which::largest_algebraic, n);
    // Wide accumulator: fp64 accumulation of the same fp32-element problem.
    auto wide = itl::lanczos<double>(A, v0, 1, itl::eigen_which::largest_algebraic, n);

    const double e_naive = std::abs(static_cast<double>(naive.values(0)) - lambda_max);
    const double e_wide  = std::abs(static_cast<double>(wide.values(0))  - lambda_max);
    INFO("lambda_max=" << lambda_max << " e_naive=" << e_naive << " e_wide=" << e_wide);
    // For this matrix, lanczos's error is dominated by the Krylov subspace
    // convergence rather than accumulation precision, so naive and wide land
    // near the same residual-tolerance noise floor -- same situation as
    // power_iteration's accuracy test. Check both sit near that floor rather
    // than asserting a strict ordering. This therefore cannot tell the two
    // policies apart; the cancelling-Rayleigh-quotient test below does.
    CHECK(e_naive < 1e-4);
    CHECK(e_wide  < 1e-4);
}

TEST_CASE("lanczos: the accumulator policy decides a cancelling Rayleigh quotient",
          "[itl][eigen][lanczos][accumulator]") {
    // subspace = 1 makes the single Ritz value exactly alpha_0 = q^T (A q): one
    // call to dot<Acc, T>, with nothing after it that could mask the policy. The
    // fixture makes that reduction catastrophic in fp32 (see cancelling_diagonal),
    // so a lanczos<double> that silently accumulated in float would return 0.
    const auto A = cancelling_diagonal();
    const vec::dense_vector<float> v0(kCancelN, 1.0f);

    auto naive = itl::lanczos(A, v0, 1, itl::eigen_which::largest_algebraic, 1);
    auto wide  = itl::lanczos<double>(A, v0, 1, itl::eigen_which::largest_algebraic, 1);

    const double r_naive = static_cast<double>(naive.values(0));
    const double r_wide  = static_cast<double>(wide.values(0));
    INFO("exact=" << kCancelRayleigh << " naive=" << r_naive << " wide=" << r_wide);
    CHECK(r_naive == 0.0);                 // every small term rounded away in fp32
    CHECK(r_wide == kCancelRayleigh);      // exact: the fp64 sum, then one rounding
}

TEST_CASE("arnoldi: both accumulator policies converge on a float nonsymmetric operator",
          "[itl][eigen][arnoldi][accumulator]") {
    // float element type + moderate n makes the Hessenberg dot products and
    // residual norm long enough for fp32 accumulation error to show up
    // against a known-exact dominant eigenvalue.
    const std::size_t n = 200;
    mat::dense2D<float> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = 0.0f;
    // Upper triangular, nonsymmetric, with well-separated diagonal entries
    // and weak coupling so the eigenvector basis stays well-conditioned
    // (a triangular matrix with close eigenvalues and strong coupling is
    // numerically defective-like and swamps any accumulator effect); the
    // superdiagonal still makes the operator genuinely non-normal so the
    // Arnoldi (not Lanczos) path is exercised.
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 2.0f - static_cast<float>(i) * 0.02f;
        if (i + 1 < n) A(i, i + 1) = 0.01f;
    }
    const double lambda_max = 2.0;  // exact: triangular, eigenvalues = diagonal, max at i=0
    vec::dense_vector<float> v0(n, 1.0f);
    v0(0) = 1.7f;

    // Configuration 1: default fp32 accumulation (Accumulator = float).
    auto naive = itl::arnoldi(A, v0, 1, itl::eigen_which::largest_algebraic, n);
    // Wide accumulator: fp64 accumulation of the same fp32-element problem.
    auto wide = itl::arnoldi<double>(A, v0, 1, itl::eigen_which::largest_algebraic, n);

    const double e_naive = std::abs(naive.values(0).real() - lambda_max);
    const double e_wide  = std::abs(wide.values(0).real()  - lambda_max);
    INFO("lambda_max=" << lambda_max << " e_naive=" << e_naive << " e_wide=" << e_wide);
    // As with lanczos, Krylov subspace convergence dominates accumulation
    // error here, so naive and wide should both land near the same
    // residual-tolerance noise floor rather than showing a strict ordering.
    CHECK(e_naive < 1e-4);
    CHECK(e_wide  < 1e-4);
}

TEST_CASE("arnoldi: the accumulator policy decides a cancelling Rayleigh quotient",
          "[itl][eigen][arnoldi][accumulator]") {
    // subspace = 1 makes the single Ritz value H(0,0), the sum of the two
    // Gram-Schmidt passes' q^T w, both accumulated in Acc. Same fixture as the
    // lanczos case: in fp32 both passes round every small term away.
    //
    // The fp64 result is not exact, and not because of the accumulator: after
    // pass 0, w -= s0 * q is applied in float, and the two +-2^20/32 entries of w
    // cannot absorb s0/32, so pass 1 measures a residual of 2*s0/n instead of 0.
    // H(0,0) = s0 * (1 + 2/n), i.e. 0.2% high at n = 1024. The bound is 1%.
    const auto A = cancelling_diagonal();
    const vec::dense_vector<float> v0(kCancelN, 1.0f);

    auto naive = itl::arnoldi(A, v0, 1, itl::eigen_which::largest_algebraic, 1);
    auto wide  = itl::arnoldi<double>(A, v0, 1, itl::eigen_which::largest_algebraic, 1);

    const double r_naive = naive.values(0).real();
    const double r_wide  = wide.values(0).real();
    INFO("exact=" << kCancelRayleigh << " naive=" << r_naive << " wide=" << r_wide);
    CHECK(r_naive == 0.0);
    CHECK(std::abs(r_wide - kCancelRayleigh) < 1e-2 * kCancelRayleigh);
}
