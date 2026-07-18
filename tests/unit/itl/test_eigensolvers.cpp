#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
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
