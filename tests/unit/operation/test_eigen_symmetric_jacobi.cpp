// eigen_symmetric_jacobi: two-sided cyclic Jacobi eigensolver (#564).
//
// Inputs come from mt19937_64 mapped to uniforms by hand: the engine is fully
// specified, so every platform sees the same matrices (std::*_distribution is
// not specified bit for bit).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/eigenvalue_symmetric_jacobi.hpp>

using namespace mtl;

namespace {

struct uniform_source {
    std::mt19937_64 gen;
    explicit uniform_source(std::uint64_t seed) : gen(seed) {}
    double operator()() { return double(gen() >> 11) * 0x1.0p-53 * 2.0 - 1.0; }   // [-1, 1)
};

template <typename T>
mat::dense2D<T> random_symmetric(std::size_t n, uniform_source& u) {
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            const T v = T(u());
            A(i, j) = v;
            A(j, i) = v;
        }
    return A;
}

/// M = A^T A with A's 2N rows projected orthogonal to a random unit vector x,
/// so x spans M's null space. x is returned through the second argument.
mat::dense2D<double> normal_with_null(std::size_t N, std::vector<double>& x, uniform_source& u) {
    x.assign(N, 0.0);
    double nx = 0.0;
    for (auto& xi : x) { xi = u(); nx += xi * xi; }
    for (auto& xi : x) xi /= std::sqrt(nx);
    const std::size_t m = 2 * N;
    mat::dense2D<double> A(m, N);
    for (std::size_t r = 0; r < m; ++r) {
        std::vector<double> row(N);
        double d = 0.0;
        for (std::size_t j = 0; j < N; ++j) { row[j] = u(); d += row[j] * x[j]; }
        for (std::size_t j = 0; j < N; ++j) A(r, j) = row[j] - d * x[j];
    }
    mat::dense2D<double> M(N, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < m; ++r) s += A(r, i) * A(r, j);
            M(i, j) = s;
        }
    return M;
}

/// ||A - V diag(lambda) V^T||_F / ||A||_F, formed in long double.
template <typename T>
long double reconstruction_error(const mat::dense2D<T>& A, const vec::dense_vector<T>& lambda,
                                 const mat::dense2D<T>& V) {
    const std::size_t n = A.num_rows();
    long double err = 0, nrm = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            long double s = 0;
            for (std::size_t k = 0; k < n; ++k)
                s += (long double)V(i, k) * (long double)lambda(k) * (long double)V(j, k);
            const long double d = (long double)A(i, j) - s;
            err += d * d;
            nrm += (long double)A(i, j) * (long double)A(i, j);
        }
    return nrm == 0 ? std::sqrt(err) : std::sqrt(err / nrm);
}

/// ||V^T V - I||_F, formed in long double.
template <typename T>
long double orthogonality_error(const mat::dense2D<T>& V) {
    const std::size_t n = V.num_rows();
    long double err = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            long double s = 0;
            for (std::size_t k = 0; k < n; ++k) s += (long double)V(k, i) * (long double)V(k, j);
            const long double d = s - (i == j ? 1.0L : 0.0L);
            err += d * d;
        }
    return std::sqrt(err);
}

template <typename T>
bool ascending(const vec::dense_vector<T>& e) {
    for (std::size_t i = 1; i < e.size(); ++i)
        if (e(i) < e(i - 1)) return false;
    return true;
}

} // namespace

TEST_CASE("eigen_symmetric_jacobi: reconstruction and orthogonality", "[operation][eigen][jacobi]") {
    uniform_source u(564);
    for (std::size_t n = 1; n <= 16; ++n) {
        auto A = random_symmetric<double>(n, u);
        auto [eigs, V] = eigen_symmetric_jacobi(A);
        REQUIRE(eigs.size() == n);
        REQUIRE(ascending(eigs));
        const double eps = std::numeric_limits<double>::epsilon();
        REQUIRE(reconstruction_error(A, eigs, V) <= 4.0L * n * eps);
        REQUIRE(orthogonality_error(V) <= 4.0L * n * eps);
    }
}

TEST_CASE("eigen_symmetric_jacobi: float", "[operation][eigen][jacobi]") {
    uniform_source u(5640);
    for (std::size_t n : {3u, 8u, 12u}) {
        auto A = random_symmetric<float>(n, u);
        auto [eigs, V] = eigen_symmetric_jacobi(A);
        REQUIRE(ascending(eigs));
        const float eps = std::numeric_limits<float>::epsilon();
        REQUIRE(reconstruction_error(A, eigs, V) <= 4.0L * n * eps);
        REQUIRE(orthogonality_error(V) <= 4.0L * n * eps);
    }
}

TEST_CASE("eigen_symmetric_jacobi: agrees with eigen_symmetric", "[operation][eigen][jacobi]") {
    uniform_source u(5641);
    const std::size_t n = 10;
    auto A = random_symmetric<double>(n, u);
    auto [ej, Vj] = eigen_symmetric_jacobi(A);
    auto [eq, Vq] = eigen_symmetric(A);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(ej(i), Catch::Matchers::WithinAbs(eq(i), 1e-13));
}

TEST_CASE("eigen_symmetric_jacobi: converges in under 20 sweeps on normal matrices",
          "[operation][eigen][jacobi]") {
    // A run capped at 19 sweeps returns the same bits as an uncapped one only if
    // it had already stopped, i.e. finished a sweep with no rotation.
    uniform_source u(5642);
    for (std::size_t N : {9u, 12u}) {
        for (int t = 0; t < 10; ++t) {
            std::vector<double> x;
            auto M = normal_with_null(N, x, u);
            auto [e_full, V_full] = eigen_symmetric_jacobi(M);
            auto [e_cap, V_cap]   = eigen_symmetric_jacobi(M, std::numeric_limits<double>::epsilon(), 19);
            for (std::size_t i = 0; i < N; ++i) {
                REQUIRE(e_cap(i) == e_full(i));
                for (std::size_t j = 0; j < N; ++j) REQUIRE(V_cap(i, j) == V_full(i, j));
            }
            // And the null vector is recovered to working precision.
            double dot = 0.0;
            for (std::size_t i = 0; i < N; ++i) dot += V_full(i, 0) * x[i];
            REQUIRE(std::abs(std::abs(dot) - 1.0) < 1e-14);
        }
    }
}

TEST_CASE("eigen_symmetric_jacobi: matrices scaled by 1e+160 and 1e-160", "[operation][eigen][jacobi]") {
    // Squared entries would overflow (1e320) or underflow (1e-320) here; the
    // solver must neither skip rotations nor lose accuracy.
    uniform_source u(5643);
    const std::size_t n = 9;
    auto A = random_symmetric<double>(n, u);
    auto [e1, V1] = eigen_symmetric_jacobi(A);
    for (double scale : {1e160, 1e-160}) {
        mat::dense2D<double> B(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) B(i, j) = A(i, j) * scale;
        auto [es, Vs] = eigen_symmetric_jacobi(B);
        REQUIRE(orthogonality_error(Vs) <= 4.0L * n * std::numeric_limits<double>::epsilon());
        REQUIRE(reconstruction_error(B, es, Vs) <= 4.0L * n * std::numeric_limits<double>::epsilon());
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE_THAT(es(i) / scale, Catch::Matchers::WithinAbs(e1(i), 1e-14));
    }
}

TEST_CASE("eigen_symmetric_jacobi: relative accuracy on a graded positive definite matrix",
          "[operation][eigen][jacobi]") {
    // A = D B D with B well conditioned and D = diag(1 ... 1e-8): eigenvalues
    // span 16 decades. Jacobi gets every one to a few eps RELATIVELY; tridiagonal
    // QR is accurate only to eps*||A||, which for the smallest is ~1e-11 relative.
    // The reference is the same algorithm in long double, which is only a
    // reference when long double is wider than double (not on MSVC or Apple).
    if (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) {
        SUCCEED("long double is double on this platform; no wider reference");
        return;
    }
    uniform_source u(5644);
    const std::size_t n = 8;
    for (int t = 0; t < 10; ++t) {
        mat::dense2D<double> G(n, n), A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) G(i, j) = u();
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double s = (i == j) ? double(n) : 0.0;
                for (std::size_t r = 0; r < n; ++r) s += G(r, i) * G(r, j);
                const double di = std::pow(10.0, -8.0 * double(i) / double(n - 1));
                const double dj = std::pow(10.0, -8.0 * double(j) / double(n - 1));
                A(i, j) = di * s * dj;
            }
        mat::dense2D<long double> AL(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) AL(i, j) = A(i, j);
        auto ref = eigen_symmetric_jacobi(AL).eigenvalues;
        auto e   = eigen_symmetric_jacobi(A).eigenvalues;
        REQUIRE(double(ref(0)) < 1e-15 * double(ref(n - 1)));   // genuinely graded
        for (std::size_t i = 0; i < n; ++i) {
            const long double rel = std::abs(((long double)e(i) - ref(i)) / ref(i));
            REQUIRE(rel < 1e-14L);
        }
    }
}

TEST_CASE("eigen_symmetric_jacobi: an entry coupling two small eigenvalues is rotated",
          "[operation][eigen][jacobi]") {
    // diag(1, 1e-10, 2e-10) with a(1,2) = 1.5e-16. Against ||A||_F = 1 that entry
    // is below eps, so a stop on off-diagonal mass <= eps^2 ||A||_F^2 returns
    // 1e-10 and 2e-10 untouched, 2.25e-12 relatively wrong. Against the diagonal
    // entries it couples (sqrt(1e-10 * 2e-10) = 1.4e-10) it is not negligible,
    // so the per-pair relative test rotates it.
    const double a = 1e-10, c = 2e-10, b = 1.5e-16;
    mat::dense2D<double> A(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = 0.0;
    A(0, 0) = 1.0;
    A(1, 1) = a; A(2, 2) = c;
    A(1, 2) = b; A(2, 1) = b;

    const double lmax = (a + c) / 2 + std::sqrt(((c - a) / 2) * ((c - a) / 2) + b * b);
    const double lmin = (a * c - b * b) / lmax;
    auto [eigs, V] = eigen_symmetric_jacobi(A);
    REQUIRE(std::abs(eigs(0) - lmin) / lmin < 1e-15);
    REQUIRE(std::abs(eigs(1) - lmax) / lmax < 1e-15);
    REQUIRE(eigs(2) == 1.0);
}

TEST_CASE("eigen_symmetric_jacobi: exact zero eigenvalues and trivial sizes", "[operation][eigen][jacobi]") {
    for (std::size_t n : {2u, 3u, 6u}) {
        mat::dense2D<double> J(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) J(i, j) = 1.0;
        auto [eigs, V] = eigen_symmetric_jacobi(J);
        for (std::size_t k = 0; k + 1 < n; ++k) REQUIRE(std::abs(eigs(k)) < double(n) * 1e-15);
        REQUIRE_THAT(eigs(n - 1), Catch::Matchers::WithinAbs(double(n), double(n) * 1e-15));
        REQUIRE(orthogonality_error(V) < double(n) * 1e-15);
    }

    mat::dense2D<double> P(2, 2);
    P(0, 0) = 0; P(0, 1) = 1;
    P(1, 0) = 1; P(1, 1) = 0;
    auto [ep, Vp] = eigen_symmetric_jacobi(P);
    REQUIRE_THAT(ep(0), Catch::Matchers::WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(ep(1), Catch::Matchers::WithinAbs(1.0, 1e-15));

    mat::dense2D<double> Z(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) Z(i, j) = 0.0;
    auto [ez, Vz] = eigen_symmetric_jacobi(Z);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(ez(i) == 0.0);
        for (std::size_t j = 0; j < 4; ++j) REQUIRE(Vz(i, j) == (i == j ? 1.0 : 0.0));
    }

    mat::dense2D<double> E(0, 0);
    auto [e0, V0] = eigen_symmetric_jacobi(E);
    REQUIRE(e0.size() == 0);
}
