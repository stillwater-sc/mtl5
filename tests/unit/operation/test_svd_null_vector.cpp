// V-only svd(A, s, V) and null_vector(A) (#566).
//
// Inputs come from mt19937_64 mapped to uniforms by hand: the engine is fully
// specified, so every platform sees the same matrices.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/svd.hpp>

using namespace mtl;

namespace {

struct uniform_source {
    std::mt19937_64 gen;
    explicit uniform_source(std::uint64_t seed) : gen(seed) {}
    double operator()() { return double(gen() >> 11) * 0x1.0p-53 * 2.0 - 1.0; }   // [-1, 1)
};

/// Random m x n matrix, column j scaled by 10^(cgrade*j/(n-1)) and row i by
/// 10^(rgrade*i/(m-1)).
template <typename T>
mat::dense2D<T> graded(std::size_t m, std::size_t n, uniform_source& u,
                       double cgrade = 0.0, double rgrade = 0.0) {
    mat::dense2D<T> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double c = std::pow(10.0, cgrade * double(j) / double(n > 1 ? n - 1 : 1));
            const double r = std::pow(10.0, rgrade * double(i) / double(m > 1 ? m - 1 : 1));
            A(i, j) = T(u() * c * r);
        }
    return A;
}

/// Same value and sign. Not memcmp: x86 long double is 80 bits in a 16-byte
/// object, and the 6 padding bytes are indeterminate.
template <typename T>
bool same_value(const T& a, const T& b) { return a == b && std::signbit(a) == std::signbit(b); }

/// ||A v_j|| for column j of V, scaled so it cannot overflow.
double image_norm(const mat::dense2D<double>& A, const mat::dense2D<double>& V, std::size_t j) {
    double s = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double r = 0.0;
        for (std::size_t k = 0; k < A.num_cols(); ++k) r += A(i, k) * V(k, j);
        s += r * r;
    }
    return std::sqrt(s);
}

double orthogonality_error(const mat::dense2D<double>& V) {
    const std::size_t n = V.num_rows();
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += V(k, i) * V(k, j);
            const double d = s - (i == j ? 1.0 : 0.0);
            err += d * d;
        }
    return std::sqrt(err);
}

} // namespace

TEST_CASE("svd(A, s, V): bit-identical to svd(A, U, S, V) when m <= n", "[operation][svd]") {
    // Without the QR step both run the same one-sided Jacobi on A. long double
    // keeps the full svd off LAPACK in a LAPACK build, so the comparison is
    // generic against generic on every platform.
    uniform_source u(566);
    for (auto [m, n] : std::vector<std::pair<std::size_t, std::size_t>>{{1, 1}, {3, 3}, {3, 5}, {9, 9}, {12, 30}}) {
        auto A = graded<long double>(m, n, u);
        mat::dense2D<long double> U, S, V, V2;
        vec::dense_vector<long double> s;
        svd(A, U, S, V);
        svd(A, s, V2);
        REQUIRE(s.size() == std::min(m, n));
        for (std::size_t j = 0; j < s.size(); ++j) REQUIRE(same_value(s(j), S(j, j)));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) REQUIRE(same_value(V(i, j), V2(i, j)));
    }
}

TEST_CASE("svd(A, s, V): QR-preconditioned tall case matches svd to rounding", "[operation][svd]") {
    // For m > n the V-only path runs Jacobi on R from A = QR. Singular values
    // and right singular vectors must agree with the full one-sided Jacobi svd
    // to rounding,
    // including when A's columns or rows are scaled across 1e8.
    struct shape { std::size_t m, n; double cgrade, rgrade; };
    const shape shapes[] = {
        {10, 3, 0, 0}, {27, 9, 0, 0}, {36, 12, 0, 0}, {400, 12, 0, 0},
        {27, 9, 6, 0}, {27, 9, 0, 6}, {400, 12, 8, 0}, {400, 12, 0, 8},
    };
    uniform_source u(5660);
    const double eps = std::numeric_limits<double>::epsilon();
    for (const auto& sh : shapes) {
        auto A = graded<double>(sh.m, sh.n, u, sh.cgrade, sh.rgrade);
        // Reference: the full generic svd of the same matrix in long double.
        // Not the double svd: in a LAPACK build that is gesdd, which is only
        // absolutely accurate, and on the graded cases its small singular
        // values are themselves off by more than the bound below.
        mat::dense2D<long double> AL(sh.m, sh.n), UL, SL, VL;
        for (std::size_t i = 0; i < sh.m; ++i)
            for (std::size_t j = 0; j < sh.n; ++j) AL(i, j) = A(i, j);
        svd(AL, UL, SL, VL);
        mat::dense2D<double> S(sh.n, sh.n), V(sh.n, sh.n), V2;
        for (std::size_t i = 0; i < sh.n; ++i)
            for (std::size_t j = 0; j < sh.n; ++j) {
                S(i, j) = double(SL(i, j));
                V(i, j) = double(VL(i, j));
            }
        vec::dense_vector<double> s;
        svd(A, s, V2);
        REQUIRE(s.size() == sh.n);
        REQUIRE(orthogonality_error(V2) < 10.0 * double(sh.n) * eps);
        for (std::size_t j = 0; j < sh.n; ++j) {
            if (j > 0) REQUIRE(s(j) <= s(j - 1));
            REQUIRE(std::abs(s(j) - S(j, j)) <= 100.0 * eps * S(j, j));
            double d = 0.0;
            for (std::size_t i = 0; i < sh.n; ++i) d += V(i, j) * V2(i, j);
            REQUIRE(1.0 - std::abs(d) < 100.0 * eps);     // same vector up to sign
            REQUIRE(std::abs(image_norm(A, V2, j) - s(j)) <= 100.0 * eps * s(0));
        }
    }
}

TEST_CASE("null_vector: recovers a known null vector of tall 2k x 9 and 2k x 12 systems",
          "[operation][svd]") {
    uniform_source u(5661);
    for (std::size_t N : {9u, 12u}) {
        for (std::size_t k : {8u, 32u, 200u}) {
            const std::size_t m = 2 * k;
            std::vector<double> x(N);
            double nx = 0.0;
            for (auto& xi : x) { xi = u(); nx += xi * xi; }
            for (auto& xi : x) xi /= std::sqrt(nx);
            mat::dense2D<double> A(m, N);
            for (std::size_t r = 0; r < m; ++r) {
                std::vector<double> row(N);
                double d = 0.0;
                for (std::size_t j = 0; j < N; ++j) { row[j] = u(); d += row[j] * x[j]; }
                for (std::size_t j = 0; j < N; ++j) A(r, j) = row[j] - d * x[j];
            }
            auto v = null_vector(A);
            REQUIRE(v.size() == N);
            double dot = 0.0, nv = 0.0;
            for (std::size_t i = 0; i < N; ++i) { dot += v(i) * x[i]; nv += v(i) * v(i); }
            REQUIRE(std::abs(std::sqrt(nv) - 1.0) < 1e-14);
            REQUIRE(1.0 - std::abs(dot) < 1e-14);
        }
    }
}

TEST_CASE("null_vector: wide, square-singular and degenerate shapes", "[operation][svd]") {
    uniform_source u(5662);
    // Wide 3 x 5: a two-dimensional null space; any unit vector in it will do.
    {
        auto A = graded<double>(3, 5, u);
        auto v = null_vector(A);
        double nv = 0.0;
        for (std::size_t i = 0; i < 5; ++i) nv += v(i) * v(i);
        REQUIRE(std::abs(std::sqrt(nv) - 1.0) < 1e-14);
        for (std::size_t i = 0; i < 3; ++i) {
            double r = 0.0;
            for (std::size_t j = 0; j < 5; ++j) r += A(i, j) * v(j);
            REQUIRE(std::abs(r) < 1e-14);
        }
    }
    // Square, rank 3 of 4: column 3 = column 0 + column 1, so x = (1, 1, 0, -1)/sqrt(3).
    {
        auto A = graded<double>(4, 4, u);
        for (std::size_t i = 0; i < 4; ++i) A(i, 3) = A(i, 0) + A(i, 1);
        auto v = null_vector(A);
        const double r3 = 1.0 / std::sqrt(3.0);
        const double dot = v(0) * r3 + v(1) * r3 - v(3) * r3;
        REQUIRE(1.0 - std::abs(dot) < 1e-14);
    }
    // No columns: empty result. No rows: every unit vector is a null vector.
    {
        mat::dense2D<double> A(3, 0);
        REQUIRE(null_vector(A).size() == 0);
        mat::dense2D<double> B(0, 3);
        auto v = null_vector(B);
        REQUIRE(v.size() == 3);
        REQUIRE(v(0) * v(0) + v(1) * v(1) + v(2) * v(2) == 1.0);
        vec::dense_vector<double> s;
        mat::dense2D<double> V;
        svd(B, s, V);
        REQUIRE(s.size() == 0);
        REQUIRE(orthogonality_error(V) == 0.0);
    }
}
