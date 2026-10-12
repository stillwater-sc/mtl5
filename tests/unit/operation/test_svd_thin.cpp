// Thin svd(A, U, s, V): U m x k, s length k, V n x k with k = min(m, n).
//
// Inputs come from mt19937_64 mapped to uniforms by hand: the engine is fully
// specified, so every platform sees the same matrices.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
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

/// Random m x n matrix, column j scaled by 10^(cgrade*j/(n-1)).
template <typename T>
mat::dense2D<T> graded(std::size_t m, std::size_t n, uniform_source& u, double cgrade = 0.0) {
    mat::dense2D<T> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = T(u() * std::pow(10.0, cgrade * double(j) / double(n > 1 ? n - 1 : 1)));
    return A;
}

/// Same value and sign. Not memcmp: x86 long double is 80 bits in a 16-byte
/// object, and the 6 padding bytes are indeterminate.
template <typename T>
bool same_value(const T& a, const T& b) { return a == b && std::signbit(a) == std::signbit(b); }

/// ||Q^T Q - I||_F for the columns of Q.
template <typename T>
double orthogonality_error(const mat::dense2D<T>& Q) {
    // Summed in T: a long double Q is orthogonal to far below double's eps.
    T err = 0;
    for (std::size_t i = 0; i < Q.num_cols(); ++i)
        for (std::size_t j = 0; j < Q.num_cols(); ++j) {
            T d = 0;
            for (std::size_t r = 0; r < Q.num_rows(); ++r) d += Q(r, i) * Q(r, j);
            d -= (i == j ? T(1) : T(0));
            err += d * d;
        }
    return double(std::sqrt(err));
}

/// max |A - U diag(s) V^T| / max |A|.
template <typename T>
double reconstruction_error(const mat::dense2D<T>& A, const mat::dense2D<T>& U,
                            const vec::dense_vector<T>& s, const mat::dense2D<T>& V) {
    T worst = 0, amax = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) {
            T r = 0;
            for (std::size_t c = 0; c < s.size(); ++c) r += U(i, c) * s(c) * V(j, c);
            worst = std::max(worst, T(std::abs(r - A(i, j))));
            amax = std::max(amax, T(std::abs(A(i, j))));
        }
    return double(amax > T(0) ? worst / amax : worst);
}

} // namespace

TEST_CASE("thin svd: bit-identical to the leading columns of the full svd when m <= n",
          "[operation][svd]") {
    // Both run one-sided Jacobi on A itself and normalize the same columns.
    // long double keeps the full svd off LAPACK in a LAPACK build.
    uniform_source u(5711);
    for (auto [m, n] : std::vector<std::pair<std::size_t, std::size_t>>{{1, 1}, {3, 3}, {3, 5}, {9, 9}, {12, 30}}) {
        auto A = graded<long double>(m, n, u);
        mat::dense2D<long double> Uf, Sf, Vf, U, V;
        vec::dense_vector<long double> s;
        svd(A, Uf, Sf, Vf);
        svd(A, U, s, V);
        INFO(m << " x " << n);
        REQUIRE(U.num_rows() == m);
        REQUIRE(U.num_cols() == m);
        REQUIRE(s.size() == m);
        REQUIRE(V.num_rows() == n);
        REQUIRE(V.num_cols() == m);
        for (std::size_t j = 0; j < m; ++j) {
            REQUIRE(same_value(s(j), Sf(j, j)));
            for (std::size_t i = 0; i < m; ++i) REQUIRE(same_value(U(i, j), Uf(i, j)));
            for (std::size_t i = 0; i < n; ++i) REQUIRE(same_value(V(i, j), Vf(i, j)));
        }
    }
}

TEST_CASE("thin svd: tall QR-preconditioned case matches the full svd to rounding",
          "[operation][svd]") {
    // For m > n the generic path runs Jacobi on R from A = QR and forms
    // U = Q [U_R; 0]. Compared in long double against the full generic svd,
    // including columns scaled across 1e8, where both are relatively accurate.
    struct shape { std::size_t m, n; double cgrade; };
    const shape shapes[] = {{4, 3, 0}, {10, 3, 0}, {27, 9, 0}, {400, 12, 0}, {27, 9, 6}, {400, 12, 8}};
    uniform_source u(5712);
    const double eps = double(std::numeric_limits<long double>::epsilon());
    for (const auto& sh : shapes) {
        auto A = graded<long double>(sh.m, sh.n, u, sh.cgrade);
        mat::dense2D<long double> Uf, Sf, Vf, U, V;
        vec::dense_vector<long double> s;
        svd(A, Uf, Sf, Vf);
        svd(A, U, s, V);
        INFO(sh.m << " x " << sh.n << ", grade 1e" << sh.cgrade);
        REQUIRE(U.num_rows() == sh.m);
        REQUIRE(U.num_cols() == sh.n);
        REQUIRE(V.num_cols() == sh.n);
        REQUIRE(orthogonality_error(U) < 10.0 * double(sh.m) * eps);
        REQUIRE(orthogonality_error(V) < 10.0 * double(sh.n) * eps);
        REQUIRE(reconstruction_error(A, U, s, V) < 10.0 * double(sh.m) * eps);
        for (std::size_t j = 0; j < sh.n; ++j) {
            REQUIRE(std::abs(double(s(j) - Sf(j, j))) <= 100.0 * eps * double(Sf(j, j)));
            // Each singular pair (u_j, v_j) is fixed up to one common sign.
            long double du = 0, dv = 0;
            for (std::size_t i = 0; i < sh.m; ++i) du += U(i, j) * Uf(i, j);
            for (std::size_t i = 0; i < sh.n; ++i) dv += V(i, j) * Vf(i, j);
            REQUIRE(double(1.0L - std::abs(du)) < 100.0 * eps);
            REQUIRE(double(1.0L - std::abs(dv)) < 100.0 * eps);
            REQUIRE((du > 0) == (dv > 0));
        }
    }
}

TEST_CASE("thin svd: double agrees with the full svd, LAPACK build or not",
          "[operation][svd]") {
    // In a LAPACK build both calls go to gesdd, thin ('S') against full ('A');
    // otherwise both are generic. Either way they must agree.
    uniform_source u(5713);
    const double eps = std::numeric_limits<double>::epsilon();
    for (auto [m, n] : std::vector<std::pair<std::size_t, std::size_t>>{{10, 3}, {64, 9}, {400, 12}, {5, 5}, {3, 8}}) {
        auto A = graded<double>(m, n, u);
        const std::size_t k = std::min(m, n);
        mat::dense2D<double> Uf, Sf, Vf, U, V;
        vec::dense_vector<double> s;
        svd(A, Uf, Sf, Vf);
        svd(A, U, s, V);
        INFO(m << " x " << n);
        REQUIRE(U.num_rows() == m);
        REQUIRE(U.num_cols() == k);
        REQUIRE(s.size() == k);
        REQUIRE(V.num_rows() == n);
        REQUIRE(V.num_cols() == k);
        REQUIRE(orthogonality_error(U) < 10.0 * double(m) * eps);
        REQUIRE(orthogonality_error(V) < 10.0 * double(n) * eps);
        REQUIRE(reconstruction_error(A, U, s, V) < 10.0 * double(m) * eps);
        for (std::size_t j = 0; j < k; ++j) {
            if (j > 0) REQUIRE(s(j) <= s(j - 1));
            REQUIRE(std::abs(s(j) - Sf(j, j)) <= 100.0 * eps * Sf(0, 0));
            double du = 0.0;
            for (std::size_t i = 0; i < m; ++i) du += U(i, j) * Uf(i, j);
            REQUIRE(1.0 - std::abs(du) < 1e-12);
        }
    }
}

TEST_CASE("thin svd: rank-deficient, zero and empty inputs", "[operation][svd]") {
    uniform_source u(5714);
    const double eps = std::numeric_limits<long double>::digits > 53
                     ? double(std::numeric_limits<long double>::epsilon())
                     : std::numeric_limits<double>::epsilon();
    // Tall rank 3 of 4: U's column for the zero singular value comes from the
    // completion, inside the range of Q, and must still be orthonormal. long
    // double keeps it on the generic path.
    {
        auto A = graded<long double>(20, 4, u);
        for (std::size_t i = 0; i < 20; ++i) A(i, 3) = A(i, 0) + A(i, 1);
        mat::dense2D<long double> U, V;
        vec::dense_vector<long double> s;
        svd(A, U, s, V);
        REQUIRE(double(s(3)) < 100.0 * eps * double(s(0)));
        REQUIRE(orthogonality_error(U) < 1000.0 * eps);
        REQUIRE(reconstruction_error(A, U, s, V) < 1000.0 * eps);
    }
    // Zero 5 x 3: all singular values zero, U orthonormal, V the identity.
    {
        mat::dense2D<long double> A(5, 3), U, V;
        for (std::size_t i = 0; i < 5; ++i)
            for (std::size_t j = 0; j < 3; ++j) A(i, j) = 0.0L;
        vec::dense_vector<long double> s;
        svd(A, U, s, V);
        for (std::size_t j = 0; j < 3; ++j) REQUIRE(s(j) == 0.0L);
        REQUIRE(orthogonality_error(U) < 10.0 * eps);
        REQUIRE(orthogonality_error(V) == 0.0);
    }
    // Empty: k = 0, so U is m x 0 and V is n x 0.
    for (auto [m, n] : std::vector<std::pair<std::size_t, std::size_t>>{{3, 0}, {0, 3}, {0, 0}}) {
        mat::dense2D<double> A(m, n), U, V;
        vec::dense_vector<double> s;
        svd(A, U, s, V);
        REQUIRE(U.num_rows() == m);
        REQUIRE(U.num_cols() == 0);
        REQUIRE(s.size() == 0);
        REQUIRE(V.num_rows() == n);
        REQUIRE(V.num_cols() == 0);
    }
}

#ifdef MTL5_HAS_LAPACK
TEST_CASE("svd: a gesdd failure throws instead of returning garbage", "[operation][svd]") {
    // gesdd rejects a NaN in A with info = -4 (LAPACK 3.10+). Both the full
    // ('A') and the thin ('S') svd must report it rather than copy out
    // whatever the output buffers hold.
    mat::dense2D<double> A(4, 3);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = double(i + j);
    A(2, 1) = std::numeric_limits<double>::quiet_NaN();
    mat::dense2D<double> U, S, V;
    vec::dense_vector<double> s;
    REQUIRE_THROWS_AS(svd(A, U, S, V), std::runtime_error);
    REQUIRE_THROWS_AS(svd(A, U, s, V), std::runtime_error);
}
#endif
