#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/sparse/eigen/shift_invert.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

using namespace mtl;
using cplx = std::complex<double>;

namespace counting {

/// Configuration-3 shape (the ADL hook) that counts its products (#585).
struct count_acc { double s; };
inline std::size_t products = 0;

inline void accumulator_clear(count_acc& a) { a.s = 0.0; }
template <typename V>
void accumulator_assign(count_acc& a, const V& v) { a.s = static_cast<double>(v); }
template <typename V>
void accumulator_add_product(count_acc& a, const V& m, const V& v) {
    ++products;
    a.s += static_cast<double>(m) * static_cast<double>(v);
}
template <typename R>
R accumulator_value(const count_acc& a, std::type_identity<R>) { return static_cast<R>(a.s); }

} // namespace counting

namespace {

// 1D Laplacian as a sparse CSR matrix (SPD tridiagonal).
mat::compressed2D<double> laplacian1d(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0)     ins[i][i - 1] << -1.0;
        ins[i][i] << 2.0;
        if (i + 1 < n) ins[i][i + 1] << -1.0;
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

// Max residual ||A y - lambda y|| over the returned (complex) eigenpairs.
double max_residual(const mat::compressed2D<double>& A,
                    const itl::ritz_pairs<cplx>& r) {
    const std::size_t n = A.num_rows();
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double worst = 0.0;
    for (std::size_t c = 0; c < r.values.size(); ++c) {
        for (std::size_t i = 0; i < n; ++i) {
            cplx Ay(0.0, 0.0);
            for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
                Ay += cplx(dat[k], 0.0) * r.vectors(ci[k], c);
            Ay -= r.values(c) * r.vectors(i, c);
            worst = std::max(worst, std::abs(Ay));
        }
    }
    return worst;
}

} // namespace

TEST_CASE("sparse_eigs: largest-magnitude eigenvalues (direct Arnoldi)", "[sparse][eigen]") {
    const std::size_t n = 40;
    auto A = laplacian1d(n);
    auto expected = laplacian1d_eigs(n);

    // The Laplacian spectrum clusters near its ends, so the extreme top
    // eigenvalue converges slowly for a small Krylov subspace; use a full
    // subspace here to validate the sparse_eigs path deterministically.
    auto r = sparse::sparse_eigs(A, 3, itl::eigen_which::largest_magnitude, n);
    REQUIRE(r.values.size() == 3);
    REQUIRE(r.converged);

    // For the SPD Laplacian, largest magnitude == largest algebraic.
    REQUIRE_THAT(r.values(0).real(), Catch::Matchers::WithinAbs(expected[n-1], 1e-7));
    REQUIRE_THAT(r.values(0).imag(), Catch::Matchers::WithinAbs(0.0, 1e-9));
    REQUIRE(max_residual(A, r) < 1e-6);
}

TEST_CASE("sparse shift-invert: smallest eigenvalues near sigma=0", "[sparse][eigen][shift-invert]") {
    const std::size_t n = 40;
    auto A = laplacian1d(n);
    auto expected = laplacian1d_eigs(n);  // ascending; smallest are expected[0..]

    // sigma slightly below the smallest eigenvalue -> the k nearest are the
    // k smallest eigenvalues of A.
    auto r = sparse::sparse_eigs_shift_invert(A, 0.0, 3);
    REQUIRE(r.values.size() == 3);
    REQUIRE(r.converged);

    std::vector<double> got(3);
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(r.values(i).imag(), Catch::Matchers::WithinAbs(0.0, 1e-8));
        got[i] = r.values(i).real();
    }
    std::sort(got.begin(), got.end());
    REQUIRE_THAT(got[0], Catch::Matchers::WithinAbs(expected[0], 1e-7));
    REQUIRE_THAT(got[1], Catch::Matchers::WithinAbs(expected[1], 1e-7));
    REQUIRE_THAT(got[2], Catch::Matchers::WithinAbs(expected[2], 1e-7));

    // Eigenvectors (unchanged by shift-invert) satisfy A y = lambda y.
    REQUIRE(max_residual(A, r) < 1e-6);
}

TEST_CASE("sparse shift-invert: interior eigenvalues near a target", "[sparse][eigen][shift-invert]") {
    const std::size_t n = 50;
    auto A = laplacian1d(n);
    auto expected = laplacian1d_eigs(n);

    // Pick an interior target (off the midpoint between two eigenvalues, so the
    // nearest eigenvalue is unambiguous) and find the eigenvalue closest to it.
    const double sigma = 2.05;   // interior of the [0,4] Laplacian spectrum
    auto r = sparse::sparse_eigs_shift_invert(A, sigma, 4);
    REQUIRE(r.values.size() == 4);

    // The closest returned eigenvalue must match the closest true eigenvalue.
    double best_true = expected[0];
    for (double e : expected)
        if (std::abs(e - sigma) < std::abs(best_true - sigma)) best_true = e;

    double best_got = r.values(0).real();
    for (std::size_t i = 0; i < r.values.size(); ++i)
        if (std::abs(r.values(i).real() - sigma) < std::abs(best_got - sigma))
            best_got = r.values(i).real();

    REQUIRE_THAT(best_got, Catch::Matchers::WithinAbs(best_true, 1e-7));
    REQUIRE(max_residual(A, r) < 1e-6);
}

TEST_CASE("sparse shift-invert: nonsymmetric matrix", "[sparse][eigen][shift-invert]") {
    // Nonsymmetric sparse matrix with known real eigenvalues 1..n on the diagonal
    // plus a subdiagonal (upper/lower triangular -> eigenvalues are the diagonal).
    const std::size_t n = 8;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << static_cast<double>(i + 1);       // eigenvalues 1..n
            if (i + 1 < n) ins[i][i + 1] << 0.5;           // upper triangular perturbation
        }
    }

    // Find eigenvalues nearest sigma = 3.2 -> should include 3.
    auto r = sparse::sparse_eigs_shift_invert(A, 3.2, 2);
    REQUIRE(r.values.size() == 2);

    bool found3 = false;
    for (std::size_t i = 0; i < r.values.size(); ++i)
        if (std::abs(r.values(i) - cplx(3.0, 0.0)) < 1e-6) found3 = true;
    REQUIRE(found3);
    REQUIRE(max_residual(A, r) < 1e-6);
}

TEST_CASE("sparse eigensolvers: the Accumulator reaches the factorization, the solves and Arnoldi (#585)",
          "[sparse][eigen][shift-invert][accumulator]") {
    const std::size_t n = 30;
    auto A = laplacian1d(n);
    vec::dense_vector<double> x(n, 1.0);

    SECTION("shift_invert_operator: factorization and every application") {
        counting::products = 0;
        sparse::shift_invert_operator<double, counting::count_acc> op(A, 0.1);
        const std::size_t factor = counting::products;
        CHECK(factor > 0);
        counting::products = 0;
        (void)(op * x);
        const std::size_t one = counting::products;
        CHECK(one > 0);
        counting::products = 0;
        (void)(op * x);
        CHECK(counting::products == one);        // each application, same work

        counting::products = 0;
        sparse::shift_invert_operator<double> v(A, 0.1);
        (void)(v * x);
        CHECK(counting::products == 0);
    }
    SECTION("sparse_eigs_shift_invert and sparse_eigs") {
        counting::products = 0;
        (void)sparse::sparse_eigs_shift_invert<counting::count_acc>(A, 0.0, 3);
        CHECK(counting::products > 0);
        counting::products = 0;
        (void)sparse::sparse_eigs<counting::count_acc>(A, 3, itl::eigen_which::largest_magnitude, n);
        CHECK(counting::products > 0);
        counting::products = 0;
        (void)sparse::sparse_eigs_shift_invert(A, 0.0, 3);
        (void)sparse::sparse_eigs(A, 3, itl::eigen_which::largest_magnitude, n);
        CHECK(counting::products == 0);
    }
}

TEST_CASE("sparse eigensolvers: an accumulated run finds the same eigenvalues (#585)",
          "[sparse][eigen][shift-invert][accumulator]") {
    const std::size_t n = 40;
    auto A = laplacian1d(n);
    auto expected = laplacian1d_eigs(n);

    auto r = sparse::sparse_eigs_shift_invert<double>(A, 0.0, 3);
    REQUIRE(r.converged);
    std::vector<double> got(3);
    for (std::size_t i = 0; i < 3; ++i) got[i] = r.values(i).real();
    std::sort(got.begin(), got.end());
    for (std::size_t i = 0; i < 3; ++i)
        CHECK_THAT(got[i], Catch::Matchers::WithinAbs(expected[i], 1e-7));
    CHECK(max_residual(A, r) < 1e-6);

    auto s = sparse::sparse_eigs<double>(A, 3, itl::eigen_which::largest_magnitude, n);
    std::vector<double> top(3);
    for (std::size_t i = 0; i < 3; ++i) top[i] = s.values(i).real();
    std::sort(top.begin(), top.end());
    for (std::size_t i = 0; i < 3; ++i)
        CHECK_THAT(top[i], Catch::Matchers::WithinAbs(expected[n - 3 + i], 1e-7));
}

// The perturbation floor is relative to each pivot column (issue 907). It was
// passed in already multiplied by max|A - sigma I|, and sparse_lu_numeric
// multiplies by the column norm again, so it grew as the magnitude squared:
// at s = 1e6 the eigenvalue s of s * tridiag(-1, 2, -1) came out 1.00106e6,
// at 1e10 complex, both reported converged. With n = 20, s itself is an
// eigenvalue (2 - 2 cos(7 pi / 21) = 1), so A - s I is exactly singular and
// the perturbation is what the factorization runs on.
TEST_CASE("sparse shift-invert: the perturbation floor does not depend on the scale",
          "[sparse][eigen][shift-invert][907]") {
    const std::size_t n = 20;
    for (const double s : {1e-6, 1.0, 1e3, 1e6, 1e10}) {
        CAPTURE(s);
        auto A = laplacian1d(n);
        for (auto& v : A.ref_data()) v *= s;
        auto r = sparse::sparse_eigs_shift_invert(A, s, 1);
        REQUIRE(r.values.size() == 1);
        REQUIRE(r.converged);
        REQUIRE(std::abs(r.values(0).imag()) <= 1e-8 * s);
        REQUIRE(std::abs(r.values(0).real() - s) <= 1e-8 * s);
    }
}

// shift_diagonal merges -sigma into each row in one pass (issue 907): rows
// with the diagonal first, last, in the middle and missing.
TEST_CASE("shift_diagonal: the diagonal wherever it falls, or missing", "[sparse][eigen][907]") {
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][2] << 2.0;   // diagonal first
        ins[1][0] << 3.0; ins[1][3] << 4.0;   // diagonal missing, between columns
        ins[2][0] << 5.0; ins[2][2] << 6.0; ins[2][3] << 7.0;   // middle
        ins[3][1] << 8.0;                     // missing, past the last column
    }
    const auto B = sparse::shift_diagonal(A, 0.5);
    const double want[4][4] = {{0.5, 0, 2, 0}, {3, -0.5, 0, 4}, {5, 0, 5.5, 7}, {0, 8, 0, -0.5}};
    const auto& rp = B.ref_major(); const auto& ci = B.ref_minor(); const auto& d = B.ref_data();
    REQUIRE(d.size() == 10);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t k = rp[i] + 1; k < rp[i + 1]; ++k) REQUIRE(ci[k] > ci[k - 1]);   // sorted
        double row[4] = {0, 0, 0, 0};
        bool diag = false;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) { row[ci[k]] = d[k]; diag = diag || ci[k] == i; }
        REQUIRE(diag);
        for (std::size_t j = 0; j < 4; ++j) REQUIRE(row[j] == want[i][j]);
    }
}
