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
#include <vector>

using namespace mtl;
using cplx = std::complex<double>;

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
