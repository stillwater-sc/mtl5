#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/generators/frank.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

using namespace mtl;
using cplx = std::complex<double>;

namespace {

// Max_i | (A v_k - lambda_k v_k)_i | over all eigenpairs (columns of V).
double max_eigen_residual(const mat::dense2D<double>& A,
                          const vec::dense_vector<cplx>& eigs,
                          const mat::dense2D<cplx>& V) {
    const std::size_t n = A.num_rows();
    double worst = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            cplx axi(0.0, 0.0);
            for (std::size_t j = 0; j < n; ++j)
                axi += cplx(A(i, j), 0.0) * V(j, k);
            axi -= eigs(k) * V(i, k);
            worst = std::max(worst, std::abs(axi));
        }
    }
    return worst;
}

// 2-norm of column k of V.
double col_norm(const mat::dense2D<cplx>& V, std::size_t k) {
    double s = 0.0;
    for (std::size_t i = 0; i < V.num_rows(); ++i)
        s += std::norm(V(i, k));
    return std::sqrt(s);
}

} // namespace

TEST_CASE("General eigen: known 2x2 real eigenvalues", "[operation][eigen]") {
    // {{0,1},{-2,-3}} has eigenvalues -1, -2 (both real).
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0;  A(0,1) = 1;
    A(1,0) = -2; A(1,1) = -3;

    auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == 2);
    REQUIRE(V.num_rows() == 2);
    REQUIRE(V.num_cols() == 2);

    // Each eigenpair satisfies A v = lambda v; each eigenvector is unit-norm.
    REQUIRE(max_eigen_residual(A, eigs, V) < 1e-9);
    REQUIRE_THAT(col_norm(V, 0), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(col_norm(V, 1), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("General eigen: complex-conjugate pair", "[operation][eigen]") {
    // {{0,-1},{1,0}} has eigenvalues +/- i with complex eigenvectors.
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0; A(0,1) = -1;
    A(1,0) = 1; A(1,1) =  0;

    auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == 2);

    // Eigenvalues are purely imaginary +/- i.
    std::vector<double> im = {eigs(0).imag(), eigs(1).imag()};
    std::sort(im.begin(), im.end());
    REQUIRE_THAT(im[0], Catch::Matchers::WithinAbs(-1.0, 1e-8));
    REQUIRE_THAT(im[1], Catch::Matchers::WithinAbs( 1.0, 1e-8));

    // Genuinely complex eigenvectors must still satisfy the eigen relation.
    REQUIRE(max_eigen_residual(A, eigs, V) < 1e-9);
    REQUIRE_THAT(col_norm(V, 0), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(col_norm(V, 1), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("General eigen: diagonal matrix -> canonical basis", "[operation][eigen]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 2; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 5; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 7;

    auto [eigs, V] = eigen(A);
    REQUIRE(max_eigen_residual(A, eigs, V) < 1e-10);

    // Each eigenvector is a unit axis vector (single nonzero, magnitude 1).
    for (std::size_t k = 0; k < 3; ++k) {
        int nonzero = 0;
        for (std::size_t i = 0; i < 3; ++i)
            if (std::abs(V(i, k)) > 1e-8) ++nonzero;
        REQUIRE(nonzero == 1);
    }
}

TEST_CASE("General eigen: 1x1 matrix", "[operation][eigen]") {
    mat::dense2D<double> A(1, 1);
    A(0,0) = 3.5;

    auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == 1);
    REQUIRE_THAT(eigs(0).real(), Catch::Matchers::WithinAbs(3.5, 1e-12));
    REQUIRE_THAT(std::abs(V(0, 0)), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("General eigen: 0x0 matrix", "[operation][eigen]") {
    mat::dense2D<double> A(0, 0);
    auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == 0);
    REQUIRE(V.num_rows() == 0);
    REQUIRE(V.num_cols() == 0);
}

TEST_CASE("General eigen: Frank matrix eigenpairs", "[operation][eigen][generator]") {
    constexpr std::size_t n = 6;
    auto F = generators::frank<double>(n);

    auto [eigs, V] = eigen(F);
    REQUIRE(eigs.size() == n);
    REQUIRE(max_eigen_residual(F, eigs, V) < 1e-8);
    for (std::size_t k = 0; k < n; ++k)
        REQUIRE_THAT(col_norm(V, k), Catch::Matchers::WithinAbs(1.0, 1e-10));
}

TEST_CASE("General eigen: mixed real + complex-conjugate spectrum", "[operation][eigen]") {
    // Block-triangular 4x4 with eigenvalues 5, 2, and the pair 1 +/- 2i.
    // Exercises real eigenvectors and a genuine complex-conjugate eigenpair
    // together (the 2x2 trailing block deflates cleanly in the QR iteration).
    mat::dense2D<double> A(4, 4);
    const double B[4][4] = {{5, 1, 0,  0},
                            {0, 2, 1,  0},
                            {0, 0, 1, -2},
                            {0, 0, 2,  1}};
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = B[i][j];

    auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == 4);
    REQUIRE(max_eigen_residual(A, eigs, V) < 1e-10);
    for (std::size_t k = 0; k < 4; ++k)
        REQUIRE_THAT(col_norm(V, k), Catch::Matchers::WithinAbs(1.0, 1e-10));

    // Confirm the spectrum: two real (5, 2) and a conjugate pair 1 +/- 2i.
    int complex_pair = 0, real_vals = 0;
    for (std::size_t k = 0; k < 4; ++k) {
        if (std::abs(eigs(k).imag()) > 1e-8) ++complex_pair; else ++real_vals;
    }
    REQUIRE(real_vals == 2);
    REQUIRE(complex_pair == 2);
}

TEST_CASE("General eigen: repeated eigenvalue yields an independent basis", "[operation][eigen]") {
    // Identity: eigenvalue 1 with multiplicity 4. A correct result returns four
    // *independent* eigenvectors spanning R^4, not four copies of one vector.
    constexpr std::size_t n = 4;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (i == j) ? 1.0 : 0.0;

    auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == n);

    // Every column is an eigenvector (trivially, for I) and unit norm.
    REQUIRE(max_eigen_residual(A, eigs, V) < 1e-12);
    for (std::size_t k = 0; k < n; ++k)
        REQUIRE_THAT(col_norm(V, k), Catch::Matchers::WithinAbs(1.0, 1e-10));

    // Columns must be mutually orthogonal (an orthonormal basis of the
    // eigenspace) -- off-diagonal Gram entries near zero.
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a + 1; b < n; ++b) {
            cplx dot(0.0, 0.0);
            for (std::size_t i = 0; i < n; ++i)
                dot += std::conj(V(i, a)) * V(i, b);
            REQUIRE(std::abs(dot) < 1e-9);
        }
    }
}

TEST_CASE("General eigen: partial multiplicity (2,2,5)", "[operation][eigen]") {
    // Diagonal diag(2,2,5): eigenvalue 2 has multiplicity 2, 5 is simple.
    // The two lambda=2 columns must be independent eigenvectors of that space.
    mat::dense2D<double> A(3, 3);
    A(0,0) = 2; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 2; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 5;

    auto [eigs, V] = eigen(A);
    REQUIRE(max_eigen_residual(A, eigs, V) < 1e-10);

    // Identify the two columns belonging to eigenvalue 2 and check independence
    // via the magnitude of their normalized cross product being non-degenerate.
    std::vector<std::size_t> two_cols;
    for (std::size_t k = 0; k < 3; ++k)
        if (std::abs(eigs(k).real() - 2.0) < 1e-9 && std::abs(eigs(k).imag()) < 1e-9)
            two_cols.push_back(k);
    REQUIRE(two_cols.size() == 2);

    cplx dot(0.0, 0.0);
    for (std::size_t i = 0; i < 3; ++i)
        dot += std::conj(V(i, two_cols[0])) * V(i, two_cols[1]);
    // Independent (here orthogonalized) -> inner product well below 1.
    REQUIRE(std::abs(dot) < 1e-6);
}

TEST_CASE("General eigen: eigenvalues match eigenvalue()", "[operation][eigen]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = -1;
    A(1,0) = 2; A(1,1) = 3; A(1,2) =  0;
    A(2,0) = 1; A(2,1) = 0; A(2,2) =  2;

    auto vals_only = eigenvalue(A);
    auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == vals_only.size());
    for (std::size_t k = 0; k < eigs.size(); ++k)
        REQUIRE(std::abs(eigs(k) - vals_only(k)) < 1e-12);
}
