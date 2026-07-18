// Exercises the LAPACK geev dispatch for the general (non-symmetric)
// eigenproblem. The dispatch triggers for column-major float/double dense
// matrices when MTL5_HAS_LAPACK is defined; without LAPACK these same cases
// run the in-house double-shift QR + inverse iteration, so the test validates
// both paths identically.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

using namespace mtl;
using cplx = std::complex<double>;
using col_params = mat::parameters<tag::col_major>;
using colmat = mat::dense2D<double, col_params>;

namespace {

template <typename Mat>
double max_eigen_residual(const Mat& A,
                          const vec::dense_vector<cplx>& eigs,
                          const mat::dense2D<cplx>& V) {
    const std::size_t n = A.num_rows();
    double worst = 0.0;
    for (std::size_t k = 0; k < n; ++k)
        for (std::size_t i = 0; i < n; ++i) {
            cplx axi(0.0, 0.0);
            for (std::size_t j = 0; j < n; ++j)
                axi += cplx(A(i, j), 0.0) * V(j, k);
            axi -= eigs(k) * V(i, k);
            worst = std::max(worst, std::abs(axi));
        }
    return worst;
}

bool spectra_match(std::vector<cplx> computed, std::vector<cplx> expected, double tol) {
    if (computed.size() != expected.size()) return false;
    for (const auto& e : expected) {
        std::size_t best = computed.size();
        double bestd = tol;
        for (std::size_t j = 0; j < computed.size(); ++j) {
            double d = std::abs(computed[j] - e);
            if (d <= bestd) { bestd = d; best = j; }
        }
        if (best == computed.size()) return false;
        computed.erase(computed.begin() + best);
    }
    return true;
}

} // namespace

TEST_CASE("geev dispatch: real spectrum eigenvalues", "[interface][lapack][geev]") {
    // Column-major matrix with eigenvalues -1, -2.
    colmat A(2, 2);
    A(0,0) = 0;  A(0,1) = 1;
    A(1,0) = -2; A(1,1) = -3;

    auto eigs = eigenvalue(A);
    REQUIRE(eigs.size() == 2);
    std::vector<cplx> computed = {eigs(0), eigs(1)};
    REQUIRE(spectra_match(computed, {cplx(-1,0), cplx(-2,0)}, 1e-9));
}

TEST_CASE("geev dispatch: complex-conjugate spectrum", "[interface][lapack][geev]") {
    // {{0,-1},{1,0}} -> eigenvalues +/- i.
    colmat A(2, 2);
    A(0,0) = 0; A(0,1) = -1;
    A(1,0) = 1; A(1,1) =  0;

    auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == 2);
    std::vector<cplx> computed = {eigs(0), eigs(1)};
    REQUIRE(spectra_match(computed, {cplx(0,1), cplx(0,-1)}, 1e-9));
    // Complex eigenvectors satisfy the eigen relation and are unit norm.
    REQUIRE(max_eigen_residual(A, eigs, V) < 1e-10);
    for (std::size_t k = 0; k < 2; ++k) {
        double s = 0.0;
        for (std::size_t i = 0; i < 2; ++i) s += std::norm(V(i, k));
        REQUIRE_THAT(std::sqrt(s), Catch::Matchers::WithinAbs(1.0, 1e-10));
    }
}

TEST_CASE("geev dispatch: mixed real + complex 4x4 eigenpairs", "[interface][lapack][geev]") {
    // Block-triangular: eigenvalues 5, 2, and 1 +/- 2i.
    colmat A(4, 4);
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

    std::vector<cplx> computed(4);
    for (std::size_t k = 0; k < 4; ++k) computed[k] = eigs(k);
    REQUIRE(spectra_match(computed, {cplx(5,0), cplx(2,0), cplx(1,2), cplx(1,-2)}, 1e-9));
}

TEST_CASE("geev dispatch: agrees with the in-house path (row-major)", "[interface][lapack][geev]") {
    // Same matrix in both orientations: column-major goes through geev (when
    // LAPACK is on), row-major always through the in-house QR. Spectra match.
    const double B[3][3] = {{4, 1, -1},
                            {2, 3,  0},
                            {1, 0,  2}};
    colmat Ac(3, 3);
    mat::dense2D<double> Ar(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) { Ac(i, j) = B[i][j]; Ar(i, j) = B[i][j]; }

    auto ec = eigenvalue(Ac);
    auto er = eigenvalue(Ar);
    std::vector<cplx> vc(3), vr(3);
    for (std::size_t k = 0; k < 3; ++k) { vc[k] = ec(k); vr[k] = er(k); }
    // Cross-check both spectra against each other.
    std::vector<cplx> vr_copy = vr;
    REQUIRE(spectra_match(vc, vr_copy, 1e-8));
}
