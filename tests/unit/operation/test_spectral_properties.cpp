// Tests for the spectral / condition / rank property queries (#244, batch 3):
// spectral_radius, condition_number, rcond, numerical_rank, nullity.
//
// Checks are written to hold for both the LAPACK and the in-house SVD/eigen
// paths: the in-house SVD yields a tiny-but-nonzero smallest singular value for
// a singular matrix, so condition_number of a rank-deficient matrix is tested as
// "very large or infinite" rather than exactly infinite, and rank-deficient rank
// counts pass an explicit tolerance.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/spectral_properties.hpp>
#include <mtl/generators/randspd.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
mat::dense2D<double> make(std::initializer_list<std::initializer_list<double>> rows) {
    const std::size_t m = rows.size();
    const std::size_t n = rows.begin()->size();
    mat::dense2D<double> A(m, n);
    std::size_t i = 0;
    for (const auto& r : rows) {
        std::size_t j = 0;
        for (double v : r) A(i, j++) = v;
        ++i;
    }
    return A;
}
mat::dense2D<double> identity(std::size_t n) {
    mat::dense2D<double> I(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) I(i, j) = (i == j) ? 1.0 : 0.0;
    return I;
}
} // namespace

TEST_CASE("spectral_radius", "[operation][properties][spectral]") {
    // Diagonal: radius is max |diag|.
    REQUIRE_THAT(spectral_radius(make({{2, 0}, {0, -3}})), WithinAbs(3.0, 1e-8));
    // 2D rotation: eigenvalues +/- i, radius 1.
    REQUIRE_THAT(spectral_radius(make({{0, -1}, {1, 0}})), WithinAbs(1.0, 1e-8));
    // SPD: radius is the largest eigenvalue.
    auto R = generators::randspd<double>(4, std::vector<double>{0.5, 2.0, 3.0, 7.0});
    REQUIRE_THAT(spectral_radius(R), WithinRel(7.0, 1e-6));
    // Empty matrix.
    REQUIRE_THAT(spectral_radius(mat::dense2D<double>(0, 0)), WithinAbs(0.0, 1e-15));
}

TEST_CASE("condition_number / rcond", "[operation][properties][spectral]") {
    // Diagonal {10, 1}: sigma = {10, 1}, cond 10, rcond 0.1.
    auto D = make({{10, 0}, {0, 1}});
    REQUIRE_THAT(condition_number(D), WithinRel(10.0, 1e-8));
    REQUIRE_THAT(rcond(D), WithinRel(0.1, 1e-8));

    // Identity is perfectly conditioned.
    REQUIRE_THAT(condition_number(identity(3)), WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(rcond(identity(3)), WithinAbs(1.0, 1e-8));

    // SPD with prescribed eigenvalues {1,2,4}: cond = 4, rcond = 0.25.
    auto R = generators::randspd<double>(3, std::vector<double>{1.0, 2.0, 4.0});
    REQUIRE_THAT(condition_number(R), WithinRel(4.0, 1e-4));
    REQUIRE_THAT(rcond(R), WithinRel(0.25, 1e-4));

    // Rank-deficient (rank 1): huge condition number, near-zero rcond.
    auto S = make({{1, 2}, {2, 4}});
    REQUIRE(condition_number(S) > 1e6);
    REQUIRE(rcond(S) < 1e-6);

    // Empty matrix returns 1 (trivial isometry).
    REQUIRE_THAT(condition_number(mat::dense2D<double>(0, 0)), WithinAbs(1.0, 1e-15));
}

TEST_CASE("numerical_rank / nullity", "[operation][properties][spectral]") {
    // Full rank.
    REQUIRE(numerical_rank(identity(3)) == 3);
    REQUIRE(nullity(identity(3)) == 0);

    // Rank 2 of 3 (one exactly-zero singular value): explicit tol for robustness.
    auto A = make({{1, 0, 0}, {0, 1, 0}, {0, 0, 0}});
    REQUIRE(numerical_rank(A, 1e-8) == 2);
    REQUIRE(nullity(A, 1e-8) == 1);

    // Rank-1 2x2.
    auto S = make({{1, 2}, {2, 4}});
    REQUIRE(numerical_rank(S, 1e-8) == 1);
    REQUIRE(nullity(S, 1e-8) == 1);

    // Explicit tol below the smallest singular value keeps full rank; a large tol
    // drops the small ones. Diagonal {4, 1e-3}: rank 2 at tight tol, 1 at loose.
    auto D = make({{4.0, 0.0}, {0.0, 1e-3}});
    REQUIRE(numerical_rank(D, 1e-6) == 2);
    REQUIRE(numerical_rank(D, 1e-2) == 1);

    // Rectangular full column rank: rank = min(m,n) = 2, nullity (cols) = 0.
    auto T = make({{1, 0}, {0, 1}, {0, 0}});
    REQUIRE(numerical_rank(T, 1e-8) == 2);
    REQUIRE(nullity(T, 1e-8) == 0);
}
