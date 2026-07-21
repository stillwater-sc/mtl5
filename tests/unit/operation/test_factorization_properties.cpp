// Tests for the factorization-backed matrix property queries (#244, batch 2):
// is_spd / is_positive_definite, is_singular / is_nonsingular / is_invertible,
// determinant.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/factorization_properties.hpp>
#include <mtl/generators/pascal.hpp>
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
} // namespace

TEST_CASE("is_spd: SPD, indefinite, non-symmetric, non-square", "[operation][properties][spd]") {
    // Tridiagonal SPD (eigenvalues 2 - 2cos(...) > 0), exactly symmetric.
    auto A = make({{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}});
    REQUIRE(is_spd(A));
    REQUIRE(is_positive_definite(A));

    // Symmetric but indefinite (eigenvalues 3, -1).
    auto Ind = make({{1, 2}, {2, 1}});
    REQUIRE_FALSE(is_spd(Ind));

    // Symmetric negative definite -> not SPD.
    auto Neg = make({{-2, 0}, {0, -3}});
    REQUIRE_FALSE(is_spd(Neg));

    // Non-symmetric (even though it would factor) -> not SPD.
    auto NS = make({{2, 1}, {0, 2}});
    REQUIRE_FALSE(is_spd(NS));

    // Non-square -> not SPD.
    REQUIRE_FALSE(is_spd(mat::dense2D<double>(2, 3)));
}

TEST_CASE("is_spd: exact Pascal and tolerance on computed SPD", "[operation][properties][spd]") {
    // Symmetric Pascal matrix: integer, exactly symmetric, SPD.
    auto P = generators::pascal<double>(5);
    REQUIRE(is_spd(P));

    // randspd = Q * diag(eig) * Q^T is SPD but only symmetric to rounding, so it
    // reads asymmetric at the exact default and SPD once sym_tol admits the noise.
    auto R = generators::randspd<double>(6, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    REQUIRE(is_spd(R, 1e-10));
}

TEST_CASE("is_singular / is_nonsingular / is_invertible", "[operation][properties][singular]") {
    // Exactly singular (row1 = 0.5*row0): elimination yields an exact zero pivot.
    auto S = make({{2, 4}, {1, 2}});
    REQUIRE(is_singular(S));
    REQUIRE_FALSE(is_nonsingular(S));
    REQUIRE_FALSE(is_invertible(S));

    // Nonsingular.
    auto N = make({{2, 1}, {1, 2}});
    REQUIRE_FALSE(is_singular(N));
    REQUIRE(is_nonsingular(N));
    REQUIRE(is_invertible(N));

    // Non-square is treated as singular (no inverse).
    REQUIRE(is_singular(mat::dense2D<double>(2, 3)));

    // Empty matrix is (vacuously) nonsingular.
    REQUIRE_FALSE(is_singular(mat::dense2D<double>(0, 0)));
}

TEST_CASE("is_singular: near-singular tolerance", "[operation][properties][singular]") {
    // Tiny-but-nonzero pivot: nonsingular at tol 0, singular once tol admits it.
    auto A = make({{1.0, 0.0}, {0.0, 1e-10}});
    REQUIRE_FALSE(is_singular(A));          // exact: 1e-10 > 0
    REQUIRE(is_singular(A, 1e-8));          // within tol -> singular
}

TEST_CASE("determinant: values, sign, singular, empty", "[operation][properties][determinant]") {
    REQUIRE_THAT(determinant(make({{1, 2}, {3, 4}})), WithinAbs(-2.0, 1e-12));
    REQUIRE_THAT(determinant(make({{2, 0}, {0, 3}})), WithinAbs(6.0, 1e-12));

    // Row swap flips the sign: det of the 2x2 exchange matrix is -1.
    REQUIRE_THAT(determinant(make({{0, 1}, {1, 0}})), WithinAbs(-1.0, 1e-12));

    // 3x3 with a known determinant (-306).
    auto A = make({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}});
    REQUIRE_THAT(determinant(A), WithinAbs(-306.0, 1e-9));

    // Triangular: determinant is the product of the diagonal.
    auto U = make({{2, 5, 9}, {0, 3, 7}, {0, 0, 4}});
    REQUIRE_THAT(determinant(U), WithinAbs(24.0, 1e-12));

    // Singular -> exactly zero.
    REQUIRE(determinant(make({{2, 4}, {1, 2}})) == 0.0);

    // Empty (0x0) determinant is 1 by convention.
    REQUIRE_THAT(determinant(mat::dense2D<double>(0, 0)), WithinAbs(1.0, 1e-15));

    // Pascal matrix has determinant exactly 1.
    REQUIRE_THAT(determinant(generators::pascal<double>(6)), WithinAbs(1.0, 1e-6));
}

TEST_CASE("determinant matches the eigenvalue product of randspd", "[operation][properties][determinant]") {
    std::vector<double> eig{1.0, 2.0, 3.0, 4.0};
    auto R = generators::randspd<double>(4, eig);
    double prod = 1.0;
    for (double e : eig) prod *= e;   // 24
    REQUIRE_THAT(determinant(R), WithinRel(prod, 1e-9));
}
