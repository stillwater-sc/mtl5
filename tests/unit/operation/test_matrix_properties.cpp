// Tests for the matrix structural property predicates (#244, batch 1):
// is_square, is_empty, is_symmetric, is_hermitian, is_upper/lower/is_triangular,
// is_diagonal, is_banded, is_diagonally_dominant.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <limits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/matrix_properties.hpp>
#include <mtl/generators/randorth.hpp>

using namespace mtl;
using cd = std::complex<double>;

namespace {
// Fill a dense2D from a nested initializer list (row by row).
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

TEST_CASE("is_square / is_empty", "[operation][properties][matrix]") {
    REQUIRE(is_square(make({{1, 2}, {3, 4}})));
    REQUIRE_FALSE(is_square(mat::dense2D<double>(2, 3)));
    REQUIRE(is_empty(mat::dense2D<double>(0, 0)));
    REQUIRE(is_empty(mat::dense2D<double>(0, 5)));
    REQUIRE(is_empty(mat::dense2D<double>(5, 0)));
    REQUIRE_FALSE(is_empty(make({{1}})));
    // A 0x0 matrix is (vacuously) square, diagonal, triangular, symmetric.
    mat::dense2D<double> z(0, 0);
    REQUIRE(is_square(z));
    REQUIRE(is_symmetric(z));
    REQUIRE(is_diagonal(z));
    REQUIRE(is_triangular(z));
    // 1x1 is everything.
    auto one = make({{7}});
    REQUIRE(is_symmetric(one));
    REQUIRE(is_diagonal(one));
    REQUIRE(is_triangular(one));
}

TEST_CASE("is_symmetric: exact, tolerance, non-square", "[operation][properties][matrix]") {
    REQUIRE(is_symmetric(make({{1, 2, 3}, {2, 4, 5}, {3, 5, 6}})));
    REQUIRE_FALSE(is_symmetric(make({{1, 2}, {3, 4}})));
    // Non-square is never symmetric.
    REQUIRE_FALSE(is_symmetric(mat::dense2D<double>(2, 3)));
    // Near-symmetric: reads asymmetric at tol=0, symmetric within tol.
    auto A = make({{1.0, 2.0}, {2.0 + 1e-9, 1.0}});
    REQUIRE_FALSE(is_symmetric(A));
    REQUIRE(is_symmetric(A, 1e-6));
}

TEST_CASE("is_hermitian", "[operation][properties][matrix]") {
    // Real symmetric is hermitian.
    mat::dense2D<cd> H(2, 2);
    H(0, 0) = cd(1, 0);  H(0, 1) = cd(2, -3);
    H(1, 0) = cd(2, 3);  H(1, 1) = cd(4, 0);
    REQUIRE(is_hermitian(H));
    REQUIRE_FALSE(is_symmetric(H));   // A(0,1) != A(1,0)
    // Non-real diagonal breaks hermitian.
    H(0, 0) = cd(1, 1);
    REQUIRE_FALSE(is_hermitian(H));
}

TEST_CASE("triangular predicates", "[operation][properties][matrix]") {
    auto U = make({{1, 2, 3}, {0, 4, 5}, {0, 0, 6}});
    auto L = make({{1, 0, 0}, {2, 3, 0}, {4, 5, 6}});
    auto D = make({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
    auto F = make({{1, 2}, {3, 4}});

    REQUIRE(is_upper_triangular(U));
    REQUIRE_FALSE(is_lower_triangular(U));
    REQUIRE(is_lower_triangular(L));
    REQUIRE_FALSE(is_upper_triangular(L));
    REQUIRE(is_triangular(U));
    REQUIRE(is_triangular(L));
    REQUIRE_FALSE(is_triangular(F));

    // Diagonal is both upper and lower.
    REQUIRE(is_upper_triangular(D));
    REQUIRE(is_lower_triangular(D));
    REQUIRE(is_diagonal(D));
    REQUIRE_FALSE(is_diagonal(U));
}

TEST_CASE("is_banded", "[operation][properties][matrix]") {
    // Tridiagonal: kl=1, ku=1.
    auto T = make({{2, -1, 0, 0},
                   {-1, 2, -1, 0},
                   {0, -1, 2, -1},
                   {0, 0, -1, 2}});
    REQUIRE(is_banded(T, 1, 1));
    REQUIRE(is_banded(T, 2, 2));      // wider band still holds
    REQUIRE_FALSE(is_banded(T, 0, 0)); // not diagonal
    REQUIRE(is_banded(T, 1, 0) == false); // has a superdiagonal
    // Diagonal is banded with kl=ku=0.
    auto D = make({{1, 0}, {0, 2}});
    REQUIRE(is_banded(D, 0, 0));
    // Lower bidiagonal: kl=1, ku=0.
    auto Lb = make({{1, 0, 0}, {2, 3, 0}, {0, 4, 5}});
    REQUIRE(is_banded(Lb, 1, 0));
    REQUIRE_FALSE(is_banded(Lb, 0, 0));
}

TEST_CASE("NaN entries fail structural predicates", "[operation][properties][matrix]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // A NaN is unordered, so `abs > tol` would silently accept it; the
    // predicates use !(dev <= tol) so a NaN must break every structural check.
    // NaN in an off-diagonal position makes the deviation abs(A(i,j)-A(j,i))
    // itself NaN, which must fail the symmetry test at any tolerance.
    auto S = make({{1.0, nan}, {2.0, 3.0}});
    REQUIRE_FALSE(is_symmetric(S));
    REQUIRE_FALSE(is_symmetric(S, 1e6));
    auto Dg = make({{nan, 0.0}, {0.0, 1.0}});  // NaN on diagonal (off-diag zero)
    REQUIRE(is_diagonal(Dg));                  // off-diagonals are zero
    auto Off = make({{1.0, nan}, {0.0, 1.0}}); // NaN off-diagonal
    REQUIRE_FALSE(is_diagonal(Off));
    REQUIRE_FALSE(is_upper_triangular(make({{1.0, 2.0}, {nan, 1.0}})));
}

TEST_CASE("is_diagonally_dominant", "[operation][properties][matrix]") {
    // Strictly dominant.
    auto A = make({{4, 1, 0}, {1, 5, -1}, {0, 1, 3}});
    REQUIRE(is_diagonally_dominant(A));
    REQUIRE(is_diagonally_dominant(A, /*strict=*/true));
    // Weakly dominant (row 0 equality): dominant but not strict.
    auto W = make({{2, 1, 1}, {0, 5, 1}, {0, 0, 4}});
    REQUIRE(is_diagonally_dominant(W));
    REQUIRE_FALSE(is_diagonally_dominant(W, /*strict=*/true));
    // Not dominant.
    auto N = make({{1, 2}, {2, 1}});
    REQUIRE_FALSE(is_diagonally_dominant(N));
    // Non-square is never dominant.
    REQUIRE_FALSE(is_diagonally_dominant(mat::dense2D<double>(2, 3)));
}

TEST_CASE("is_orthogonal / is_unitary", "[operation][properties][orthogonal]") {
    // Identity and permutation are exactly orthogonal.
    auto I = make({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    REQUIRE(is_orthogonal(I));
    REQUIRE(is_orthogonal(make({{0, 1}, {1, 0}})));

    // Rotation matrix (orthogonal within rounding).
    const double th = 0.7;
    auto R = make({{std::cos(th), -std::sin(th)}, {std::sin(th), std::cos(th)}});
    REQUIRE(is_orthogonal(R));

    // Non-orthogonal (columns not unit / not perpendicular).
    REQUIRE_FALSE(is_orthogonal(make({{1, 1}, {0, 1}})));
    // Scaled identity: orthogonal columns but not unit norm.
    REQUIRE_FALSE(is_orthogonal(make({{2, 0}, {0, 2}})));
    // Non-square is never orthogonal.
    REQUIRE_FALSE(is_orthogonal(mat::dense2D<double>(3, 2)));

    // A random orthogonal matrix (Q from QR) passes at the default tolerance.
    REQUIRE(is_orthogonal(generators::randorth<double>(6)));

    // Complex unitary: diag(i, 1) has A^H A = I.
    mat::dense2D<cd> U(2, 2);
    U(0, 0) = cd(0, 1); U(0, 1) = cd(0, 0);
    U(1, 0) = cd(0, 0); U(1, 1) = cd(1, 0);
    REQUIRE(is_unitary(U));
}

TEST_CASE("is_normal", "[operation][properties][normal]") {
    // Symmetric, skew-symmetric, orthogonal, and diagonal matrices are normal.
    REQUIRE(is_normal(make({{1, 2, 3}, {2, 4, 5}, {3, 5, 6}})));   // symmetric
    REQUIRE(is_normal(make({{0, 1}, {-1, 0}})));                   // skew
    REQUIRE(is_normal(make({{0, 1}, {1, 0}})));                    // orthogonal
    REQUIRE(is_normal(make({{5, 0}, {0, -2}})));                   // diagonal

    // A generic non-symmetric matrix is not normal (A A^T != A^T A).
    REQUIRE_FALSE(is_normal(make({{1, 1}, {0, 1}})));
    // Non-square is never normal.
    REQUIRE_FALSE(is_normal(mat::dense2D<double>(2, 3)));
}
