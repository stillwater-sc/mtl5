#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <complex>
#include <cstdint>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randspd.hpp>
#include <mtl/generators/pascal.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/lehmer.hpp>

using namespace mtl;

TEST_CASE("Cholesky factorization: L*L^T reproduces A", "[operation][cholesky]") {
    // SPD matrix: A = {{4,2,1},{2,5,3},{1,3,6}}
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    // Extract L from lower triangle of A
    mat::dense2D<double> L(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    // L * L^T should equal Aorig
    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("Cholesky solve", "[operation][cholesky]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);
    cholesky_solve(A, x, b);

    // Verify Aorig * x = b
    auto r = Aorig * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("Cholesky detects non-SPD matrix", "[operation][cholesky]") {
    // Not positive definite
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1;  A(0,1) = 3;
    A(1,0) = 3;  A(1,1) = 1;

    int info = cholesky_factor(A);
    REQUIRE(info != 0);
}

// -- Generator-based Cholesky tests ------------------------------------

TEST_CASE("Cholesky on randspd with known eigenvalues", "[operation][cholesky][generator]") {
    constexpr std::size_t n = 5;
    auto A = generators::randspd<double>(n, {8.0, 4.0, 2.0, 1.0, 0.5});

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    // Extract L and verify L*L^T = A
    mat::dense2D<double> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("Cholesky on Pascal matrix", "[operation][cholesky][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::pascal<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    mat::dense2D<double> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("Cholesky on Moler matrix", "[operation][cholesky][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::moler<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    mat::dense2D<double> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

TEST_CASE("Cholesky on Lehmer matrix", "[operation][cholesky][generator]") {
    constexpr std::size_t n = 6;
    generators::lehmer<double> L_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = L_gen(i, j);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    mat::dense2D<double> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

// ---------------------------------------------------------------------------
// #353: cholesky did not compile for complex element types.
//
// The first failure was an opaque "no match for operator<=" on the definiteness
// test, but the gap was larger than the comparison: the inner product was
// unconjugated too, so repairing only the comparison would have produced
// something that compiles and computes the WRONG factorization -- which is
// precisely what happened to ldlt in #352.
//
// Resolution: cholesky_factor is now restricted to real element types with a
// diagnostic that names the alternative, and cholesky_h_factor /
// cholesky_h_solve provide A = L*L^H for Hermitian positive definite input.
// The rejection of complex is pinned by
// tests/unit/compile_fail/cholesky_factor_complex.cpp.
// ---------------------------------------------------------------------------

TEST_CASE("Cholesky LL^H factors a Hermitian positive definite matrix (#353)",
          "[operation][cholesky][regression]") {
    using cd = std::complex<double>;
    const std::size_t n = 2;
    mat::dense2D<cd> A(n, n);
    A(0,0) = cd(4, 0); A(0,1) = cd(1,-1);
    A(1,0) = cd(1, 1); A(1,1) = cd(3, 0);      // HPD, eigenvalues 5 and 2

    mat::dense2D<cd> L(A);
    REQUIRE(cholesky_h_factor(L) == 0);

    // The factor's diagonal is real for a Hermitian A.
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(L(i,i).imag()) < 1e-14);

    // L*L^H reconstructs A over the lower triangle (the only part read).
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            cd s(0, 0);
            for (std::size_t k = 0; k <= j; ++k) s += L(i,k) * std::conj(L(j,k));
            INFO("i=" << i << " j=" << j);
            REQUIRE(std::abs(s - A(i,j)) < 1e-12);
        }

    // And it solves.
    vec::dense_vector<cd> xt(n), b(n), x(n);
    xt[0] = cd(1, 0); xt[1] = cd(0, 1);
    mult(A, xt, b);
    cholesky_h_solve(L, x, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(x(i) - xt(i)) < 1e-12);
}

TEST_CASE("Cholesky LL^H over random Hermitian positive definite matrices (#353)",
          "[operation][cholesky][regression]") {
    using cd = std::complex<double>;
    std::uint64_t seed = 20260803u;
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((seed >> 11) % 2000001) / 1000000.0 - 1.0;
    };

    for (std::size_t n : {3u, 5u, 8u}) {
        // A = B^H*B + n*I is Hermitian positive definite by construction.
        mat::dense2D<cd> B(n, n), A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) B(i,j) = cd(next(), next());
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                cd s(0, 0);
                for (std::size_t k = 0; k < n; ++k) s += std::conj(B(k,i)) * B(k,j);
                A(i,j) = s + (i == j ? cd(static_cast<double>(n), 0) : cd(0, 0));
            }

        vec::dense_vector<cd> xt(n), b(n), x(n);
        for (std::size_t i = 0; i < n; ++i) xt[i] = cd(next(), next());
        mult(A, xt, b);

        mat::dense2D<cd> L(A);
        INFO("n = " << n);
        REQUIRE(cholesky_h_factor(L) == 0);
        cholesky_h_solve(L, x, b);
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(std::abs(x(i) - xt(i)) < 1e-9);
    }
}

TEST_CASE("Cholesky LL^H refuses input with a non-real diagonal (#353)",
          "[operation][cholesky][regression]") {
    // A Hermitian matrix cannot have a non-real diagonal, so complex-symmetric
    // input is a caller mistake. Taking the real part would silently factor a
    // DIFFERENT matrix and return 0 -- the mirror of the ldlt bug in #352.
    using cd = std::complex<double>;
    mat::dense2D<cd> S(2, 2);
    S(0,0) = cd(4, 1); S(0,1) = cd(1, 2);
    S(1,0) = cd(1, 2); S(1,1) = cd(5,-1);
    REQUIRE(cholesky_h_factor(S) == CHOLESKY_NOT_HERMITIAN);
}

TEST_CASE("Cholesky LL^H detects a non-positive-definite Hermitian matrix (#353)",
          "[operation][cholesky][regression]") {
    using cd = std::complex<double>;
    mat::dense2D<cd> N(2, 2);
    N(0,0) = cd(1, 0); N(0,1) = cd(2, 0);
    N(1,0) = cd(2, 0); N(1,1) = cd(1, 0);      // Hermitian, indefinite
    REQUIRE(cholesky_h_factor(N) == 2);        // fails at pivot k=1 -> k+1
}

TEST_CASE("Cholesky LL^H handles the degenerate sizes (#353)",
          "[operation][cholesky][regression]") {
    SECTION("empty") {
        mat::dense2D<double> A(0, 0);
        vec::dense_vector<double> x(0), b(0);
        REQUIRE(cholesky_h_factor(A) == 0);
        REQUIRE_NOTHROW(cholesky_h_solve(A, x, b));
    }
    SECTION("1x1 real") {
        mat::dense2D<double> A(1, 1);
        A(0,0) = 4.0;
        vec::dense_vector<double> x(1), b(1);
        b[0] = 8.0;
        REQUIRE(cholesky_h_factor(A) == 0);
        REQUIRE_THAT(A(0,0), Catch::Matchers::WithinAbs(2.0, 1e-14));
        cholesky_h_solve(A, x, b);
        REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(2.0, 1e-14));
    }
    SECTION("1x1 Hermitian complex") {
        using cd = std::complex<double>;
        mat::dense2D<cd> A(1, 1);
        A(0,0) = cd(4, 0);                     // a 1x1 Hermitian matrix is real
        vec::dense_vector<cd> x(1), b(1);
        b[0] = cd(8, 4);
        REQUIRE(cholesky_h_factor(A) == 0);
        cholesky_h_solve(A, x, b);
        REQUIRE(std::abs(x(0) - cd(2, 1)) < 1e-14);
    }
    SECTION("1x1 with a non-real diagonal is refused") {
        using cd = std::complex<double>;
        mat::dense2D<cd> A(1, 1);
        A(0,0) = cd(4, 1);
        REQUIRE(cholesky_h_factor(A) == CHOLESKY_NOT_HERMITIAN);
    }
    SECTION("1x1 non-positive is refused") {
        mat::dense2D<double> A(1, 1);
        A(0,0) = -1.0;
        REQUIRE(cholesky_h_factor(A) == 1);
    }
}

TEST_CASE("Cholesky LL^H equals LL^T for real symmetric input (#353)",
          "[operation][cholesky][regression]") {
    const std::size_t n = 6;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A(i,i) = 4.0 + static_cast<double>(i);
        for (std::size_t j = i + 1; j < n; ++j) {
            const double v = 0.5 + 0.25 * static_cast<double>(i + j);
            A(i,j) = v; A(j,i) = v;
        }
    }
    vec::dense_vector<double> b(n), x1(n), x2(n);
    for (std::size_t i = 0; i < n; ++i) b[i] = 1.0 + static_cast<double>(i);

    mat::dense2D<double> L1(A), L2(A);
    REQUIRE(cholesky_factor(L1) == 0);
    REQUIRE(cholesky_h_factor(L2) == 0);
    cholesky_solve(L1, x1, b);
    cholesky_h_solve(L2, x2, b);

    // Deliberately EXACT, not epsilon-based -- the same invariant test_ldlt.cpp
    // asserts for ldlt_h. The claim is not "these agree numerically" but "for a
    // real type the LL^H path IS the LL^T path", conj<T>::apply being the
    // identity. A one-ULP difference means the arithmetic in one path diverged
    // from the other, which is exactly what this catches. Do not relax it to a
    // tolerance; that would silently retire the invariant.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            REQUIRE(L1(i,j) == L2(i,j));
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x1(i) == x2(i));
}
