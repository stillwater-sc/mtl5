#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stdexcept>
#include <complex>
#include <cmath>
#include <cstdint>
#include <mtl/operation/mult.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randspd.hpp>
#include <mtl/generators/pascal.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/lehmer.hpp>

using namespace mtl;

TEST_CASE("LDL^T factorization: L*D*L^T reproduces A", "[operation][ldlt]") {
    // SPD matrix: A = {{4,2,1},{2,5,3},{1,3,6}}
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // Extract L (unit lower triangular) and D from A
    mat::dense2D<double> L(3, 3);
    mat::dense2D<double> D(3, 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (i == j) {
                L(i, j) = 1.0;  // unit diagonal
                D(i, j) = A(i, j);  // D on diagonal of A
            } else if (j < i) {
                L(i, j) = A(i, j);  // strictly lower triangle
                D(i, j) = 0.0;
            } else {
                L(i, j) = 0.0;
                D(i, j) = 0.0;
            }
        }
    }

    // L * D * L^T should equal Aorig
    auto LDLt = L * D * trans(L);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(LDLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("LDL^T solve", "[operation][ldlt]") {
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

    int info = ldlt_factor(A);
    REQUIRE(info == 0);
    ldlt_solve(A, x, b);

    // Verify Aorig * x = b
    auto r = Aorig * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("LDL^T handles symmetric indefinite matrix", "[operation][ldlt]") {
    // Symmetric indefinite: eigenvalues of different signs
    // A = {{1, 2}, {2, -1}}  eigenvalues: +/- sqrt(5)
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1;  A(0,1) = 2;
    A(1,0) = 2;  A(1,1) = -1;

    mat::dense2D<double> Aorig(2, 2);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // D should have entries of different signs
    // D(0) = 1, D(1) = -1 - 4/1 = -5
    REQUIRE_THAT(A(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(A(1, 1), Catch::Matchers::WithinAbs(-5.0, 1e-12));

    // Solve
    vec::dense_vector<double> b = {5.0, 3.0};
    vec::dense_vector<double> x(2);
    ldlt_solve(A, x, b);

    auto r = Aorig * x;
    for (std::size_t i = 0; i < 2; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("LDL^T detects zero pivot", "[operation][ldlt]") {
    // Singular: A = {{0, 1}, {1, 0}}
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0;  A(0,1) = 1;
    A(1,0) = 1;  A(1,1) = 0;

    int info = ldlt_factor(A);
    REQUIRE(info != 0);  // D(0,0) = 0 -> returns 1
    REQUIRE(info == 1);

    // Verify ldlt_solve throws on zero diagonal
    vec::dense_vector<double> b = {1.0, 2.0};
    vec::dense_vector<double> x(2);
    REQUIRE_THROWS_AS(ldlt_solve(A, x, b), std::domain_error);
}

TEST_CASE("LDL^T on randspd with known eigenvalues", "[operation][ldlt][generator]") {
    constexpr std::size_t n = 5;
    auto A = generators::randspd<double>(n, {8.0, 4.0, 2.0, 1.0, 0.5});

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // Extract L and D, verify L*D*L^T = A
    mat::dense2D<double> L(n, n);
    mat::dense2D<double> D(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                L(i, j) = 1.0;
                D(i, j) = A(i, j);
            } else if (j < i) {
                L(i, j) = A(i, j);
                D(i, j) = 0.0;
            } else {
                L(i, j) = 0.0;
                D(i, j) = 0.0;
            }
        }
    }

    auto LDLt = L * D * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LDLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("LDL^T on Pascal matrix", "[operation][ldlt][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::pascal<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // Pascal LDL^T: L should have integer entries (Pascal's triangle)
    // and D should be all 1s
    for (std::size_t j = 0; j < n; ++j)
        REQUIRE_THAT(A(j, j), Catch::Matchers::WithinAbs(1.0, 1e-10));
}

TEST_CASE("LDL^T on Moler matrix", "[operation][ldlt][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::moler<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    mat::dense2D<double> L(n, n);
    mat::dense2D<double> D(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) { L(i, j) = 1.0; D(i, j) = A(i, j); }
            else if (j < i) { L(i, j) = A(i, j); D(i, j) = 0.0; }
            else { L(i, j) = 0.0; D(i, j) = 0.0; }
        }
    }

    auto LDLt = L * D * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LDLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

TEST_CASE("LDL^T on Lehmer matrix", "[operation][ldlt][generator]") {
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

    int info = ldlt_factor(A);
    REQUIRE(info == 0);

    // All D entries should be positive (Lehmer is SPD)
    for (std::size_t j = 0; j < n; ++j)
        REQUIRE(A(j, j) > 0.0);

    // Verify via solve
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n);
    ldlt_solve(A, x, b);

    auto r = Aorig * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("LDL^T on 1x1 matrix", "[operation][ldlt]") {
    mat::dense2D<double> A(1, 1);
    A(0, 0) = 7.0;

    int info = ldlt_factor(A);
    REQUIRE(info == 0);
    REQUIRE_THAT(A(0, 0), Catch::Matchers::WithinAbs(7.0, 1e-14));

    // Re-create for solve (factor overwrites D on diagonal)
    mat::dense2D<double> Af(1, 1);
    Af(0, 0) = 7.0;
    int info_af = ldlt_factor(Af);
    REQUIRE(info_af == 0);

    vec::dense_vector<double> b = {21.0};
    vec::dense_vector<double> x(1);
    ldlt_solve(Af, x, b);
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("LDL^T on empty (0x0) matrix", "[operation][ldlt]") {
    mat::dense2D<double> A(0, 0);
    int info = ldlt_factor(A);
    REQUIRE(info == 0);
}

// ---------------------------------------------------------------------------
// Regression: #352 -- ldlt returned a wrong solution for a Hermitian complex
// matrix while reporting info == 0.
//
// operation/ldlt.hpp contains no conjugation, so it computes A = L*D*L^T. That
// is the CORRECT factorization for a complex symmetric matrix (A == A^T) and the
// WRONG one for a Hermitian matrix (A == A^H) -- and nothing distinguished the
// two, so Hermitian input ran to completion and returned a plausible but wrong
// answer. Max element error on the reported 2x2 was 9.5e-01.
//
// The existing cases above are all real-valued, where conjugation is the
// identity, which is why this survived.
//
// Resolution: ldlt_factor now refuses Hermitian-but-not-symmetric complex input
// with LDLT_NOT_SYMMETRIC, and ldlt_h_factor/ldlt_h_solve provide the LDL^H
// factorization that input actually has.
// ---------------------------------------------------------------------------

TEST_CASE("LDL^T refuses Hermitian complex input instead of answering wrongly (#352)",
          "[operation][ldlt][regression]") {
    using cd = std::complex<double>;
    mat::dense2D<cd> H(2, 2);
    H(0,0) = cd(2, 0); H(0,1) = cd(1,-1);
    H(1,0) = cd(1, 1); H(1,1) = cd(3, 0);      // Hermitian, not symmetric

    mat::dense2D<cd> LD(H);
    REQUIRE(ldlt_factor(LD) == LDLT_NOT_SYMMETRIC);
}

TEST_CASE("LDL^T still handles complex SYMMETRIC input (#352)",
          "[operation][ldlt][regression]") {
    // The guard must not catch this case: A == A^T is exactly what LDL^T is for,
    // and it was already correct.
    using cd = std::complex<double>;
    const std::size_t n = 2;
    mat::dense2D<cd> S(n, n);
    S(0,0) = cd(2, 1); S(0,1) = cd(1,-1);
    S(1,0) = cd(1,-1); S(1,1) = cd(3, 2);      // symmetric, not Hermitian

    vec::dense_vector<cd> xt(n), b(n), x(n);
    xt[0] = cd(1, 0); xt[1] = cd(0, 1);
    mult(S, xt, b);

    mat::dense2D<cd> LD(S);
    REQUIRE(ldlt_factor(LD) == 0);
    ldlt_solve(LD, x, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(x(i) - xt(i)) < 1e-12);
}

TEST_CASE("LDL^H solves the Hermitian system the reporter's case describes (#352)",
          "[operation][ldlt][regression]") {
    using cd = std::complex<double>;
    const std::size_t n = 2;
    mat::dense2D<cd> H(n, n);
    H(0,0) = cd(2, 0); H(0,1) = cd(1,-1);
    H(1,0) = cd(1, 1); H(1,1) = cd(3, 0);

    vec::dense_vector<cd> xt(n), b(n), x(n);
    xt[0] = cd(1, 0); xt[1] = cd(0, 1);
    mult(H, xt, b);

    mat::dense2D<cd> LD(H);
    REQUIRE(ldlt_h_factor(LD) == 0);
    ldlt_h_solve(LD, x, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(x(i) - xt(i)) < 1e-12);

    // D must come out real for a Hermitian A.
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(LD(i,i).imag()) < 1e-14);
}

TEST_CASE("LDL^H over random Hermitian matrices (#352)", "[operation][ldlt][regression]") {
    using cd = std::complex<double>;
    std::uint64_t seed = 20260803u;
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((seed >> 11) % 2000001) / 1000000.0 - 1.0;
    };

    for (std::size_t n : {3u, 5u, 8u}) {
        mat::dense2D<cd> A(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            A(i,i) = cd(next() + static_cast<double>(n), 0.0);   // real diagonal, diagonally dominant
            for (std::size_t j = i + 1; j < n; ++j) {
                const cd v(next(), next());
                A(i,j) = v;
                A(j,i) = std::conj(v);
            }
        }

        vec::dense_vector<cd> xt(n), b(n), x(n);
        for (std::size_t i = 0; i < n; ++i) xt[i] = cd(next(), next());
        mult(A, xt, b);

        mat::dense2D<cd> LD(A);
        INFO("n = " << n);
        REQUIRE(ldlt_h_factor(LD) == 0);
        ldlt_h_solve(LD, x, b);
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(std::abs(x(i) - xt(i)) < 1e-9);
    }
}

TEST_CASE("LDL^T refuses Hermitian input that is Hermitian only to rounding (#352)",
          "[operation][ldlt][regression]") {
    // A matrix assembled in floating point is Hermitian to rounding, not
    // bit-exactly. An exact structure test misses those: ONE ULP in one entry
    // was enough to slip past it into the LDL^T path and return an answer wrong
    // in the first significant digit under info == 0 -- the very failure #352
    // is about. The guard is scale-relative for that reason.
    using cd = std::complex<double>;
    mat::dense2D<cd> H(2, 2);
    const cd off(1.0, 2.0);
    H(0,0) = cd(4, 0); H(0,1) = std::conj(off);
    H(1,0) = off;      H(1,1) = cd(5, 0);

    mat::dense2D<cd> exact(H);
    REQUIRE(ldlt_factor(exact) == LDLT_NOT_SYMMETRIC);

    // Perturb one entry by a single ULP: still Hermitian for any practical
    // purpose, and must still be refused.
    mat::dense2D<cd> noisy(H);
    noisy(1,0) = cd(std::nextafter(off.real(), 2.0), off.imag());
    REQUIRE(ldlt_factor(noisy) == LDLT_NOT_SYMMETRIC);
}

TEST_CASE("LDL^H refuses input with a non-real diagonal (#352)",
          "[operation][ldlt][regression]") {
    // Mirror of the guard on ldlt_factor. A Hermitian matrix cannot have a
    // non-real diagonal, so complex-symmetric input reaching ldlt_h_factor is a
    // caller mistake; taking Re() of the diagonal would silently discard the
    // imaginary part and return a wrong factorization under info == 0.
    using cd = std::complex<double>;
    mat::dense2D<cd> S(2, 2);
    S(0,0) = cd(4, 1); S(0,1) = cd(1, 2);
    S(1,0) = cd(1, 2); S(1,1) = cd(5,-1);      // symmetric, non-real diagonal

    mat::dense2D<cd> LD(S);
    REQUIRE(ldlt_h_factor(LD) == LDLT_NOT_HERMITIAN);

    // ...and the routine that input actually belongs to still accepts it.
    mat::dense2D<cd> LT(S);
    REQUIRE(ldlt_factor(LT) == 0);
}

TEST_CASE("LDL^H handles the degenerate sizes (#352)", "[operation][ldlt][regression]") {
    SECTION("empty") {
        mat::dense2D<double> A(0, 0);
        vec::dense_vector<double> x(0), b(0);
        REQUIRE(ldlt_h_factor(A) == 0);
        REQUIRE_NOTHROW(ldlt_h_solve(A, x, b));
    }
    SECTION("1x1 real") {
        mat::dense2D<double> A(1, 1);
        A(0,0) = 4.0;
        vec::dense_vector<double> x(1), b(1);
        b[0] = 8.0;
        REQUIRE(ldlt_h_factor(A) == 0);
        ldlt_h_solve(A, x, b);
        REQUIRE(std::abs(x(0) - 2.0) < 1e-14);
    }
    SECTION("1x1 Hermitian complex") {
        using cd = std::complex<double>;
        mat::dense2D<cd> A(1, 1);
        A(0,0) = cd(4, 0);                      // a 1x1 Hermitian matrix is real
        vec::dense_vector<cd> x(1), b(1);
        b[0] = cd(8, 4);
        REQUIRE(ldlt_h_factor(A) == 0);
        REQUIRE(std::abs(A(0,0).imag()) < 1e-14);
        ldlt_h_solve(A, x, b);
        REQUIRE(std::abs(x(0) - cd(2, 1)) < 1e-14);
    }
    SECTION("1x1 with a non-real diagonal is refused") {
        using cd = std::complex<double>;
        mat::dense2D<cd> A(1, 1);
        A(0,0) = cd(4, 1);
        REQUIRE(ldlt_h_factor(A) == LDLT_NOT_HERMITIAN);
    }
}

TEST_CASE("LDL^H equals LDL^T for real symmetric input (#352)",
          "[operation][ldlt][regression]") {
    // Conjugation is the identity for real types, so the two must agree exactly.
    const std::size_t n = 4;
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
    REQUIRE(ldlt_factor(L1) == 0);
    REQUIRE(ldlt_h_factor(L2) == 0);
    ldlt_solve(L1, x1, b);
    ldlt_h_solve(L2, x2, b);

    // Deliberately EXACT, not epsilon-based. The claim under test is not "these
    // two agree numerically" -- that is too weak to be worth asserting -- but
    // "for a real type the LDL^H path IS the LDL^T path", conj<T>::apply being
    // the identity. The two bodies are the same expression tree, so any
    // contraction or scheduling the compiler applies, it applies to both. A
    // difference of even one ULP here means the arithmetic in one path diverged
    // from the other, which is exactly what this test exists to catch. Do not
    // relax it to a tolerance; that would silently retire the invariant.
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x1(i) == x2(i));
}
