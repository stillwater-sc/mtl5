#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <complex>
#include <cstdint>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/kahan.hpp>
#include <mtl/generators/randorth.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/vandermonde.hpp>
#include <mtl/generators/randsvd.hpp>

#include <cmath>
#include <vector>

using namespace mtl;

TEST_CASE("QR factorization: Q*R reproduces A", "[operation][qr]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 12; A(0,1) = -51; A(0,2) =   4;
    A(1,0) =  6; A(1,1) = 167; A(1,2) = -68;
    A(2,0) = -4; A(2,1) =  24; A(2,2) = -41;

    // Save original
    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(3);
    int info = qr_factor(A, tau);
    REQUIRE(info == 0);

    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Verify Q*R = A
    auto QR = Q * R;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(QR(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

TEST_CASE("QR factorization: Q is orthogonal", "[operation][qr]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) =  3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) =  6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    vec::dense_vector<double> tau(3);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);

    // Q^T * Q should be I
    auto QtQ = trans(Q) * Q;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(QtQ(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("QR solve: least-squares for square system", "[operation][qr]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) =  3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) =  6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    // Save original
    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);

    vec::dense_vector<double> tau(3);
    qr_factor(A, tau);
    qr_solve(A, tau, x, b);

    // Verify Aorig * x = b
    auto r = Aorig * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("LQ factorization: L*Q reproduces A", "[operation][lq]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 12; A(0,1) = -51; A(0,2) =   4;
    A(1,0) =  6; A(1,1) = 167; A(1,2) = -68;
    A(2,0) = -4; A(2,1) =  24; A(2,2) = -41;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(3);
    int info = lq_factor(A, tau);
    REQUIRE(info == 0);

    auto L = lq_extract_L(A);
    auto Q = lq_extract_Q(A, tau);

    auto LQ = L * Q;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(LQ(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

// -- Generator-based QR tests -----------------------------------------

TEST_CASE("QR on Kahan matrix", "[operation][qr][generator]") {
    // Kahan is upper triangular + ill-conditioned -- classic QR stress test
    constexpr std::size_t n = 6;
    auto A = generators::kahan<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(n);
    int info = qr_factor(A, tau);
    REQUIRE(info == 0);

    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Verify Q*R = A (reconstruction)
    auto QR = Q * R;
    double residual = frobenius_norm(QR - Aorig);
    REQUIRE(residual / frobenius_norm(Aorig) < 1e-8);
}

TEST_CASE("QR orthogonality with randorth", "[operation][qr][generator]") {
    // QR of an already-orthogonal matrix: Q should be orthogonal
    constexpr std::size_t n = 8;
    auto Qorig = generators::randorth<double>(n);

    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = Qorig(i, j);

    vec::dense_vector<double> tau(n);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);

    // Q^T * Q should be I
    auto QtQ = trans(Q) * Q;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(QtQ(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("QR on Frank (Hessenberg) matrix", "[operation][qr][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::frank<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(n);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Q*R = A
    auto QR = Q * R;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(QR(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));

    // R should be upper triangular
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            REQUIRE_THAT(R(i, j), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("QR reconstruction on Vandermonde", "[operation][qr][generator]") {
    // Vandermonde is ill-conditioned -- verify Q*R = A reconstruction
    auto A = generators::vandermonde<double>({1.0, 2.0, 3.0, 4.0, 5.0});
    std::size_t n = 5;

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(n);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    auto QR = Q * R;
    double rel_error = frobenius_norm(QR - Aorig) / frobenius_norm(Aorig);
    REQUIRE(rel_error < 1e-8);
}

TEST_CASE("QR on randsvd with known condition number", "[operation][qr][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::randsvd<double>(n, 100.0, 3);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> tau(n);
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Reconstruction accuracy
    auto QR = Q * R;
    double rel_error = frobenius_norm(QR - Aorig) / frobenius_norm(Aorig);
    REQUIRE(rel_error < 1e-8);

    // Q must be orthogonal
    auto QtQ = trans(Q) * Q;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(QtQ(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

// ---------------------------------------------------------------------------
// #353: qr did not compile for complex element types.
//
// householder() failed on `axi > scale` and `x0 <= 0` -- relational operators
// applied to a complex where a MAGNITUDE was meant. But the gap was larger than
// the comparisons: sigma accumulated z^2 instead of |z|^2, and the reflector was
// written I - beta*v*v^T rather than I - tau*v*v^H, so repairing only the types
// would have produced something that compiles and computes a wrong Q.
//
// The reflector is now H = I - tau*v*v^H with H*x = beta*e_1, beta real. H is
// unitary but NOT Hermitian for complex tau, so H^-1 = H^H -- which is why
// qr_extract_Q applies conj(tau) and lq_factor applies H^H while lq_extract_Q
// applies H. Those adjoints are the whole content of the fix; every one of them
// is invisible for real input, where conj is the identity.
// ---------------------------------------------------------------------------

namespace {

std::uint64_t cx_seed = 4242u;
double cx_next() {
    cx_seed = cx_seed * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<double>((cx_seed >> 11) % 2000001) / 1000000.0 - 1.0;
}

using cd = std::complex<double>;

/// max |Q^H Q - I| over the whole matrix
double unitarity_error(const mat::dense2D<cd>& Q) {
    const std::size_t n = Q.num_rows();
    double worst = 0.0;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) {
            cd s(0, 0);
            for (std::size_t c = 0; c < n; ++c) s += std::conj(Q(c,a)) * Q(c,b);
            worst = std::max(worst, std::abs(s - (a == b ? cd(1,0) : cd(0,0))));
        }
    return worst;
}

}  // namespace

TEST_CASE("Complex QR: Q is unitary and Q*R reproduces A (#353)",
          "[operation][qr][regression]") {
    for (auto shape : {std::pair<std::size_t,std::size_t>{4,4},
                       {6,3},      // tall  -- least squares shape
                       {3,6},      // wide
                       {5,5}}) {
        const std::size_t m = shape.first, n = shape.second;
        mat::dense2D<cd> A(m, n), A0(m, n);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) { A(i,j) = cd(cx_next(), cx_next()); A0(i,j) = A(i,j); }

        vec::dense_vector<cd> tau;
        INFO("m = " << m << ", n = " << n);
        REQUIRE(qr_factor(A, tau) == 0);
        auto Q = qr_extract_Q(A, tau);
        auto R = qr_extract_R(A);

        REQUIRE(unitarity_error(Q) < 1e-12);

        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                cd s(0, 0);
                for (std::size_t c = 0; c < m; ++c) s += Q(i,c) * R(c,j);
                REQUIRE(std::abs(s - A0(i,j)) < 1e-12);
            }

        // R is upper triangular.
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n && j < i; ++j)
                REQUIRE(std::abs(R(i,j)) < 1e-14);
    }
}

TEST_CASE("Complex QR solves a square system (#353)", "[operation][qr][regression]") {
    const std::size_t n = 5;
    mat::dense2D<cd> A(n, n), A0(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            A(i,j) = cd(cx_next(), cx_next()) + (i == j ? cd(4,0) : cd(0,0));
            A0(i,j) = A(i,j);
        }
    vec::dense_vector<cd> xt(n), b(n), x(n), tau;
    for (std::size_t i = 0; i < n; ++i) xt[i] = cd(cx_next(), cx_next());
    for (std::size_t i = 0; i < n; ++i) {
        cd s(0,0);
        for (std::size_t j = 0; j < n; ++j) s += A0(i,j) * xt[j];
        b[i] = s;
    }
    REQUIRE(qr_factor(A, tau) == 0);
    qr_solve(A, tau, x, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(x(i) - xt(i)) < 1e-10);
}

TEST_CASE("Complex LQ: Q is unitary and L*Q reproduces A (#353)",
          "[operation][lq][regression]") {
    // LQ is NOT symmetric with QR here, and the difference is the substance of
    // the fix. householder() annihilates a COLUMN; LQ annihilates a ROW, so the
    // reflector is built from the conjugated row and applied as H^H. Getting
    // only one of those two right leaves Q perfectly unitary while L*Q differs
    // from A by O(1) -- which is what the first attempt produced, and why the
    // reconstruction check below matters more than the unitarity check.
    for (auto shape : {std::pair<std::size_t,std::size_t>{4,4}, {3,6}, {6,3}}) {
        const std::size_t m = shape.first, n = shape.second;
        mat::dense2D<cd> A(m, n), A0(m, n);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) { A(i,j) = cd(cx_next(), cx_next()); A0(i,j) = A(i,j); }

        vec::dense_vector<cd> tau;
        INFO("m = " << m << ", n = " << n);
        REQUIRE(lq_factor(A, tau) == 0);
        auto Q = lq_extract_Q(A, tau);
        auto L = lq_extract_L(A);

        REQUIRE(unitarity_error(Q) < 1e-12);

        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                cd s(0, 0);
                for (std::size_t c = 0; c < n; ++c) s += L(i,c) * Q(c,j);
                REQUIRE(std::abs(s - A0(i,j)) < 1e-12);
            }
    }
}

TEST_CASE("Complex Householder: H*x = beta*e_1 with beta real, and H is unitary (#353)",
          "[operation][qr][regression]") {
    const std::size_t n = 4;
    vec::dense_vector<cd> x(n);
    x[0] = cd(1.5,-2.0); x[1] = cd(0.5,1.0); x[2] = cd(-2.0,0.25); x[3] = cd(3.0,-1.5);

    auto [v, tau] = householder(x);
    REQUIRE(std::abs(v(0) - cd(1,0)) < 1e-15);      // v(0) is implicit 1

    cd vhx(0,0);
    for (std::size_t i = 0; i < n; ++i) vhx += std::conj(v(i)) * x(i);

    double nx2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) nx2 += std::norm(x(i));
    const double nrm = std::sqrt(nx2);

    const cd h0 = x(0) - tau * v(0) * vhx;
    REQUIRE(std::abs(h0.imag()) < 1e-12);            // beta is REAL
    REQUIRE(std::abs(std::abs(h0) - nrm) < 1e-12);   // |beta| == ||x||
    for (std::size_t i = 1; i < n; ++i)
        REQUIRE(std::abs(x(i) - tau * v(i) * vhx) < 1e-12);   // tail annihilated
}

TEST_CASE("Complex Householder edge cases: negligible tail and n == 1 (#353)",
          "[operation][qr][regression]") {
    // Found in review. The `sigma == 0` shortcut ran BEFORE the real/complex
    // split, so a vector whose tail vanishes but whose leading entry is complex
    // returned tau = 0 -- the identity reflection, leaving H*x = x(0)*e_1 with a
    // COMPLEX leading entry and breaking the documented "beta is real"
    // guarantee. Silent, because Q stayed unitary and Q*R still reproduced A;
    // the only visible symptom was a complex R diagonal.
    SECTION("negligible tail, complex leading entry") {
        vec::dense_vector<cd> x(4);
        x[0] = cd(1.0, 1.0); x[1] = cd(0,0); x[2] = cd(0,0); x[3] = cd(0,0);

        auto [v, tau] = householder(x);
        cd vhx(0,0);
        for (std::size_t i = 0; i < 4; ++i) vhx += std::conj(v(i)) * x(i);
        const cd h0 = x(0) - tau * v(0) * vhx;

        REQUIRE(std::abs(h0.imag()) < 1e-12);                        // beta REAL
        REQUIRE(std::abs(std::abs(h0) - std::abs(x(0))) < 1e-12);    // |beta| == |x0|
        for (std::size_t i = 1; i < 4; ++i)
            REQUIRE(std::abs(x(i) - tau * v(i) * vhx) < 1e-12);
    }

    SECTION("real leading entry keeps the identity shortcut") {
        // The complement of the case above: when x(0) is already real there is
        // nothing to rotate, and tau must still be exactly zero.
        vec::dense_vector<cd> x(3);
        x[0] = cd(2.0, 0.0); x[1] = cd(0,0); x[2] = cd(0,0);
        auto [v, tau] = householder(x);
        REQUIRE(tau == cd(0,0));
    }

    SECTION("n == 1") {
        vec::dense_vector<cd> y(1);
        y[0] = cd(-2.0, 0.5);
        auto [v1, tau1] = householder(y);
        REQUIRE(v1.size() == 1);
        REQUIRE(std::abs(v1(0) - cd(1,0)) < 1e-15);
        // A 1-element reflector still removes the phase: H*y = beta, beta real.
        const cd h0 = y(0) - tau1 * v1(0) * (std::conj(v1(0)) * y(0));
        REQUIRE(std::abs(h0.imag()) < 1e-12);
        REQUIRE(std::abs(std::abs(h0) - std::abs(y(0))) < 1e-12);
    }

    SECTION("n == 0") {
        vec::dense_vector<cd> z(0);
        auto [v0, tau0] = householder(z);
        REQUIRE(v0.size() == 0);
        REQUIRE(tau0 == cd(0,0));
    }
}

TEST_CASE("Complex QR produces a real R diagonal (#353)", "[operation][qr][regression]") {
    // The consequence of the fix above, and the property LAPACK guarantees.
    // This is what a caller relying on a real triangular diagonal would notice.
    const std::size_t n = 5;
    mat::dense2D<cd> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i,j) = cd(cx_next(), cx_next());
    // Force the awkward shape too: a column already along e_1 with complex head.
    A(0,0) = cd(1.0, 1.0);
    for (std::size_t i = 1; i < n; ++i) A(i,0) = cd(0,0);

    vec::dense_vector<cd> tau;
    REQUIRE(qr_factor(A, tau) == 0);
    auto R = qr_extract_R(A);
    for (std::size_t i = 0; i < n; ++i) {
        INFO("R(" << i << "," << i << ") = " << R(i,i).real() << " + " << R(i,i).imag() << "i");
        REQUIRE(std::abs(R(i,i).imag()) < 1e-12);
    }
}
