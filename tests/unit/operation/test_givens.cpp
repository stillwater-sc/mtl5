// Givens rotations, real and complex (#853): math::givens_rotation and the
// GMRES helpers apply_givens_rotation / apply_stored_rotation.
//
// The complex branch used to take the real formula over complex numbers,
// r = sqrt(a*a + b*b) -- a complex square, which is 0 for b = i a -- and the
// helpers applied s unconjugated, so the rotation was not unitary.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/math/pow2_scaling.hpp>
#include <mtl/operation/givens.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;
using C = std::complex<double>;

namespace {

constexpr double tol = 8 * std::numeric_limits<double>::epsilon();

/// G = [c s; -conj(s) c] is unitary and maps (a, b) to (r, 0), |r| = |(a, b)|.
void check_rotation(C a, C b) {
    INFO("a = " << a << ", b = " << b);
    const auto g = math::givens_rotation(a, b);
    CHECK(std::abs(g.c.imag()) == 0.0);                               // c is real
    CHECK(std::abs(std::norm(g.c) + std::norm(g.s) - 1.0) <= tol);   // unitary
    const C top = g.c * a + g.s * b;
    const C bottom = -std::conj(g.s) * a + g.c * b;
    const double scale = std::hypot(std::abs(a), std::abs(b));
    CHECK(std::abs(top - g.r) <= tol * scale);
    CHECK(std::abs(bottom) <= tol * scale);
    CHECK(std::abs(std::abs(g.r) - scale) <= tol * scale);
}

} // namespace

TEST_CASE("complex givens_rotation is unitary and zeroes b", "[givens][853]") {
    // (1, i): r was 0 and the rotation the identity.
    check_rotation(C(1, 0), C(0, 1));
    // (1+i, 1-i): r was 0, |r| = 2.
    check_rotation(C(1, 1), C(1, -1));
    const auto g = math::givens_rotation(C(1, 1), C(1, -1));
    CHECK(std::abs(std::abs(g.r) - 2.0) <= tol * 2);
    check_rotation(C(3, -4), C(-2, 7));
    check_rotation(C(0, 2), C(5, 0));
    check_rotation(C(1e200, 1e200), C(-1e200, 3e200));   // scaled like the real case
    check_rotation(C(1e-200, 0), C(0, 2e-200));
}

TEST_CASE("complex givens_rotation: zero arguments", "[givens][853]") {
    auto g = math::givens_rotation(C(2, -1), C(0, 0));     // b = 0: the identity, r = a
    CHECK((g.c == C(1) && g.s == C(0) && g.r == C(2, -1)));
    g = math::givens_rotation(C(0, 0), C(0, 3));            // a = 0: c = 0, r = |b|
    CHECK(g.c == C(0));
    CHECK(g.r == C(3));
    CHECK(std::abs(g.s - C(0, -1)) <= tol);                 // conj(b) / |b|
    check_rotation(C(0, 0), C(0, 3));
    g = math::givens_rotation(C(0, 0), C(0, 0));
    CHECK((g.c == C(1) && g.s == C(0) && g.r == C(0)));
}

TEST_CASE("real givens_rotation is unchanged", "[givens][853]") {
    const auto g = math::givens_rotation(3.0, 4.0);
    CHECK((g.c == 0.6 && g.s == 0.8 && g.r == 5.0));
    const auto h = math::givens_rotation(-3.0, 4.0);
    CHECK((h.c == -0.6 && h.s == 0.8 && h.r == 5.0));
}

TEST_CASE("apply_givens_rotation and apply_stored_rotation, complex", "[givens][853]") {
    // A 3x2 Hessenberg block; rotate column 0, then apply the stored rotation
    // to column 1. The rotated H must be G H for the unitary G, and g likewise.
    mat::dense2D<C> H(3, 2);
    H(0, 0) = C(1, 1);  H(0, 1) = C(2, -1);
    H(1, 0) = C(1, -1); H(1, 1) = C(0, 3);
    H(2, 0) = C(0);     H(2, 1) = C(1, 1);
    const mat::dense2D<C> H0(H);
    vec::dense_vector<C> g(3), c(2), s(2);
    g(0) = C(2, 0); g(1) = C(0); g(2) = C(0);
    apply_givens_rotation(H, g, c, s, std::size_t(0));
    CHECK(std::abs(H(1, 0)) == 0.0);
    CHECK(std::abs(std::abs(H(0, 0)) - 2.0) <= tol * 2);
    apply_stored_rotation(H, c, s, std::size_t(0), std::size_t(1));
    // G applied to rows 0, 1 of column 1.
    const C top = c(0) * H0(0, 1) + s(0) * H0(1, 1);
    const C bottom = -std::conj(s(0)) * H0(0, 1) + c(0) * H0(1, 1);
    CHECK(std::abs(H(0, 1) - top) <= tol * 4);
    CHECK(std::abs(H(1, 1) - bottom) <= tol * 4);
    // Unitary: column norms are kept.
    const double before = std::norm(H0(0, 1)) + std::norm(H0(1, 1));
    const double after = std::norm(H(0, 1)) + std::norm(H(1, 1));
    CHECK(std::abs(before - after) <= tol * 8);
    // The right-hand side: g = G (2, 0), so |g(0)|^2 + |g(1)|^2 = 4.
    CHECK(std::abs(std::norm(g(0)) + std::norm(g(1)) - 4.0) <= tol * 8);
}
