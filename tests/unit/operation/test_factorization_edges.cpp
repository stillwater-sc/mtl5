// Edge cases of the dense factorizations and solves (#854): NaN pivots,
// singular working factors, qr_solve's shape, LU/inv/LQ edges, trsm with
// alpha = 0, and the Bunch-Kaufman update's loop order.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/ldlt_bk.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/lu_iterative_refine.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/trsm.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

const double qnan = std::numeric_limits<double>::quiet_NaN();

template <typename M = mat::dense2D<double>>
M random_matrix(std::size_t m, std::size_t n, unsigned seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    M A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = u(g);
    return A;
}

} // namespace

TEST_CASE("a NaN pivot stops the factorization", "[operation][ldlt][lu][854]") {
    // Only pivot == 0 was tested, so [[NaN, 1], [1, 2]] factored with NaN in D
    // and L and info 0.
    mat::dense2D<double> A{{qnan, 1}, {1, 2}};
    {
        auto F = A;
        CHECK(ldlt_factor(F) == 1);
        auto G = A;
        CHECK(ldlt_factor<double>(G) == 1);   // the accumulator path
    }
    {
        using C = std::complex<double>;
        mat::dense2D<C> H{{C(qnan, 0), C(1, 1)}, {C(1, -1), C(2, 0)}};
        CHECK(ldlt_h_factor(H) == 1);
    }
    {
        auto F = A;
        bk_pivot_info p;
        // The NaN fails Bunch-Kaufman's pivot comparisons, so which step
        // reports it depends on the pivoting; it returned 0 before.
        CHECK(ldlt_bk_factor(F, p) != 0);
        auto G = A;
        bk_pivot_info q;
        CHECK(ldlt_bk_factor<double>(G, q) != 0);
    }
    {
        // lu_factor's partial pivoting compares |a| > best, which a NaN in the
        // pivot position wins by default.
        mat::dense2D<double> L{{qnan, 1}, {0.5, 2}};
        std::vector<std::size_t> piv;
        auto L2 = L;
        CHECK(lu_factor<double>(L, piv) == 1);
        // getrf where LAPACK is built, which flags only an exact zero; its
        // idamax may or may not pick the NaN row, so the step can be 1 or 2.
        CHECK(lu_factor(L2, piv) != 0);
    }
    // A NaN further down: the pivot it produces is NaN.
    mat::dense2D<double> B{{2, 1, 0}, {1, qnan, 1}, {0, 1, 3}};
    auto F = B;
    CHECK(ldlt_factor(F) == 2);
}

TEST_CASE("lu_iterative_refine reports a singular working factor", "[operation][lu][854]") {
    // Rank 2 in every precision: lu_solve divided by the zero pivot, and x
    // came back inf/NaN with converged = false and no error.
    mat::dense2D<double> A{{1, 2, 3}, {2, 4, 6}, {1, 0, 1}};
    vec::dense_vector<double> b{1.0, 2.0, 3.0}, x(3);
    CHECK_THROWS_AS(lu_iterative_refine<float>(A, b, x), std::runtime_error);
    CHECK_THROWS_AS(lu_iterative_refine<double>(A, b, x), std::runtime_error);
    // An entry that overflows the working type becomes Inf, a usable pivot:
    // only a zero or NaN one throws.
}

TEST_CASE("qr_solve: a tall least-squares solve, and its shape checks", "[operation][qr][854]") {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{6, 3}, {9, 9}, {20, 4}}) {
        INFO(m << " x " << n);
        const auto A = random_matrix(m, n, unsigned(m * 31 + n));
        auto QR = A;
        vec::dense_vector<double> tau;
        REQUIRE(qr_factor(QR, tau) == 0);
        vec::dense_vector<double> b(m), x(n);
        for (std::size_t i = 0; i < m; ++i) b(i) = double(i % 4) - 1.5;
        qr_solve(QR, tau, x, b);
        // The least-squares optimality condition: A^T (A x - b) = 0.
        double worst = 0, scale = 0;
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0;
            for (std::size_t i = 0; i < m; ++i) {
                double r = -b(i);
                for (std::size_t k = 0; k < n; ++k) r += A(i, k) * x(k);
                s += A(i, j) * r;
            }
            worst = std::max(worst, std::abs(s));
        }
        for (std::size_t i = 0; i < m; ++i) scale = std::max(scale, std::abs(b(i)));
        CHECK(worst <= 1e-12 * double(m) * scale);
    }
    // m < n: it read y(i) and R(i, i) past row m.
    auto W = random_matrix(3, 5, 1);
    vec::dense_vector<double> tau;
    REQUIRE(qr_factor(W, tau) == 0);
    vec::dense_vector<double> x5(5), b3(3, 1.0);
    CHECK_THROWS_AS(qr_solve(W, tau, x5, b3), std::invalid_argument);
    auto T = random_matrix(5, 3, 2);
    REQUIRE(qr_factor(T, tau) == 0);
    vec::dense_vector<double> x3(3), b4(4, 1.0);
    CHECK_THROWS_AS(qr_solve(T, tau, x3, b4), std::invalid_argument);
}

TEST_CASE("qr_extract_Q: Q R = A, Q orthogonal, for tall and square A", "[operation][qr][854]") {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{7, 3}, {6, 6}, {12, 5}}) {
        INFO(m << " x " << n);
        const auto A = random_matrix(m, n, unsigned(m + 7 * n));
        auto QR = A;
        vec::dense_vector<double> tau;
        REQUIRE(qr_factor(QR, tau) == 0);
        const auto Q = qr_extract_Q(QR, tau);
        const auto R = qr_extract_R(QR);
        double back = 0, orth = 0;
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double s = 0;
                for (std::size_t k = 0; k < m; ++k) s += Q(i, k) * R(k, j);
                back = std::max(back, std::abs(s - A(i, j)));
            }
            for (std::size_t j = 0; j < m; ++j) {
                double s = 0;
                for (std::size_t k = 0; k < m; ++k) s += Q(k, i) * Q(k, j);
                orth = std::max(orth, std::abs(s - (i == j ? 1.0 : 0.0)));
            }
        }
        CHECK(back <= 1e-14 * double(m));
        CHECK(orth <= 1e-14 * double(m));
    }
}

TEST_CASE("LU and inv: singular, 0x0, 1x1, both orientations", "[operation][lu][854]") {
    // Singular, on the in-house path in every build.
    mat::dense2D<double> S{{1, 2}, {2, 4}};
    std::vector<std::size_t> piv;
    auto F = S;
    CHECK(lu_factor<double>(F, piv) == 2);
    CHECK_THROWS_AS(inv(S), std::runtime_error);
    CHECK_THROWS_AS(inv<double>(S), std::runtime_error);
    // Empty and 1x1.
    mat::dense2D<double> E(0, 0);
    CHECK(lu_factor(E, piv) == 0);
    CHECK(inv(E).num_rows() == 0);
    mat::dense2D<double> one{{4.0}};
    const auto I1 = inv(one);
    CHECK(I1(0, 0) == 0.25);
    vec::dense_vector<double> b1{2.0}, x1(1);
    auto L1 = one;
    REQUIRE(lu_factor(L1, piv) == 0);
    lu_solve(L1, piv, x1, b1);
    CHECK(x1(0) == 0.5);
    // Not square.
    CHECK_THROWS_AS(inv(mat::dense2D<double>(2, 3)), std::invalid_argument);
    // Row- and column-major give the same solution.
    const auto A = random_matrix(7, 7, 11);
    const auto Ac = random_matrix<mat::dense2D<double, mat::parameters<tag::col_major>>>(7, 7, 11);
    vec::dense_vector<double> b(7), xr(7), xc(7);
    for (std::size_t i = 0; i < 7; ++i) b(i) = double(i) - 3;
    auto Lr = A;
    auto Lc = Ac;
    std::vector<std::size_t> pr, pc;
    REQUIRE(lu_factor<double>(Lr, pr) == 0);
    REQUIRE(lu_factor<double>(Lc, pc) == 0);
    lu_solve<double>(Lr, pr, xr, b);
    lu_solve<double>(Lc, pc, xc, b);
    for (std::size_t i = 0; i < 7; ++i) CHECK(std::abs(xr(i) - xc(i)) <= 1e-13);
}

TEST_CASE("LQ of a tall matrix", "[operation][lq][854]") {
    // Only m <= n was tested.
    const std::size_t m = 6, n = 3;
    const auto A = random_matrix(m, n, 21);
    auto F = A;
    vec::dense_vector<double> tau;
    REQUIRE(lq_factor(F, tau) == 0);
    const auto Q = lq_extract_Q(F, tau);
    const auto L = lq_extract_L(F);
    REQUIRE(L.num_rows() == m);
    double back = 0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0;
            for (std::size_t k = 0; k < L.num_cols(); ++k) s += L(i, k) * Q(k, j);
            back = std::max(back, std::abs(s - A(i, j)));
        }
    CHECK(back <= 1e-14);
}

TEST_CASE("trsm with alpha = 0 gives zero, NaN or Inf in B included", "[operation][trsm][854]") {
    // BLAS ?trsm sets B to 0 without reading it; the generic path formed
    // 0 * B and turned a NaN or Inf into NaN.
    mat::dense2D<double> L{{2, 0}, {1, 3}};
    for (bool upper : {false, true}) {
        mat::dense2D<double> B{{qnan, 1}, {std::numeric_limits<double>::infinity(), 2}};
        auto C = B;
        trsm(0.0, L, B, upper);
        trsm<double>(0.0, L, C, upper);   // the accumulator path
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 2; ++j) {
                CHECK(B(i, j) == 0.0);
                CHECK(C(i, j) == 0.0);
            }
    }
}

TEST_CASE("ldlt_bk_factor: row- and column-major give the same factor", "[operation][ldlt_bk][854]") {
    // The trailing update runs by rows on row-major storage and by columns
    // otherwise (#854); each entry gets the same update either way.
    const std::size_t n = 40;
    auto A = random_matrix(n, n, 31);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) A(j, i) = A(i, j);
    mat::dense2D<double, mat::parameters<tag::col_major>> Ac(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Ac(i, j) = A(i, j);
    bk_pivot_info pr, pc;
    REQUIRE(ldlt_bk_factor(A, pr) == 0);
    REQUIRE(ldlt_bk_factor(Ac, pc) == 0);
    CHECK(pr.ipiv == pc.ipiv);
    bool same = true;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) same = same && A(i, j) == Ac(i, j);
    CHECK(same);
}
