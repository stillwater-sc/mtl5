// MTL5 -- Tests for the SuiteSparseQR (SPQR) binding (#774, #839)
//
// These tests only do real work when MTL5_HAS_SPQR is defined (CMake option
// MTL5_WITH_SUITESPARSE_SPQR=ON with SuiteSparse installed). Otherwise a
// placeholder confirms the test infrastructure compiles and links.
#include <catch2/catch_test_macros.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

#ifdef MTL5_HAS_SPQR

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <mtl/interface/spqr.hpp>

using mtl::interface::spqr_solver;

namespace {

using csr = mat::compressed2D<double>;

/// A non-symmetric m x n matrix (m >= n) with full column rank: a dominant
/// diagonal, a superdiagonal and a sub-diagonal of different values, and, for
/// m > n, extra rows below.
csr tall(std::size_t m, std::size_t n) {
    csr A(m, n);
    {   // the inserter writes A when it closes, which must be before the return
        mat::inserter<csr> ins(A);
        for (std::size_t r = 0; r < m; ++r) {
            if (r < n) ins[r][r] << 5.0 + double(r % 3);
            if (r + 1 < n) ins[r][r + 1] << 2.0;          // superdiagonal
            if (r > 0 && r - 1 < n) ins[r][r - 1] << -1.0; // subdiagonal
            if (r >= n) ins[r][r % n] << 1.0 + double(r % 5);
        }
    }
    return A;
}

vec::dense_vector<double> rhs(std::size_t m) {
    vec::dense_vector<double> b(m);
    for (std::size_t i = 0; i < m; ++i) b(i) = 1.0 + double(i % 7);
    return b;
}

/// ||A^T (b - A x)||_inf / (||A||_F^2 ||x||_inf + ||A||_F ||b||_inf): zero at
/// the least-squares solution, whatever the residual itself is. Infinite for a
/// nonfinite x.
double normal_eq_error(const csr& A, const vec::dense_vector<double>& x,
                       const vec::dense_vector<double>& b) {
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!std::isfinite(x(i))) return std::numeric_limits<double>::infinity();
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& v  = A.ref_data();
    std::vector<double> r(A.num_rows()), g(A.num_cols(), 0.0);
    double fro2 = 0, xn = 0, bn = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double ax = 0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) {
            ax += v[k] * x(ci[k]);
            fro2 += v[k] * v[k];
        }
        r[i] = b(i) - ax;
        bn = std::max(bn, std::abs(b(i)));
    }
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) g[ci[k]] += v[k] * r[i];
    double gn = 0;
    for (double gi : g) gn = std::max(gn, std::abs(gi));
    for (std::size_t j = 0; j < x.size(); ++j) xn = std::max(xn, std::abs(x(j)));
    return gn / (fro2 * xn + std::sqrt(fro2) * bn);
}

constexpr double tol = 1e-13;

} // namespace

TEST_CASE("spqr: square non-symmetric system, exact solution", "[interface][spqr]") {
    // A = [2 1; 0 3] is not symmetric, so solving with A^T instead of A
    // (reading the CSR arrays as CCS) gives a different x. b = A [1; 2].
    csr A(2, 2);
    {
        mat::inserter<csr> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][1] << 3.0;
    }
    vec::dense_vector<double> b = {4.0, 6.0}, x(2, 0.0);
    interface::spqr_solve(A, x, b);
    REQUIRE(std::abs(x(0) - 1.0) < tol);
    REQUIRE(std::abs(x(1) - 2.0) < tol);
}

TEST_CASE("spqr: square and overdetermined least squares", "[interface][spqr]") {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{60, 60}, {90, 40}}) {
        const csr A = tall(m, n);
        const auto b = rhs(m);
        spqr_solver s(A);
        REQUIRE(s.num_rows() == m);
        REQUIRE(s.num_cols() == n);
        vec::dense_vector<double> x(n, 0.0);
        s.solve(x, b);
        REQUIRE(normal_eq_error(A, x, b) < tol);
        // A second solve on the same solver gives the same answer.
        vec::dense_vector<double> y(n, 0.0);
        s.solve(y, b);
        for (std::size_t j = 0; j < n; ++j) REQUIRE(y(j) == x(j));
    }
}

TEST_CASE("spqr: argument checks", "[interface][spqr]") {
    REQUIRE_THROWS_AS(spqr_solver(csr(3, 4)), std::invalid_argument);   // m < n
    // n = INT_MAX + 1 with no entries costs two row pointers: the size check
    // comes before the shape check, so this is the int32 limit, not m < n.
    REQUIRE_THROWS_AS(spqr_solver(csr(1, std::size_t(INT_MAX) + 1)), std::length_error);
    spqr_solver s(tall(10, 6));
    vec::dense_vector<double> x(6, 0.0), b(9, 1.0);
    REQUIRE_THROWS_AS(s.solve(x, b), std::invalid_argument);
    vec::dense_vector<double> x5(5, 0.0), b10(10, 1.0);
    REQUIRE_THROWS_AS(s.solve(x5, b10), std::invalid_argument);
}

TEST_CASE("spqr: 1x1, 2x1 and empty systems", "[interface][spqr]") {
    csr one(1, 1);
    { mat::inserter<csr> ins(one); ins[0][0] << 4.0; }
    vec::dense_vector<double> x(1, 0.0), b(1, 2.0);
    interface::spqr_solve(one, x, b);
    REQUIRE(std::abs(x(0) - 0.5) < tol);

    // min ||[1; 1] x - [1; 3]|| is x = 2.
    csr col(2, 1);
    { mat::inserter<csr> ins(col); ins[0][0] << 1.0; ins[1][0] << 1.0; }
    vec::dense_vector<double> b2 = {1.0, 3.0};
    interface::spqr_solve(col, x, b2);
    REQUIRE(std::abs(x(0) - 2.0) < tol);

    // 0x0: nothing to solve, and no allocation or free goes wrong.
    csr empty(0, 0);
    vec::dense_vector<double> x0(0), b0(0);
    spqr_solver s(empty);
    REQUIRE(s.num_rows() == 0);
    REQUIRE(s.num_cols() == 0);
    s.solve(x0, b0);
}

TEST_CASE("spqr: move construction and move assignment", "[interface][spqr]") {
    static_assert(std::is_nothrow_move_constructible_v<spqr_solver>);
    static_assert(std::is_nothrow_move_assignable_v<spqr_solver>);
    const csr A = tall(30, 20);
    const auto b = rhs(30);
    vec::dense_vector<double> x(20, 0.0);

    // Both objects are destroyed at the end of the scope; the workspace moved
    // as a pointer, so it is finished once. The struct copy this replaced
    // finished it twice, a double free once a solve had filled the workspace
    // -- hence the solve before the move. Only AddressSanitizer reports it.
    spqr_solver a(A);
    a.solve(x, b);
    spqr_solver moved(std::move(a));
    moved.solve(x, b);
    REQUIRE(normal_eq_error(A, x, b) < tol);
    REQUIRE_THROWS_AS(a.solve(x, b), std::logic_error);   // NOLINT(bugprone-use-after-move)
    REQUIRE(a.num_rows() == 0);                           // NOLINT(bugprone-use-after-move)
    REQUIRE(a.num_cols() == 0);                           // NOLINT(bugprone-use-after-move)

    const csr B = tall(12, 12);
    spqr_solver c(B);
    c = std::move(moved);                   // frees B's copy, takes A's
    REQUIRE(moved.num_rows() == 0);         // NOLINT(bugprone-use-after-move)
    REQUIRE(c.num_rows() == 30);
    std::fill(x.begin(), x.end(), 0.0);
    c.solve(x, b);
    REQUIRE(normal_eq_error(A, x, b) < tol);

    a = spqr_solver(B);                     // a moved-from solver can be assigned again
    vec::dense_vector<double> y(12, 0.0);
    const auto bb = rhs(12);
    a.solve(y, bb);
    REQUIRE(normal_eq_error(B, y, bb) < tol);
}

TEST_CASE("spqr: concurrent solves on one const solver", "[interface][spqr]") {
    // Each solve runs on a workspace of its own (#839). This is a smoke test:
    // it checks the answers of concurrent solves, and it passed on the shared
    // workspace too. The race was on the workspace's counters inside
    // SuiteSparse, which is not instrumented, so ThreadSanitizer does not see
    // it either. What rules it out is the code: solve() reads the solver's
    // cholmod_common only to copy its settings, and otherwise only the stored matrix.
    const csr A = tall(80, 50);
    const spqr_solver s(A);
    constexpr int threads = 4, rounds = 10;
    std::vector<vec::dense_vector<double>> bs, xs;
    for (int t = 0; t < threads; ++t) {
        vec::dense_vector<double> b(80);
        for (std::size_t i = 0; i < 80; ++i) b(i) = 1.0 + double((i + 3 * t) % 11);
        bs.push_back(b);
        xs.emplace_back(50, 0.0);
    }
    std::vector<int> ok(threads, 0);
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t)
            pool.emplace_back([&, t] {
                for (int r = 0; r < rounds; ++r) {
                    s.solve(xs[t], bs[t]);
                    ok[t] += normal_eq_error(A, xs[t], bs[t]) < tol;
                }
            });
        for (auto& th : pool) th.join();
    }
    for (int t = 0; t < threads; ++t) REQUIRE(ok[t] == rounds);
}

#else

TEST_CASE("spqr: placeholder (SPQR not available)", "[interface][spqr]") {
    REQUIRE(true);
}

#endif // MTL5_HAS_SPQR
