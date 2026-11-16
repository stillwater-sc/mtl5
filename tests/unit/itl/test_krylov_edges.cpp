// Krylov solvers and iteration objects: the edge cases of issue 866 (QA
// sweep #718).
//
//   - bicgstab_ell applies its last BiCG update before the early exit, and
//     judges the residual there;
//   - bicgstab and bicgstab_ell test only convergence mid-iteration, so the
//     last allowed iteration is a full one; bicgstab reports omega == 0;
//   - gmres rejects restart < 1, idr_s s < 1, cyclic_iteration cycle < 1;
//     idr_s with max_iter = 0 makes no step;
//   - an iteration object reused for a second solve works for every solver;
//   - cyclic_iteration prints its summary once, through `return iter;`;
//   - breakdowns are reported (cg, bicg, bicgstab, cgs, tfqmr, bicgstab_ell);
//   - max_iter = 0, zero rhs, a converged x0 and 1x1 systems for each solver;
//   - relresid() and the atol test; minres with a varying diagonal PC.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/itl.hpp>

using namespace mtl;

namespace {

using csr  = mat::compressed2D<double>;
using vec_t   = vec::dense_vector<double>;
using entries = std::vector<std::tuple<std::size_t, std::size_t, double>>;
using iter_t  = itl::basic_iteration<double>;
using pc_id   = itl::pc::identity<csr>;

csr make(std::size_t n, const entries& e) {
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (const auto& [i, j, v] : e) ins[i][j] << v;
    }
    return A;
}

csr scaled_identity(std::size_t n, double d) {
    entries e;
    for (std::size_t i = 0; i < n; ++i) e.emplace_back(i, i, d);
    return make(n, e);
}

/// tridiag(-1, 4, -1) plus a small skew part, nonsymmetric when skew != 0.
csr tridiagonal(std::size_t n, double skew = 0.0) {
    entries e;
    for (std::size_t i = 0; i < n; ++i) {
        e.emplace_back(i, i, 4.0);
        if (i + 1 < n) { e.emplace_back(i, i + 1, -1.0 + skew); e.emplace_back(i + 1, i, -1.0 - skew); }
    }
    return make(n, e);
}

vec_t ramp(std::size_t n) {
    vec_t b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + double(i % 3);
    return b;
}

double true_residual(const csr& A, const vec_t& x, const vec_t& b) {
    const auto& rp = A.ref_major(); const auto& ci = A.ref_minor(); const auto& d = A.ref_data();
    double r2 = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) ax += d[k] * x(ci[k]);
        r2 += (b(i) - ax) * (b(i) - ax);
    }
    return std::sqrt(r2);
}

double norm(const vec_t& v) { return mtl::two_norm(v); }

/// Every solver, as solve(A, x, b, pc, iter) with an identity PC.
struct named_solver {
    const char* name;
    std::function<int(const csr&, vec_t&, const vec_t&, const pc_id&, iter_t&)> solve;
};

std::vector<named_solver> all_solvers() {
    return {
        {"cg",             [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::cg(A, x, b, M, it); }},
        {"cgs",            [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::cgs(A, x, b, M, it); }},
        {"bicg",           [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::bicg(A, x, b, M, it); }},
        {"bicgstab",       [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::bicgstab(A, x, b, M, it); }},
        {"bicgstab_ell(2)",[](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::bicgstab_ell(A, x, b, M, it, 2); }},
        {"qmr",            [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::qmr(A, x, b, M, it); }},
        {"tfqmr",          [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::tfqmr(A, x, b, M, it); }},
        {"minres",         [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::minres(A, x, b, M, it); }},
        {"gmres(10)",      [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::gmres(A, x, b, M, it, 10); }},
        {"idr_s(2)",       [](auto& A, auto& x, auto& b, auto& M, auto& it) { return itl::idr_s(A, x, b, M, it, 2); }},
    };
}

} // namespace

// --- bicgstab_ell's early exit ---------------------------------------------------

TEST_CASE("bicgstab_ell: the early exit keeps its last update", "[itl][bicgstab_ell][866]") {
    for (std::size_t n : {1u, 2u, 3u}) {
        for (std::size_t ell : {1u, 2u, 3u}) {
            CAPTURE(n, ell);
            // A = 2I, b = 1, x0 = 0: it returned x = 0 with error 0 (answer 0.5).
            const csr A = scaled_identity(n, 2.0);
            const vec_t b(n, 1.0);
            vec_t x(n, 0.0);
            const pc_id M(A);
            iter_t iter(b, 100, 1e-12);
            REQUIRE(itl::bicgstab_ell(A, x, b, M, iter, ell) == 0);
            for (std::size_t i = 0; i < n; ++i) CHECK(std::abs(x(i) - 0.5) < 1e-14);
        }
    }
}

TEST_CASE("bicgstab_ell: the reported residual is the true one", "[itl][bicgstab_ell][866]") {
    {
        // [[4,1],[2,3]], b = (1,2), ell = 2: reported 1.3e-15, true 0.41.
        const csr A = make(2, {{0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 2.0}, {1, 1, 3.0}});
        vec_t b(2), x(2, 0.0);
        b(0) = 1.0; b(1) = 2.0;
        const pc_id M(A);
        iter_t iter(b, 100, 1e-12);
        REQUIRE(itl::bicgstab_ell(A, x, b, M, iter, 2) == 0);
        CHECK(true_residual(A, x, b) < 1e-11 * norm(b));
    }
    for (std::size_t ell : {1u, 2u, 4u}) {
        // 30-point tridiagonal, rtol 1e-8: reported 3.9e-8, true 1.36e-7.
        CAPTURE(ell);
        const csr A = tridiagonal(30, 0.3);
        const vec_t b = ramp(30);
        vec_t x(30, 0.0);
        const itl::pc::diagonal<csr> M(A);
        iter_t iter(b, 500, 1e-8);
        REQUIRE(itl::bicgstab_ell(A, x, b, M, iter, ell) == 0);
        CHECK(true_residual(A, x, b) <= 2e-8 * norm(b));
    }
}

// --- the last allowed iteration ----------------------------------------------------

TEST_CASE("bicgstab: max_iter = 1 makes one full iteration", "[itl][bicgstab][866]") {
    const csr A = tridiagonal(30, 0.2);
    const vec_t b = ramp(30);
    // One BiCGSTAB step from x = 0 by hand: p = r = b, v = A p,
    // alpha = r.r / r.v, s = r - alpha v, t = A s, omega = t.s / t.t,
    // x = alpha p + omega s.
    vec_t v = A * b;
    const double alpha = mtl::dot(b, b) / mtl::dot(b, v);
    vec_t s(30);
    for (std::size_t i = 0; i < 30; ++i) s(i) = b(i) - alpha * v(i);
    vec_t t = A * s;
    const double omega = mtl::dot(t, s) / mtl::dot(t, t);
    vec_t x(30, 0.0);
    const pc_id M(A);
    iter_t iter(b, 1, 1e-12);
    // It stopped after the half step x = alpha p: residual 0.66 from 5.5.
    CHECK(itl::bicgstab(A, x, b, M, iter) == 1);
    CHECK(iter.iterations() == 1);
    for (std::size_t i = 0; i < 30; ++i) CHECK(std::abs(x(i) - (alpha * b(i) + omega * s(i))) < 1e-14);
}

TEST_CASE("bicgstab: omega == 0 is a breakdown", "[itl][bicgstab][866]") {
    // [[1,1],[1,0]], b = e1: s = (0,-1), t = A s = (-1,0), t.s = 0. The next
    // beta divided by omega = 0, and x went to NaN.
    const csr A = make(2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}});
    vec_t b(2, 0.0), x(2, 0.0);
    b(0) = 1.0;
    const pc_id M(A);
    iter_t iter(b, 50, 1e-12);
    CHECK(itl::bicgstab(A, x, b, M, iter) == 5);
    CHECK(std::isfinite(x(0)));
    CHECK(std::isfinite(x(1)));
}

TEST_CASE("bicgstab_ell: max_iter = 1 runs the whole cycle", "[itl][bicgstab_ell][866]") {
    // max_iter = 1 ended the first cycle at its first BiCG step; with the full
    // cycle, BiCGSTAB(4) on a 4x4 system is exact after one.
    const csr A = tridiagonal(4, 0.3);
    const vec_t b = ramp(4);
    vec_t x(4, 0.0);
    const pc_id M(A);
    iter_t iter(b, 1, 1e-10);
    itl::bicgstab_ell(A, x, b, M, iter, 4);
    CHECK(true_residual(A, x, b) < 1e-10 * norm(b));
}

// --- argument checks -----------------------------------------------------------------

TEST_CASE("gmres, idr_s and cyclic_iteration reject bad arguments", "[itl][866]") {
    const csr A = tridiagonal(10);
    const vec_t b = ramp(10);
    const pc_id M(A);
    for (int restart : {0, -1}) {
        // restart = 0 spun forever; a negative one crashed.
        vec_t x(10, 0.0);
        iter_t iter(b, 100, 1e-10);
        CHECK_THROWS_AS(itl::gmres(A, x, b, M, iter, restart), std::invalid_argument);
    }
    {
        vec_t x(10, 0.0);
        iter_t iter(b, 100, 1e-10);
        CHECK_THROWS_AS(itl::idr_s(A, x, b, M, iter, 0), std::invalid_argument);
    }
    std::ostringstream out;
    CHECK_THROWS_AS(itl::cyclic_iteration<double>(b, 100, 1e-10, 0, 0.0, out), std::invalid_argument);
    CHECK_THROWS_AS(itl::cyclic_iteration<double>(1.0, 100, 1e-10, -2, 0.0, out), std::invalid_argument);
}

TEST_CASE("gmres: restart > n", "[itl][gmres][866]") {
    const csr A = tridiagonal(10, 0.2);
    const vec_t b = ramp(10);
    vec_t x(10, 0.0);
    const pc_id M(A);
    iter_t iter(b, 100, 1e-12);
    CHECK(itl::gmres(A, x, b, M, iter, 50) == 0);
    CHECK(true_residual(A, x, b) < 1e-11 * norm(b));
}

TEST_CASE("idr_s: max_iter = 0 makes no step", "[itl][idr_s][866]") {
    // A = 2I, max_iter = 0 gave x = 0.5 with error 0.
    const csr A = scaled_identity(3, 2.0);
    const vec_t b(3, 1.0);
    vec_t x(3, 0.0);
    const pc_id M(A);
    iter_t iter(b, 0, 1e-10);
    CHECK(itl::idr_s(A, x, b, M, iter, 2) == 1);
    CHECK(iter.iterations() == 0);
    CHECK(x(0) == 0.0);
}

// --- reuse, output, breakdown -----------------------------------------------------------

TEST_CASE("every solver: an iteration object reused for a second solve", "[itl][iteration][866]") {
    const csr A = tridiagonal(20);   // SPD, for cg and minres too
    const vec_t b = ramp(20);
    const pc_id M(A);
    for (const auto& s : all_solvers()) {
        CAPTURE(s.name);
        iter_t iter(b, 1000, 1e-10);
        vec_t x(20, 0.0);
        REQUIRE(s.solve(A, x, b, M, iter) == 0);
        const int first_iters = iter.iterations();
        // The second solve: cg used rho_1 = 0 (first() was false) and returned
        // NaN; gmres saw is_finished() still true and returned without a step.
        vec_t y(20, 0.0);
        CHECK(s.solve(A, y, b, M, iter) == 0);
        CHECK(iter.iterations() > first_iters);
        CHECK(true_residual(A, y, b) < 1e-9 * norm(b));
    }
}

TEST_CASE("cyclic_iteration prints its summary once, through return iter", "[itl][iteration][866]") {
    const csr A = tridiagonal(20);
    const vec_t b = ramp(20);
    const pc_id M(A);
    auto count = [](const std::string& text, const std::string& what) {
        std::size_t c = 0;
        for (std::size_t p = text.find(what); p != std::string::npos; p = text.find(what, p + 1)) ++c;
        return c;
    };
    {
        std::ostringstream out;
        itl::cyclic_iteration<double> iter(b, 1000, 1e-10, 2, 0.0, out);
        vec_t x(20, 0.0);
        // `return iter;` went through basic_iteration's operator int: no summary.
        const int rc = itl::cg(A, x, b, M, iter);
        CHECK(rc == 0);
        CHECK(count(out.str(), "converged in") == 1);
        CHECK(count(out.str(), "iteration ") >= 2);
    }
    {
        std::ostringstream out;
        itl::cyclic_iteration<double> iter(b, 1000, 1e-10, 1, 0.0, out);
        vec_t x(20, 0.0);
        // gmres asked error_code() at every restart: the summary repeated.
        CHECK(itl::gmres(A, x, b, M, iter, 3) == 0);
        CHECK(count(out.str(), "converged in") == 1);
    }
    {
        std::ostringstream out;
        itl::noisy_iteration<double> iter(b, 2, 1e-14, 0.0, out);
        vec_t x(20, 0.0);
        CHECK(itl::cg(A, x, b, M, iter) == 1);
        CHECK(count(out.str(), "NOT converged after 2 iterations") == 1);
        CHECK(count(out.str(), "iteration ") == 3);   // i = 0, 1, 2
    }
}

TEST_CASE("breakdowns are reported, not turned into NaN", "[itl][breakdown][866]") {
    // [[0,1],[1,0]], b = e1: r^T A r = 0, the first denominator of each.
    const csr A = make(2, {{0, 1, 1.0}, {1, 0, 1.0}});
    vec_t b(2, 0.0);
    b(0) = 1.0;
    const pc_id M(A);
    auto check = [&](const char* name, auto solve, int code) {
        CAPTURE(name);
        vec_t x(2, 0.0);
        iter_t iter(b, 50, 1e-12);
        CHECK(solve(x, iter) == code);
        CHECK(std::isfinite(x(0)));
        CHECK(std::isfinite(x(1)));
    };
    // cg divided rho by p^T A p = 0; bicg had no test of p~^T A p.
    check("cg", [&](vec_t& x, iter_t& it) { return itl::cg(A, x, b, M, it); }, 2);
    check("bicg", [&](vec_t& x, iter_t& it) { return itl::bicg(A, x, b, M, it); }, 3);
    check("bicgstab", [&](vec_t& x, iter_t& it) { return itl::bicgstab(A, x, b, M, it); }, 3);
    check("cgs", [&](vec_t& x, iter_t& it) { return itl::cgs(A, x, b, M, it); }, 3);
    check("tfqmr", [&](vec_t& x, iter_t& it) { return itl::tfqmr(A, x, b, M, it); }, 2);
    check("bicgstab_ell", [&](vec_t& x, iter_t& it) { return itl::bicgstab_ell(A, x, b, M, it, 2); }, 3);
    // minres and gmres do not break down here: they solve it.
    for (int which = 0; which < 2; ++which) {
        vec_t x(2, 0.0);
        iter_t iter(b, 50, 1e-12);
        const int rc = which == 0 ? itl::minres(A, x, b, M, iter) : itl::gmres(A, x, b, M, iter, 5);
        CHECK(rc == 0);
        CHECK(true_residual(A, x, b) < 1e-12);
    }
}

// --- trivial systems ------------------------------------------------------------------------

TEST_CASE("every solver: max_iter = 0, zero rhs, converged x0, 1x1", "[itl][866]") {
    const csr A = tridiagonal(6);
    const vec_t b = ramp(6);
    const pc_id M(A);
    // The exact solution, for the converged-x0 case.
    vec_t xs(6, 0.0);
    {
        iter_t iter(b, 1000, 1e-15);
        itl::gmres(A, xs, b, M, iter, 6);
    }
    const csr A1 = make(1, {{0, 0, 3.0}});
    const vec_t b1(1, 6.0);
    const pc_id M1(A1);
    for (const auto& s : all_solvers()) {
        CAPTURE(s.name);
        {   // max_iter = 0: no step, error 1
            vec_t x(6, 0.0);
            iter_t iter(b, 0, 1e-10);
            CHECK(s.solve(A, x, b, M, iter) == 1);
            CHECK(iter.iterations() == 0);
            CHECK(norm(x) == 0.0);
        }
        {   // zero rhs from x = 0: converged at once
            const vec_t z(6, 0.0);
            vec_t x(6, 0.0);
            iter_t iter(z, 10, 1e-10, 1e-300);
            CHECK(s.solve(A, x, z, M, iter) == 0);
            CHECK(iter.iterations() == 0);
            CHECK(norm(x) == 0.0);
        }
        {   // an already converged x0 is returned as it is
            vec_t x = xs;
            iter_t iter(b, 10, 1e-8);
            CHECK(s.solve(A, x, b, M, iter) == 0);
            CHECK(iter.iterations() == 0);
            for (std::size_t i = 0; i < 6; ++i) CHECK(x(i) == xs(i));
        }
        {   // 1x1
            vec_t x(1, 0.0);
            iter_t iter(b1, 10, 1e-12);
            CHECK(s.solve(A1, x, b1, M1, iter) == 0);
            CHECK(std::abs(x(0) - 2.0) < 1e-14);
        }
    }
}

TEST_CASE("relresid() and the atol test", "[itl][iteration][866]") {
    const csr A = tridiagonal(20);
    const vec_t b = ramp(20);
    const pc_id M(A);
    vec_t x(20, 0.0);
    // rtol 0: only atol can stop it.
    iter_t iter(b, 1000, 0.0, 1e-9);
    REQUIRE(itl::cg(A, x, b, M, iter) == 0);
    CHECK(iter.resid() <= 1e-9);
    CHECK(iter.norm_r0() == norm(b));
    CHECK(iter.relresid() == iter.resid() / norm(b));
    CHECK(true_residual(A, x, b) < 1e-8);
    // A zero r0 reports the absolute residual.
    iter_t zero(0.0, 10, 1e-8);
    zero.finished(0.25);
    CHECK(zero.relresid() == 0.25);
}

TEST_CASE("minres with a varying diagonal preconditioner", "[itl][minres][866]") {
    // test_minres.cpp's PC is a constant diagonal 4, a scalar multiple of I.
    entries e;
    const std::size_t n = 40;
    for (std::size_t i = 0; i < n; ++i) {
        e.emplace_back(i, i, 2.0 + double(i));
        if (i + 1 < n) { e.emplace_back(i, i + 1, -1.0); e.emplace_back(i + 1, i, -1.0); }
    }
    const csr A = make(n, e);
    const vec_t b = ramp(n);
    vec_t x(n, 0.0);
    const itl::pc::diagonal<csr> M(A);
    iter_t iter(b, 500, 1e-12);
    REQUIRE(itl::minres(A, x, b, M, iter) == 0);
    CHECK(true_residual(A, x, b) < 1e-10 * norm(b));
}
