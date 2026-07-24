#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/backward_error.hpp>
#include <mtl/operation/lu_iterative_refine.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

namespace {
    // A well-conditioned, diagonally dominant system with exact solution all-ones.
    // Integer entries -> float LU is already exact (residual 0). Used where the
    // test only needs a solvable system.
    template <typename T>
    void build(mat::dense2D<T>& A, vec::dense_vector<T>& b, std::size_t n) {
        A = mat::dense2D<T>(n, n);
        b = vec::dense_vector<T>(n, T(0));
        for (std::size_t i = 0; i < n; ++i) {
            T s(0);
            for (std::size_t j = 0; j < n; ++j) {
                T v = (i == j) ? T(4) : ((i + 1 == j || j + 1 == i) ? T(-1) : T(0));
                A(i, j) = v;
                s = s + v;
            }
            b[i] = s;              // row sum -> exact solution is all ones
        }
    }

    // A diagonally dominant system with NON-float-exact entries, so a float
    // factorization carries real rounding error (nonzero residual) that a
    // higher-precision residual can then refine away.
    template <typename T>
    void build_hard(mat::dense2D<T>& A, vec::dense_vector<T>& b, std::size_t n) {
        A = mat::dense2D<T>(n, n);
        b = vec::dense_vector<T>(n, T(1));         // arbitrary RHS
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A(i, j) = (i == j) ? T(double(n)) : T(1.0 / double(i + j + 2));  // Hilbert-ish off-diagonal
    }

    // plain (unrefined) working-precision LU residual ||b - A x||_inf, in double
    template <typename T>
    double plain_lu_residual(std::size_t n, bool hard = false) {
        mat::dense2D<T> A; vec::dense_vector<T> b;
        if (hard) build_hard(A, b, n); else build(A, b, n);
        mat::dense2D<T> LU(A);
        std::vector<typename mat::dense2D<T>::size_type> piv;
        lu_factor(LU, piv);
        vec::dense_vector<T> x(n, T(0));
        lu_solve(LU, piv, x, b);
        double r = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double ax = 0.0;
            for (std::size_t j = 0; j < n; ++j) ax += double(A(i, j)) * double(x[j]);
            r = std::max(r, std::abs(double(b[i]) - ax));
        }
        return r;
    }
}

TEST_CASE("lu_iterative_refine same-precision solves exactly", "[operation][ir]") {
    mat::dense2D<double> A; vec::dense_vector<double> b;
    build(A, b, 8);
    vec::dense_vector<double> x;
    auto res = lu_iterative_refine<double>(A, b, x);
    REQUIRE(x.size() == 8);
    for (std::size_t i = 0; i < 8; ++i) REQUIRE_THAT(double(x[i]), WithinAbs(1.0, 1e-10));
    REQUIRE(res.rel_residual <= 1e-12);
}

TEST_CASE("lu_iterative_refine recovers accuracy with a higher-precision residual", "[operation][ir]") {
    // Working = float factorization, Residual = double residual, on a system
    // whose non-float-exact entries make the plain float solve inexact.
    mat::dense2D<double> A; vec::dense_vector<double> b;
    build_hard(A, b, 12);
    vec::dense_vector<double> x;
    lu_refine_options opt; opt.max_iter = 30; opt.rel_tol = 1e-13;
    auto res = lu_iterative_refine<float>(A, b, x, opt);

    const double plain = plain_lu_residual<float>(12, /*hard=*/true);
    REQUIRE(plain > 1e-8);                          // float alone leaves a real residual
    REQUIRE(res.rel_residual < plain);             // refinement improves on it...
    REQUIRE(res.rel_residual <= 1e-12);            // ...down to the double residual floor
}

TEST_CASE("lu_iterative_refine tolerance and convergence flag", "[operation][ir]") {
    mat::dense2D<double> A; vec::dense_vector<double> b;
    build(A, b, 6);
    vec::dense_vector<double> x;
    lu_refine_options opt; opt.rel_tol = 1e-8; opt.max_iter = 30;
    auto res = lu_iterative_refine<double>(A, b, x, opt);
    REQUIRE(res.converged);
    REQUIRE(res.rel_residual <= 1e-8);
}

TEST_CASE("lu_iterative_refine best-iterate never worse than the plain solve", "[operation][ir]") {
    mat::dense2D<double> A; vec::dense_vector<double> b;
    build(A, b, 10);
    vec::dense_vector<double> x;
    lu_refine_options opt; opt.max_iter = 20;   // no rel_tol: runs the budget, keeps best
    auto res = lu_iterative_refine<float>(A, b, x, opt);
    // Compare like-for-like: res.rel_residual is best_rn/||b||_inf, whereas
    // plain_lu_residual is the ABSOLUTE ||b - A x||_inf. build(10)'s boundary rows
    // give ||b||_inf = 3, so recover the absolute best residual before comparing;
    // the refined best-iterate must not be worse than the unrefined float solve.
    double bnorm = 0.0;
    for (std::size_t i = 0; i < 10; ++i) bnorm = std::max(bnorm, std::abs(double(b[i])));
    REQUIRE(res.rel_residual * bnorm <= plain_lu_residual<float>(10) + 1e-12);
}

TEST_CASE("normwise_backward_error is tiny for the exact solution", "[operation][ir][nbe]") {
    mat::dense2D<double> A; vec::dense_vector<double> b;
    build(A, b, 8);
    vec::dense_vector<double> xexact(8, 1.0);
    REQUIRE(normwise_backward_error(A, xexact, b) <= 1e-15);

    vec::dense_vector<double> xbad(8, 0.0);       // a wrong solution -> O(1) backward error
    REQUIRE(normwise_backward_error(A, xbad, b) > 1e-3);
}

TEST_CASE("lu_iterative_refine handles 1x1 and empty systems", "[operation][ir][edge]") {
    SECTION("1x1") {
        mat::dense2D<double> A; vec::dense_vector<double> b;
        build(A, b, 1);                              // A = [4], b = [4] -> x = [1]
        vec::dense_vector<double> x;
        auto res = lu_iterative_refine<double>(A, b, x);
        REQUIRE(x.size() == 1);
        REQUIRE_THAT(double(x[0]), WithinAbs(1.0, 1e-12));
        REQUIRE(res.rel_residual <= 1e-12);
    }
    SECTION("empty") {
        mat::dense2D<double> A(0, 0);
        vec::dense_vector<double> b(0), x;
        auto res = lu_iterative_refine<double>(A, b, x);
        REQUIRE(x.size() == 0);
        REQUIRE(res.rel_residual == 0.0);            // no equations -> nothing to refine
    }
}

TEST_CASE("normwise_backward_error rejects ill-formed inputs", "[operation][ir][nbe][edge]") {
    // 1x1 sanity: exact solution has a tiny backward error.
    mat::dense2D<double> A1(1, 1); A1(0, 0) = 2.0;
    vec::dense_vector<double> b1(1, 4.0), x1(1, 2.0);   // 2*2 == 4
    REQUIRE(normwise_backward_error(A1, x1, b1) <= 1e-15);

    // Non-square A must throw.
    mat::dense2D<double> R(2, 3);
    vec::dense_vector<double> v2(2, 1.0);
    REQUIRE_THROWS_AS(normwise_backward_error(R, v2, v2), std::invalid_argument);

    // Square A but mismatched x / b lengths must throw.
    mat::dense2D<double> S(2, 2);
    vec::dense_vector<double> good2(2, 1.0), short1(1, 1.0);
    REQUIRE_THROWS_AS(normwise_backward_error(S, short1, good2), std::invalid_argument);
    REQUIRE_THROWS_AS(normwise_backward_error(S, good2, short1), std::invalid_argument);
}
