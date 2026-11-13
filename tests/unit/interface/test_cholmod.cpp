// MTL5 -- Tests for the CHOLMOD sparse Cholesky binding (#774, #810)
//
// These tests only do real work when MTL5_HAS_CHOLMOD is defined (CMake option
// MTL5_WITH_SUITESPARSE_CHOLMOD=ON with SuiteSparse installed). Otherwise a
// placeholder confirms the test infrastructure compiles and links.
#include <catch2/catch_test_macros.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

#ifdef MTL5_HAS_CHOLMOD

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <mtl/interface/cholmod.hpp>

using mtl::interface::cholmod_solver;

namespace {

using csr = mat::compressed2D<double>;

/// Normwise backward error ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf).
double backward_error(const csr& A, const vec::dense_vector<double>& x,
                      const vec::dense_vector<double>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& v  = A.ref_data();
    double r = 0, an = 0, xn = 0, bn = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double ax = 0, row = 0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) {
            ax += v[k] * x(ci[k]);
            row += std::abs(v[k]);
        }
        r = std::max(r, std::abs(b(i) - ax));
        an = std::max(an, row);
        xn = std::max(xn, std::abs(x(i)));
        bn = std::max(bn, std::abs(b(i)));
    }
    return r / (an * xn + bn);
}

/// The 5-point Laplacian on a k x k grid, with `shift` added to the diagonal.
/// Its eigenvalues lie in (0, 8), so shift = -5 makes it indefinite.
csr laplacian(std::size_t k, double shift = 0.0) {
    const std::size_t n = k * k;
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t r = i * k + j;
                ins[r][r] << 4.0 + shift;
                if (i > 0)     ins[r][r - k] << -1.0;
                if (i + 1 < k) ins[r][r + k] << -1.0;
                if (j > 0)     ins[r][r - 1] << -1.0;
                if (j + 1 < k) ins[r][r + 1] << -1.0;
            }
    }
    return A;
}

vec::dense_vector<double> rhs(std::size_t n) {
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + double(i % 7);
    return b;
}

constexpr double tol = 1e-13;

} // namespace

TEST_CASE("cholmod: small SPD system, exact solution", "[interface][cholmod]") {
    // A = [4 1; 1 3], b = [6; 7] => x = [1; 2]
    csr A(2, 2);
    {
        mat::inserter<csr> ins(A);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0;
    }
    vec::dense_vector<double> b = {6.0, 7.0}, x(2, 0.0);
    interface::cholmod_solve(A, x, b);
    REQUIRE(std::abs(x(0) - 1.0) < tol);
    REQUIRE(std::abs(x(1) - 2.0) < tol);
}

TEST_CASE("cholmod: simplicial and supernodal both solve a 2D Laplacian", "[interface][cholmod]") {
    const csr A = laplacian(40);
    const auto b = rhs(A.num_rows());
    for (int method : {CHOLMOD_SIMPLICIAL, CHOLMOD_SUPERNODAL}) {
        cholmod_solver s;
        s.common().supernodal = method;
        s.analyze(A);
        s.factorize();
        REQUIRE(s.is_supernodal() == (method == CHOLMOD_SUPERNODAL));
        REQUIRE(s.factor_nnz() >= double(A.num_rows()));
        vec::dense_vector<double> x(A.num_rows(), 0.0);
        s.solve(x, b);
        REQUIRE(backward_error(A, x, b) < tol);
    }
}

TEST_CASE("cholmod: an indefinite matrix throws, simplicial and supernodal", "[interface][cholmod]") {
    // CHOLMOD's simplicial default is L D L^T, which factors this matrix with
    // status OK; the binding asks for L L^T, so both paths must refuse it.
    const csr A = laplacian(20, -5.0);
    for (int method : {CHOLMOD_SIMPLICIAL, CHOLMOD_SUPERNODAL}) {
        cholmod_solver s;
        s.common().supernodal = method;
        s.analyze(A);
        REQUIRE_THROWS_AS(s.factorize(), std::runtime_error);
        REQUIRE_FALSE(s.factored());
        vec::dense_vector<double> x(A.num_rows(), 0.0);
        REQUIRE_THROWS_AS(s.solve(x, rhs(A.num_rows())), std::logic_error);
    }
    // The one-call constructor throws too, and frees what it had allocated
    // (the sanitizer builds would report a leak).
    REQUIRE_THROWS_AS(cholmod_solver(A), std::runtime_error);
    vec::dense_vector<double> x(A.num_rows(), 0.0);
    REQUIRE_THROWS_AS(interface::cholmod_solve(A, x, rhs(A.num_rows())), std::runtime_error);
}

TEST_CASE("cholmod: refactorize with new values, same pattern", "[interface][cholmod]") {
    const csr A = laplacian(16);
    const csr A2 = laplacian(16, 1.5);   // same pattern, a different SPD matrix
    const auto b = rhs(A.num_rows());
    cholmod_solver s(A);
    s.factorize(A2);
    vec::dense_vector<double> x(A.num_rows(), 0.0);
    s.solve(x, b);
    REQUIRE(backward_error(A2, x, b) < tol);

    // A different pattern is refused, and the factor of A2 is still usable.
    csr wider(A.num_rows(), A.num_rows());
    {
        mat::inserter<csr> ins(wider);
        for (std::size_t r = 0; r < A.num_rows(); ++r) ins[r][r] << 1.0;
    }
    REQUIRE_THROWS_AS(s.factorize(wider), std::invalid_argument);
    REQUIRE(s.factored());
    s.solve(x, b);
    REQUIRE(backward_error(A2, x, b) < tol);
}

TEST_CASE("cholmod: phases out of order are rejected", "[interface][cholmod]") {
    cholmod_solver s;
    REQUIRE_FALSE(s.analyzed());
    REQUIRE_THROWS_AS(s.factorize(), std::logic_error);
    vec::dense_vector<double> x(4, 0.0), b(4, 1.0);
    REQUIRE_THROWS_AS(s.solve(x, b), std::logic_error);
}

TEST_CASE("cholmod: move construction and move assignment", "[interface][cholmod]") {
    static_assert(std::is_nothrow_move_constructible_v<cholmod_solver>);
    static_assert(std::is_nothrow_move_assignable_v<cholmod_solver>);
    const csr A = laplacian(12);
    const auto b = rhs(A.num_rows());
    vec::dense_vector<double> x(A.num_rows(), 0.0);

    cholmod_solver a(A);
    cholmod_solver moved(std::move(a));     // a's workspace moves; no double free
    moved.solve(x, b);
    REQUIRE(backward_error(A, x, b) < tol);
    REQUIRE_THROWS_AS(a.solve(x, b), std::logic_error);   // NOLINT(bugprone-use-after-move)
    REQUIRE_THROWS_AS(a.common(), std::logic_error);      // NOLINT(bugprone-use-after-move)

    const csr B = laplacian(9, 2.0);
    cholmod_solver c(B);
    c = std::move(moved);                   // frees B's factor, takes A's
    std::fill(x.begin(), x.end(), 0.0);
    c.solve(x, b);
    REQUIRE(backward_error(A, x, b) < tol);
    REQUIRE(c.num_rows() == A.num_rows());

    a = cholmod_solver(B);                  // a moved-from solver can be assigned again
    vec::dense_vector<double> y(B.num_rows(), 0.0);
    const auto bb = rhs(B.num_rows());
    a.solve(y, bb);
    REQUIRE(backward_error(B, y, bb) < tol);
}

TEST_CASE("cholmod: argument checks", "[interface][cholmod]") {
    REQUIRE_THROWS_AS(cholmod_solver(csr(3, 4)), std::invalid_argument);
    cholmod_solver s(laplacian(4));
    vec::dense_vector<double> x(15, 0.0), b(16, 1.0);
    REQUIRE_THROWS_AS(s.solve(x, b), std::invalid_argument);
}

TEST_CASE("cholmod: n = 1 and n = 0", "[interface][cholmod]") {
    csr one(1, 1);
    { mat::inserter<csr> ins(one); ins[0][0] << 4.0; }
    vec::dense_vector<double> x(1, 0.0), b(1, 2.0);
    interface::cholmod_solve(one, x, b);
    REQUIRE(std::abs(x(0) - 0.5) < tol);

    csr empty(0, 0);
    vec::dense_vector<double> x0(0), b0(0);
    cholmod_solver s(empty);
    s.solve(x0, b0);
    REQUIRE(s.num_rows() == 0);
}

#else

TEST_CASE("cholmod: placeholder (CHOLMOD not available)", "[interface][cholmod]") {
    REQUIRE(true);
}

#endif // MTL5_HAS_CHOLMOD
