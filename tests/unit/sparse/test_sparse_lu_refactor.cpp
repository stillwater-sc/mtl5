// MTL5 -- Tests for sparse_lu_refactor (analyze/factor/refactor, issue #136).
// Refactor reuses a prior factorization's symbolic structure + pivot sequence
// and recomputes only the numeric values -- the fast path for repeated solves of
// a fixed sparsity pattern with changing values (SPICE transient analysis).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

namespace {

// Unsymmetric tridiagonal with a tunable diagonal scale, so we can change values
// while keeping the sparsity pattern fixed.
mat::compressed2D<double> tridiag(std::size_t n, double diag) {
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0)     ins[i][i - 1] << -1.0;
        ins[i][i] << diag;
        if (i + 1 < n) ins[i][i + 1] << -2.0;
    }
    return A;
}

double residual_inf(const mat::compressed2D<double>& A,
                    const vec::dense_vector<double>& x,
                    const vec::dense_vector<double>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double m = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        double ax = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            ax += dat[k] * x(static_cast<int>(ci[k]));
        m = std::max(m, std::abs(ax - b(static_cast<int>(r))));
    }
    return m;
}

} // namespace

TEST_CASE("refactor reproduces factor on the same matrix", "[sparse][lu][refactor]") {
    std::size_t n = 30;
    auto A = tridiag(n, 4.0);
    auto sym = sparse::factorization::sparse_lu_symbolic(A);
    auto num = sparse::factorization::sparse_lu_numeric(A, sym);
    auto re  = sparse::factorization::sparse_lu_refactor(A, num);

    vec::dense_vector<double> b(n, 1.0), x_num(n, 0.0), x_re(n, 0.0);
    num.solve(x_num, b);
    re.solve(x_re, b);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(x_re(static_cast<int>(i)),
                     WithinAbs(x_num(static_cast<int>(i)), 1e-12));
    REQUIRE(residual_inf(A, x_re, b) < 1e-10);
}

TEST_CASE("refactor solves a same-pattern matrix with new values",
          "[sparse][lu][refactor]") {
    std::size_t n = 40;
    auto A1 = tridiag(n, 4.0);
    auto num = sparse::factorization::sparse_lu_numeric(
        A1, sparse::factorization::sparse_lu_symbolic(A1));

    // Same pattern, different diagonal -> refactor must give the correct solve
    // of A2, matching a fresh factorization of A2.
    auto A2 = tridiag(n, 7.5);
    auto re = sparse::factorization::sparse_lu_refactor(A2, num);

    vec::dense_vector<double> b(n), x_re(n, 0.0), x_fresh(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = static_cast<double>(i % 5 + 1);

    re.solve(x_re, b);
    sparse::factorization::sparse_lu_solve(A2, x_fresh, b);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(x_re(static_cast<int>(i)),
                     WithinAbs(x_fresh(static_cast<int>(i)), 1e-9));
    REQUIRE(residual_inf(A2, x_re, b) < 1e-9);
}

TEST_CASE("repeated refactor is stable across many value updates",
          "[sparse][lu][refactor]") {
    std::size_t n = 25;
    auto num = sparse::factorization::sparse_lu_numeric(
        tridiag(n, 4.0), sparse::factorization::sparse_lu_symbolic(tridiag(n, 4.0)));

    vec::dense_vector<double> b(n, 1.0), x(n, 0.0);
    for (int step = 0; step < 5; ++step) {
        auto A = tridiag(n, 4.0 + 0.5 * step);
        num = sparse::factorization::sparse_lu_refactor(A, num);
        num.solve(x, b);
        REQUIRE(residual_inf(A, x, b) < 1e-10);
    }
}
