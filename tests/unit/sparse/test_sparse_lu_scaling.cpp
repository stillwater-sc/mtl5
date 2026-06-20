// MTL5 -- Regression tests for native LU / KLU complexity (issue #127).
//
// The original sparse_lu_numeric was O(n^2) in time (dense left-looking loops)
// and the BTF matching was O(n^2) (per-row reset), so real matrices took hours
// and tens of GB. These tests guard the fix: factorization fill stays O(n) on a
// banded matrix, and a large sparse solve completes in time consistent with
// O(flops), not O(n^2).
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>

using namespace mtl;

namespace {

// Unsymmetric tridiagonal: A(i,i)=4, A(i,i-1)=-1, A(i,i+1)=-2 (irreducible ->
// one BTF block, so this exercises the per-block factorization at full size).
mat::compressed2D<double> make_unsym_tridiag(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0)     ins[i][i - 1] << -1.0;
        ins[i][i] << 4.0;
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

TEST_CASE("sparse LU fill stays linear on a banded matrix", "[sparse][lu][scaling]") {
    // Tridiagonal LU (natural ordering) has bidiagonal L and U: nnz(L)+nnz(U)
    // is O(n). A regression to dense behavior would blow this far past the
    // linear bound.
    for (std::size_t n : {1000u, 8000u}) {
        auto A = make_unsym_tridiag(n);
        auto sym = sparse::factorization::sparse_lu_symbolic(A);   // natural
        auto num = sparse::factorization::sparse_lu_numeric(A, sym);
        std::size_t fill = num.L.nnz() + num.U.nnz();
        REQUIRE(fill < 8 * n);   // ~4n expected; generous linear bound
    }
}

TEST_CASE("native KLU solves a large sparse system in O(flops) time",
          "[sparse][klu][scaling]") {
    // At n=40000 a correct O(flops) solve is milliseconds; the previous O(n^2)
    // implementation would take well over a minute. The 5s bound has a ~100x
    // margin over the expected runtime, so it distinguishes O(n) from O(n^2)
    // without being sensitive to CI machine speed.
    const std::size_t n = 40000;
    auto A = make_unsym_tridiag(n);
    vec::dense_vector<double> b(n, 1.0), x(n, 0.0);

    auto t0 = std::chrono::steady_clock::now();
    sparse::factorization::native_klu_solve(A, x, b);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    REQUIRE(residual_inf(A, x, b) < 1e-9);
    REQUIRE(elapsed_s < 5.0);   // guards against an O(n^2) regression
}
