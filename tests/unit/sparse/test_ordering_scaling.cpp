// MTL5 -- Regression test for AMD/COLAMD complexity (issue #128).
//
// The original amd/colamd were O(n^2) (linear min-degree scan + dense fill
// insertion), which re-introduced quadratic cost wherever they were used. This
// guards the near-linear cs_amd-based implementation: ordering a large banded
// matrix must complete in time consistent with O(nnz), not O(n^2).
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/sparse/util/permutation.hpp>

using namespace mtl;
using namespace mtl::sparse;

namespace {

// Symmetric tridiagonal (nnz ~ 3n): a correct minimum-degree ordering is
// near-linear; an O(n^2) implementation would take far longer.
mat::compressed2D<double> make_tridiag(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << 4.0;
        if (i + 1 < n) { ins[i][i + 1] << -1.0; ins[i + 1][i] << -1.0; }
    }
    return A;
}

} // namespace

TEST_CASE("AMD orders a large matrix in O(flops) time", "[sparse][amd][scaling]") {
    const std::size_t n = 50000;
    auto A = make_tridiag(n);

    auto t0 = std::chrono::steady_clock::now();
    auto perm = ordering::amd{}(A);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    REQUIRE(perm.size() == n);
    REQUIRE(util::is_valid_permutation(perm));
    REQUIRE(elapsed_s < 5.0);   // ~tens of ms expected; guards O(n^2) regression
}

TEST_CASE("COLAMD orders a large matrix in O(flops) time",
          "[sparse][colamd][scaling]") {
    const std::size_t n = 50000;
    auto A = make_tridiag(n);

    auto t0 = std::chrono::steady_clock::now();
    auto perm = ordering::colamd{}(A);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    REQUIRE(perm.size() == n);
    REQUIRE(util::is_valid_permutation(perm));
    REQUIRE(elapsed_s < 5.0);
}
