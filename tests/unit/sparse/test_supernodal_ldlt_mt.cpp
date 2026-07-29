// Threaded supernodal LDL^T solve (#297 batch 7). The supernodal factor's solve
// routes its unit forward / transpose triangular solves through the
// level-scheduled kernels (level_scheduled_unit_lower_solve /
// _transpose_solve), which are proven bit-identical to the serial
// dense_unit_lower_solve / dense_unit_lower_transpose_solve in batch 6. Here we
// verify the composed supernodal solve is BIT-IDENTICAL to the same solve driven
// by the dense kernels on the identical factor (same L, D, permutation), so the
// parallel path changes nothing but the schedule. The permutation and diagonal
// steps are unchanged serial loops, so exact equality (==) is the correct bar.
//
// Sets MTL5_NUM_THREADS before the pool's first use (the only in-process way to
// exercise the threading -- CI otherwise runs serial). Run under TSan
// (-DMTL5_SANITIZE=thread) to race-check the parallel levels.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/detail/thread_pool.hpp>

using namespace mtl;
using namespace mtl::sparse;

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

// SPD tridiagonal: A(i,i)=4, A(i,i+-1)=-1.
mat::compressed2D<double> make_spd_tridiag(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << 4.0;
        if (i + 1 < n) { ins[i][i + 1] << -1.0; ins[i + 1][i] << -1.0; }
    }
    return A;
}

// 5-point 2D Laplacian on a g x g grid (n = g*g): nontrivial fill / supernodes,
// so the factor L has a genuine multi-level dependency structure.
mat::compressed2D<double> make_laplacian2d(std::size_t g) {
    std::size_t n = g * g;
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    auto id = [g](std::size_t r, std::size_t c) { return r * g + c; };
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            std::size_t i = id(r, c);
            ins[i][i] << 4.0;
            if (r + 1 < g) { ins[i][id(r + 1, c)] << -1.0; ins[id(r + 1, c)][i] << -1.0; }
            if (c + 1 < g) { ins[i][id(r, c + 1)] << -1.0; ins[id(r, c + 1)][i] << -1.0; }
        }
    return A;
}

// Deterministic RHS.
vec::dense_vector<double> make_rhs(std::size_t n) {
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i)
        b(static_cast<int>(i)) = 0.5 + std::sin(0.9 * static_cast<double>(i));
    return b;
}

// Reference solve using the serial dense unit kernels on the SAME factor
// (identical L, D, permutation) that the threaded solve uses.
template <typename Value>
std::vector<double> dense_reference_solve(
    const factorization::supernodal_ldlt_factor<Value>& fac,
    const vec::dense_vector<double>& b)
{
    const std::size_t n = fac.num_rows();
    const auto& L = fac.factorL();
    const auto& D = fac.diagonal();
    const auto& p = fac.symbolic.sperm;

    std::vector<Value> w(n);
    for (std::size_t i = 0; i < n; ++i) w[i] = static_cast<Value>(b(static_cast<int>(p[i])));
    factorization::dense_unit_lower_solve(L, w);
    for (std::size_t i = 0; i < n; ++i) w[i] /= D[i];
    factorization::dense_unit_lower_transpose_solve(L, w);

    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[p[i]] = static_cast<double>(w[i]);
    return x;
}

// Factor A, then assert the threaded solve == the dense-kernel reference,
// bit-for-bit, for the same factor.
void require_supernodal_solve_bit_identical(const mat::compressed2D<double>& A) {
    const std::size_t n = A.num_rows();
    auto b = make_rhs(n);

    auto sym = factorization::supernodal_ldlt_symbolic(A, ordering::amd{});
    auto fac = factorization::supernodal_ldlt_numeric(A, sym);

    const auto ref = dense_reference_solve(fac, b);

    vec::dense_vector<double> x(n);
    fac.solve(x, b);

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x(static_cast<int>(i)) == ref[i]);   // exact equality
}

} // namespace

TEST_CASE("supernodal LDL^T threaded solve == dense-kernel reference",
          "[sparse][supernodal][ldlt][threading][mt]") {
    if (mtl::detail::thread_pool::instance().size() < 2)
        WARN("single-core runner: threading not exercised");

    require_supernodal_solve_bit_identical(make_spd_tridiag(500));
    require_supernodal_solve_bit_identical(make_laplacian2d(40));   // n=1600, real supernodes
    require_supernodal_solve_bit_identical(make_laplacian2d(60));   // n=3600, deeper levels
}

TEST_CASE("supernodal LDL^T threaded solve boundary cases",
          "[sparse][supernodal][ldlt][threading][mt][edge]") {
    require_supernodal_solve_bit_identical(make_spd_tridiag(1));    // 1x1
    require_supernodal_solve_bit_identical(make_spd_tridiag(2));    // 2x2
}

// A factor with `symbolic` installed (n > 0) but no factor must reject solve
// rather than index the empty diagonal.
TEST_CASE("supernodal LDL^T solve rejects a missing factor",
          "[sparse][supernodal][ldlt][edge]") {
    factorization::supernodal_ldlt_factor<double> fac;
    fac.symbolic.n = 3;                       // symbolic set, set_factor never called
    vec::dense_vector<double> x(3), b(3);
    for (int i = 0; i < 3; ++i) b(i) = 1.0;
    REQUIRE_THROWS_AS(fac.solve(x, b), std::logic_error);
}
