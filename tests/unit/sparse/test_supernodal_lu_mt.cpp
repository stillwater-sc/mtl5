// Threaded supernodal LU solve (#297 batch 8). The supernodal LU factor's solve
// routes its lower / upper triangular solves through the level-scheduled kernels
// (level_scheduled_lower_solve / level_scheduled_upper_solve), which are proven
// bit-identical to the serial dense_lower_solve / dense_upper_solve in batches
// 3/5. Here we verify the composed supernodal solve is BIT-IDENTICAL to the same
// solve driven by the dense kernels on the identical factor (same L, U, row and
// column permutations, row scaling), so the parallel path changes nothing but
// the schedule. The permutation / scaling steps are unchanged serial loops, so
// exact equality (==) is the correct bar.
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
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
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

// Unsymmetric, diagonally dominant banded matrix (nonsingular, LU-stable):
// diag 8, sub-diagonals -1 and -0.5, super-diagonal -2. |off| = 3.5 < 8.
mat::compressed2D<double> make_unsym_banded(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << 8.0;
        if (i >= 1)     ins[i][i - 1] << -1.0;
        if (i >= 2)     ins[i][i - 2] << -0.5;
        if (i + 1 < n)  ins[i][i + 1] << -2.0;
    }
    return A;
}

// Unsymmetric 5-point-style stencil on a g x g grid (n = g*g): asymmetric
// neighbor weights so L != U^T, with genuine fill / multi-level structure.
// diag 8; up -1, down -2, left -1, right -2 => |off| <= 6 < 8 (dominant).
mat::compressed2D<double> make_unsym_grid(std::size_t g) {
    std::size_t n = g * g;
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    auto id = [g](std::size_t r, std::size_t c) { return r * g + c; };
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            std::size_t i = id(r, c);
            ins[i][i] << 8.0;
            if (r >= 1)     ins[i][id(r - 1, c)] << -1.0;
            if (r + 1 < g)  ins[i][id(r + 1, c)] << -2.0;
            if (c >= 1)     ins[i][id(r, c - 1)] << -1.0;
            if (c + 1 < g)  ins[i][id(r, c + 1)] << -2.0;
        }
    return A;
}

vec::dense_vector<double> make_rhs(std::size_t n) {
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i)
        b(static_cast<int>(i)) = 0.5 + std::sin(0.9 * static_cast<double>(i));
    return b;
}

// Reference solve using the serial dense kernels on the SAME factor (identical
// L, U, permutations, row scaling) that the threaded solve uses.
template <typename Value>
std::vector<double> dense_reference_solve(
    const factorization::supernodal_lu_factor<Value>& fac,
    const vec::dense_vector<double>& b)
{
    const std::size_t n = fac.num_rows();
    const auto& L  = fac.factorL();
    const auto& U  = fac.factorU();
    const auto& rp = fac.row_perm;
    const auto& rs = fac.row_scale;
    const auto& cp = fac.symbolic.col_perm;

    std::vector<Value> w(n);
    const bool scaled = !rs.empty();
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t orow = rp[i];
        w[i] = static_cast<Value>(b(static_cast<int>(orow))) * (scaled ? rs[orow] : Value{1});
    }
    factorization::dense_lower_solve(L, w);
    factorization::dense_upper_solve(U, w);

    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[cp[i]] = static_cast<double>(w[i]);
    return x;
}

// Factor A, then assert the threaded solve == the dense-kernel reference,
// bit-for-bit, for the same factor.
void require_supernodal_lu_solve_bit_identical(const mat::compressed2D<double>& A) {
    const std::size_t n = A.num_rows();
    auto b = make_rhs(n);

    auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});
    auto fac = factorization::supernodal_lu_numeric(A, sym);

    const auto ref = dense_reference_solve(fac, b);

    vec::dense_vector<double> x(n);
    fac.solve(x, b);

    // ORDER: identical accumulation order to the serial dense kernels. A
    // source-level claim, observable only with FP contraction pinned -- the
    // target pins it (tests/unit/CMakeLists.txt, #381).
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x(static_cast<int>(i)) == ref[i]);   // exact equality

    // DETERMINISM: the threaded solve is reproducible run to run. Same function,
    // same expression tree, so this holds under any optimization flags, and it
    // is what guards against a racing or order-dependent reduction.
    for (int rep = 0; rep < 3; ++rep) {
        vec::dense_vector<double> again(n);
        fac.solve(again, b);
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(again(static_cast<int>(i)) == x(static_cast<int>(i)));
    }
}

} // namespace

TEST_CASE("supernodal LU threaded solve == dense-kernel reference",
          "[sparse][supernodal][lu][threading][mt]") {
    if (mtl::detail::thread_pool::instance().size() < 2)
        WARN("single-core runner: threading not exercised");

    require_supernodal_lu_solve_bit_identical(make_unsym_banded(500));
    require_supernodal_lu_solve_bit_identical(make_unsym_grid(40));   // n=1600, real fill
    require_supernodal_lu_solve_bit_identical(make_unsym_grid(60));   // n=3600, deeper levels
}

TEST_CASE("supernodal LU threaded solve boundary cases",
          "[sparse][supernodal][lu][threading][mt][edge]") {
    require_supernodal_lu_solve_bit_identical(make_unsym_banded(1));  // 1x1
    require_supernodal_lu_solve_bit_identical(make_unsym_banded(2));  // 2x2
}

// A factor with `symbolic` installed (n > 0) but no factor must reject solve
// rather than read empty factor arrays (#307 pattern).
TEST_CASE("supernodal LU solve rejects a missing factor",
          "[sparse][supernodal][lu][edge]") {
    factorization::supernodal_lu_factor<double> fac;
    fac.symbolic.n = 3;                       // symbolic set, set_factor never called
    fac.row_perm = {0, 1, 2};                 // present but irrelevant: guard fires first
    vec::dense_vector<double> x(3), b(3);
    for (int i = 0; i < 3; ++i) b(i) = 1.0;
    REQUIRE_THROWS_AS(fac.solve(x, b), std::logic_error);
}
