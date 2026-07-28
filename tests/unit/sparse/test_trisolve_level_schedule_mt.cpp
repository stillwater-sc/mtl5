// Level-scheduled sparse lower-triangular forward solve (#297 batch 3, pilot).
// build_lower_solve_schedule + level_scheduled_lower_solve must be BIT-IDENTICAL
// to the serial dense_lower_solve: the row-gather accumulates each x[i] in the
// same increasing-column order the CSC scatter uses. We assert exact equality
// (==) against dense_lower_solve, on structures that span the level spectrum
// (all-independent -> fully sequential) including a wide level that actually
// splits across the pool. Run under TSan (-DMTL5_SANITIZE=thread) for races.
//
// Sets MTL5_NUM_THREADS before the pool's first use (the only in-process way to
// exercise the threading -- CI otherwise runs serial).
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/factorization/level_schedule.hpp>
#include <mtl/detail/thread_pool.hpp>

using namespace mtl::sparse;
using mtl::sparse::util::csc_matrix;

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

// Block-diagonal lower triangular: B dense m x m lower-triangular blocks.
// Row (b*m + r) depends on rows b*m .. b*m+r-1, so it sits at level r; each level
// r holds B mutually independent rows -> genuinely parallel levels.
csc_matrix<double> make_block_diag_lower(std::size_t B, std::size_t m, std::uint64_t seed) {
    csc_matrix<double> L;
    const std::size_t n = B * m;
    L.nrows = n; L.ncols = n;
    L.col_ptr.assign(n + 1, 0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t cc = 0; cc < m; ++cc) {
            const std::size_t j = b * m + cc;
            L.col_ptr[j + 1] = L.col_ptr[j] + (m - cc);   // diagonal + entries below in block
        }
    L.row_ind.resize(L.col_ptr[n]);
    L.values.resize(L.col_ptr[n]);
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t cc = 0; cc < m; ++cc) {
            const std::size_t j = b * m + cc;
            std::size_t p = L.col_ptr[j];
            L.row_ind[p] = j; L.values[p] = double(m) + 1.0; ++p;   // diagonal first
            for (std::size_t r = cc + 1; r < m; ++r) { L.row_ind[p] = b * m + r; L.values[p] = d(rng); ++p; }
        }
    return L;
}

// Arrow lower triangular: dense first column (rows 0..n-1), rest diagonal.
// Rows 1..n-1 all depend only on row 0 -> one wide level of n-1 independent
// rows, which splits across the pool.
csc_matrix<double> make_arrow_lower(std::size_t n, std::uint64_t seed) {
    csc_matrix<double> L;
    L.nrows = n; L.ncols = n;
    L.col_ptr.assign(n + 1, 0);
    L.col_ptr[1] = n;                       // column 0 is dense
    for (std::size_t j = 1; j < n; ++j) L.col_ptr[j + 1] = L.col_ptr[j] + 1;   // diagonal only
    L.row_ind.resize(L.col_ptr[n]);
    L.values.resize(L.col_ptr[n]);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    std::size_t p = 0;
    L.row_ind[p] = 0; L.values[p] = double(n); ++p;                 // diagonal of col 0
    for (std::size_t i = 1; i < n; ++i) { L.row_ind[p] = i; L.values[p] = d(rng); ++p; }
    for (std::size_t j = 1; j < n; ++j) { L.row_ind[p] = j; L.values[p] = double(n); ++p; }
    return L;
}

std::vector<double> random_rhs(std::size_t n, std::uint64_t seed) {
    std::vector<double> b(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (auto& v : b) v = d(rng);
    return b;
}

// Assert level_scheduled_lower_solve == dense_lower_solve, bit-for-bit.
void require_bit_identical(const csc_matrix<double>& L, std::uint64_t rhs_seed) {
    const std::size_t n = L.ncols;
    auto ref = random_rhs(n, rhs_seed);
    auto got = ref;
    factorization::dense_lower_solve(L, ref);                        // serial reference
    auto sched = factorization::build_lower_solve_schedule(L);
    factorization::level_scheduled_lower_solve(L, sched, got);       // level-scheduled
    for (std::size_t i = 0; i < n; ++i) REQUIRE(got[i] == ref[i]);
}

} // namespace

TEST_CASE("level_scheduled_lower_solve == dense_lower_solve (block-diagonal)",
          "[sparse][trisolve][threading][mt]") {
    require_bit_identical(make_block_diag_lower(64, 8, 11), 101);    // narrow levels
    require_bit_identical(make_block_diag_lower(1, 200, 22), 202);   // one dense block: fully sequential
}

TEST_CASE("level_scheduled_lower_solve == dense_lower_solve (wide level splits)",
          "[sparse][trisolve][threading][mt]") {
    if (mtl::detail::thread_pool::instance().size() < 2) WARN("single-core runner: threading not exercised");
    // n large enough that the single wide level exceeds the parallel_for grain
    // and is partitioned across the pool.
    require_bit_identical(make_arrow_lower(80000, 33), 303);
}

TEST_CASE("level_scheduled_lower_solve boundary cases",
          "[sparse][trisolve][threading][mt][edge]") {
    // Empty 0x0: build (zero levels) and solve must be no-ops.
    {
        csc_matrix<double> L; L.nrows = 0; L.ncols = 0; L.col_ptr = {0};
        auto sched = factorization::build_lower_solve_schedule(L);
        REQUIRE(sched.n == 0);
        std::vector<double> x;   // empty
        factorization::level_scheduled_lower_solve(L, sched, x);   // no-op
        REQUIRE(x.empty());
    }
    // 1x1
    {
        csc_matrix<double> L; L.nrows = 1; L.ncols = 1;
        L.col_ptr = {0, 1}; L.row_ind = {0}; L.values = {3.0};
        require_bit_identical(L, 1);
    }
    // Purely diagonal: every row is level 0 (maximally parallel).
    {
        const std::size_t n = 50000;
        csc_matrix<double> L; L.nrows = n; L.ncols = n;
        L.col_ptr.resize(n + 1);
        for (std::size_t j = 0; j <= n; ++j) L.col_ptr[j] = j;
        L.row_ind.resize(n); L.values.resize(n);
        for (std::size_t j = 0; j < n; ++j) { L.row_ind[j] = j; L.values[j] = 2.0 + double(j % 7); }
        require_bit_identical(L, 7);
    }
}

// -- transpose solve (#297 batch 4) --------------------------------------

namespace {

// Lower triangular whose last row is dense: column col (col < n-1) has its
// diagonal plus one entry in row n-1. In the TRANSPOSE solve every such column
// gathers from x[n-1], so columns 0..n-2 form one wide level that splits across
// the pool (the arrow's transpose would instead collapse to a single column).
csc_matrix<double> make_last_row_dense_lower(std::size_t n, std::uint64_t seed) {
    csc_matrix<double> L;
    L.nrows = n; L.ncols = n;
    L.col_ptr.assign(n + 1, 0);
    for (std::size_t col = 0; col + 1 < n; ++col) L.col_ptr[col + 1] = L.col_ptr[col] + 2;  // diag + row n-1
    if (n) L.col_ptr[n] = L.col_ptr[n - 1] + 1;                                              // last col: diag only
    L.row_ind.resize(L.col_ptr[n]);
    L.values.resize(L.col_ptr[n]);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    std::size_t p = 0;
    for (std::size_t col = 0; col + 1 < n; ++col) {
        L.row_ind[p] = col;   L.values[p] = double(n); ++p;      // diagonal
        L.row_ind[p] = n - 1; L.values[p] = d(rng);   ++p;       // entry in the dense last row
    }
    if (n) { L.row_ind[p] = n - 1; L.values[p] = double(n); }     // last diagonal
    return L;
}

// Assert level_scheduled_lower_transpose_solve == dense_lower_transpose_solve.
void require_transpose_bit_identical(const csc_matrix<double>& L, std::uint64_t rhs_seed) {
    const std::size_t n = L.ncols;
    auto ref = random_rhs(n, rhs_seed);
    auto got = ref;
    factorization::dense_lower_transpose_solve(L, ref);                    // serial reference
    auto sched = factorization::build_lower_transpose_solve_schedule(L);
    factorization::level_scheduled_lower_transpose_solve(L, sched, got);   // level-scheduled
    for (std::size_t i = 0; i < n; ++i) REQUIRE(got[i] == ref[i]);
}

} // namespace

TEST_CASE("level_scheduled_lower_transpose_solve == dense_lower_transpose_solve (block-diagonal)",
          "[sparse][trisolve][transpose][threading][mt]") {
    require_transpose_bit_identical(make_block_diag_lower(64, 8, 11), 401);
    require_transpose_bit_identical(make_block_diag_lower(1, 200, 22), 402);   // one dense block: sequential
}

TEST_CASE("level_scheduled_lower_transpose_solve == dense_lower_transpose_solve (wide level splits)",
          "[sparse][trisolve][transpose][threading][mt]") {
    if (mtl::detail::thread_pool::instance().size() < 2) WARN("single-core runner: threading not exercised");
    require_transpose_bit_identical(make_last_row_dense_lower(80000, 33), 403);   // level 1 splits across the pool
}

TEST_CASE("level_scheduled_lower_transpose_solve boundary cases",
          "[sparse][trisolve][transpose][threading][mt][edge]") {
    // Empty 0x0
    {
        csc_matrix<double> L; L.nrows = 0; L.ncols = 0; L.col_ptr = {0};
        auto sched = factorization::build_lower_transpose_solve_schedule(L);
        REQUIRE(sched.n == 0);
        std::vector<double> x;
        factorization::level_scheduled_lower_transpose_solve(L, sched, x);   // no-op
        REQUIRE(x.empty());
    }
    // 1x1
    {
        csc_matrix<double> L; L.nrows = 1; L.ncols = 1;
        L.col_ptr = {0, 1}; L.row_ind = {0}; L.values = {3.0};
        require_transpose_bit_identical(L, 1);
    }
    // Diagonal: every column is level 0 (maximally parallel).
    {
        const std::size_t n = 50000;
        csc_matrix<double> L; L.nrows = n; L.ncols = n;
        L.col_ptr.resize(n + 1);
        for (std::size_t j = 0; j <= n; ++j) L.col_ptr[j] = j;
        L.row_ind.resize(n); L.values.resize(n);
        for (std::size_t j = 0; j < n; ++j) { L.row_ind[j] = j; L.values[j] = 2.0 + double(j % 7); }
        require_transpose_bit_identical(L, 7);
    }
}
