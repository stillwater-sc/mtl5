// Threaded sparse CRS matvec (#446 follow-up).
//
// `mat::operator*(compressed2D, dense_vector)` has threaded its row loop for any
// value type since #221, but `mult(compressed2D, x, y)` -- the accumulator-aware
// entry point, and the one a mixed-precision sweep calls -- ran the identical
// traversal serially. The two now agree.
//
// Same two claims as the dense threading tests, and the same reason they must be
// separate: bit-identity alone is satisfied by a kernel that quietly stayed
// serial, which is the bug being fixed.
//
//   1. BIT-IDENTITY against the serial `detail::mult_sparse_crs`, asserted with
//      `==`. A row band owns y(r) outright and reads row r's CSR slice, so the
//      partition cannot change a value or a summation order.
//   2. THE ROWS ACTUALLY SPLIT -- the stand-in scalar's multiply records the
//      executing thread.
//
// The TRANSPOSED kernel is checked too, for the opposite property: it scatters
// into y and must NOT be threaded, so its result has to stay correct while the
// pool is live. That test would fail loudly if someone parallelized that loop
// without privatizing the accumulators.
//
// Sets MTL5_NUM_THREADS before the pool's first use. Worth running under TSan
// (-DMTL5_SANITIZE=thread) for races on the disjoint output writes and on the
// concurrent gather from x.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <algorithm>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/mat/view/transposed_view.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>

using namespace mtl;

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

std::mutex g_tid_mtx;
std::unordered_set<std::size_t> g_tids;
std::atomic<bool> g_record{false};

void note_thread() {
    if (!g_record.load(std::memory_order_relaxed)) return;
    const std::size_t h = std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::lock_guard<std::mutex> lk(g_tid_mtx);
    g_tids.insert(h);
}

/// Stand-in for a Universal number type: a class scalar with the field
/// operations MTL5 needs and nothing a hardware kernel can use. Conversion to
/// double is explicit, so no mixed expression can silently leave the type.
struct emul {
    double x = 0.0;
    emul() = default;
    emul(double d) : x(d) {}
    explicit operator double() const { return x; }
    emul& operator+=(emul o) { x += o.x; return *this; }
};

emul operator+(emul a, emul b) { return emul{a.x + b.x}; }
emul operator-(emul a, emul b) { return emul{a.x - b.x}; }
emul operator-(emul a)         { return emul{-a.x}; }
emul operator/(emul a, emul b) { return emul{a.x / b.x}; }
emul operator*(emul a, emul b) { note_thread(); return emul{a.x * b.x}; }
bool operator==(emul a, emul b) { return a.x == b.x; }

static_assert(!std::is_arithmetic_v<emul>,
              "emul must be a class type to stand in for an emulated format");
static_assert(Field<emul>, "emul must satisfy the scalar concept mult() requires");

/// Banded CSR matrix: `band` entries per row, wrapping. Regular by design --
/// the grain heuristic averages nnz/row, so a skewed matrix would test the
/// scheduler's balance rather than the correctness this file is about.
mat::compressed2D<emul> banded(std::size_t n, std::size_t band) {
    mat::compressed2D<emul> A(n, n);
    {
        mat::inserter<mat::compressed2D<emul>> ins(A, band);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t b = 0; b < band; ++b) {
                const std::size_t c = (r + b * 7 + 1) % n;
                ins[r][c] << emul{0.5 + static_cast<double>((r * 31 + c) % 97) / 32.0};
            }
    }
    return A;
}

vec::dense_vector<emul> rhs(std::size_t n) {
    vec::dense_vector<emul> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x(i) = emul{-0.25 + static_cast<double>((i * 53) % 89) / 8.0};
    return x;
}

} // namespace

TEST_CASE("threaded sparse CRS matvec is bit-identical to the serial loop",
          "[operation][mult][sparse][threading][mt][universal]") {
    if (detail::thread_pool::instance().size() < 2)
        WARN("pool < 2 workers: the row partition is not exercised (set MTL5_NUM_THREADS)");

    const std::size_t n = 4096, band = 6;
    const std::size_t grain = interface::row_grain<emul>(band);
    REQUIRE(n / grain >= 2);              // the problem really does split

    const auto A = banded(n, band);
    const auto x = rhs(n);
    vec::dense_vector<emul> y(n), yref(n);

    detail::mult_sparse_crs(A, x, yref);  // the serial reference loop
    mult(A, x, y);                        // dispatch -> mult_sparse_crs_par

    for (std::size_t r = 0; r < n; ++r)
        REQUIRE(y(r).x == yref(r).x);
}

TEST_CASE("threaded sparse CRS matvec honors a custom accumulator, bit-identically",
          "[operation][mult][sparse][threading][mt][universal][accumulator]") {
    const std::size_t n = 4096, band = 6;
    const auto A = banded(n, band);
    const auto x = rhs(n);
    vec::dense_vector<emul> y(n), yref(n);

    detail::mult_sparse_crs<double>(A, x, yref);
    mult<double>(A, x, y);

    for (std::size_t r = 0; r < n; ++r)
        REQUIRE(y(r).x == yref(r).x);
}

TEST_CASE("the row-range kernel agrees with the duplicated serial loop",
          "[operation][mult][sparse][threading][mt][universal]") {
    // `mult_sparse_crs` holds its own copy of the traversal rather than calling
    // `mult_sparse_crs_rows` over the whole range: the delegating form measured
    // 6.5% slower on double SpMV at one thread, which is the Krylov inner loop.
    // Two bodies means they can drift, so pin them together -- covering both the
    // whole range in one call and a hand-made partition, since it is the banded
    // call that the threaded path actually makes.
    const std::size_t n = 512, band = 6;
    const auto A = banded(n, band);
    const auto x = rhs(n);
    vec::dense_vector<emul> serial(n), whole(n), banded_(n);

    detail::mult_sparse_crs(A, x, serial);
    detail::mult_sparse_crs_rows(A, x, whole, std::size_t{0}, n);
    for (std::size_t b = 0; b < n; b += 97)
        detail::mult_sparse_crs_rows(A, x, banded_, b, std::min(b + 97, n));

    for (std::size_t r = 0; r < n; ++r) {
        REQUIRE(whole(r).x   == serial(r).x);
        REQUIRE(banded_(r).x == serial(r).x);
    }
}

TEST_CASE("mult(compressed2D) agrees with operator* on the same matvec",
          "[operation][mult][sparse][threading][mt][universal]") {
    // The two entry points that had disagreed about threading must still agree
    // about arithmetic -- both are the same traversal, so exactly.
    const std::size_t n = 4096, band = 6;
    const auto A = banded(n, band);
    const auto x = rhs(n);
    vec::dense_vector<emul> y(n);
    mult(A, x, y);
    const auto z = A * x;

    REQUIRE(z.size() == y.size());
    for (std::size_t r = 0; r < n; ++r)
        REQUIRE(z(r).x == y(r).x);
}

TEST_CASE("threaded sparse CRS matvec distributes its rows over the pool",
          "[operation][mult][sparse][threading][mt][universal]") {
    const auto pool = detail::thread_pool::instance().size();
    if (pool < 2) {
        WARN("pool < 2 workers: nothing to distribute (set MTL5_NUM_THREADS)");
        return;
    }

    const std::size_t n = 4096, band = 6;
    const auto A = banded(n, band);
    const auto x = rhs(n);
    vec::dense_vector<emul> y(n);

    { std::lock_guard<std::mutex> lk(g_tid_mtx); g_tids.clear(); }
    g_record.store(true, std::memory_order_relaxed);
    mult(A, x, y);
    g_record.store(false, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(g_tid_mtx);
    INFO("threads observed: " << g_tids.size() << " (pool size " << pool << ")");
    REQUIRE(g_tids.size() > 1);
}

TEST_CASE("transposed sparse matvec stays correct -- its scatter is not threaded",
          "[operation][mult][sparse][threading][mt][universal]") {
    // y = A^T * x accumulates into y(indices[k]) from many rows at once, so it
    // must NOT be row-partitioned. Checked against a dense transpose reference
    // computed independently: a naive partition of that loop races and drops
    // contributions, which shows up here as a wrong sum rather than a crash.
    const std::size_t n = 1024, band = 5;
    const auto A = banded(n, band);
    const auto x = rhs(n);
    vec::dense_vector<emul> y(n);

    mult(mat::view::transposed_view<mat::compressed2D<emul>>(A), x, y);

    std::vector<double> ref(n, 0.0);
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();
    for (std::size_t r = 0; r < n; ++r)
        for (auto k = starts[r]; k < starts[r + 1]; ++k)
            ref[indices[k]] += data[k].x * x(r).x;

    for (std::size_t j = 0; j < n; ++j)
        REQUIRE(y(j).x == ref[j]);
}
