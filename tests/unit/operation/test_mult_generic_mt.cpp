// Threaded generic dense GEMV/GEMM for NON-BLAS scalar types (#446).
//
// `is_blas_scalar_v` used to gate two unrelated questions at once: "may this go
// to OpenBLAS / a SIMD micro-kernel?" and "may these output rows be split across
// cores?". Only the first is about the scalar type, so every software-emulated
// format -- posit, cfloat, lns, the dd/qd cascades -- ran its dense matvec and
// matmul on one core no matter what MTL5_NUM_THREADS said. These tests pin the
// separation: a class-typed scalar that no BLAS and no `batch<>` can touch still
// gets its rows distributed, and the answer does not change when it does.
//
// MTL5 has no dependency on Universal, so `emul` below stands in for a posit:
// a class type (so `std::is_arithmetic_v` is false and every SIMD/BLAS gate
// rejects it) wrapping a double (so the expected values are computable inline).
//
// Two distinct claims are asserted, and they need different machinery:
//
//   1. BIT-IDENTITY. Each output element is produced by exactly one row band
//      running the identical inner loop, so the threaded result must equal the
//      serial `detail::mult_generic` EXACTLY -- `==`, not a tolerance. A
//      tolerance here would pass even if the partition changed the summation
//      order, which is the failure this is watching for.
//   2. THE ROWS ACTUALLY SPLIT. Bit-identity is trivially satisfied by a kernel
//      that silently stayed serial, which is precisely the bug. So `emul`'s
//      multiply records the thread that executed it, and one test asserts more
//      than one thread appears. Recording is off by default (an atomic flag)
//      because the lock it takes would dominate the larger cases.
//
// Sets MTL5_NUM_THREADS before the pool's first use -- the only in-process way
// to exercise the threading. Worth running under TSan (-DMTL5_SANITIZE=thread)
// for races on the disjoint output writes.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>

#include <mtl/mat/dense2D.hpp>
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
/// operations MTL5 needs and nothing a hardware kernel can use. The conversion
/// to double is EXPLICIT on purpose -- an implicit one would make `emul + 1.0`
/// ambiguous and, worse, could let a mixed expression silently leave the type.
struct emul {
    double x = 0.0;
    emul() = default;
    emul(double d) : x(d) {}
    explicit operator double() const { return x; }
};

emul operator+(emul a, emul b) { return emul{a.x + b.x}; }
emul operator-(emul a, emul b) { return emul{a.x - b.x}; }
emul operator-(emul a)         { return emul{-a.x}; }
emul operator/(emul a, emul b) { return emul{a.x / b.x}; }
emul operator*(emul a, emul b) { note_thread(); return emul{a.x * b.x}; }
bool operator==(emul a, emul b) { return a.x == b.x; }

static_assert(!std::is_arithmetic_v<emul>,
              "emul must be a class type, or it would reach the accelerated paths "
              "and stop standing in for an emulated format");
static_assert(Field<emul>, "emul must satisfy the scalar concept mult() requires");
static_assert(interface::ThreadableDenseMatrix<mat::dense2D<emul>>,
              "the whole point of #446: a non-BLAS scalar is still threadable");
static_assert(!interface::BlasDenseMatrix<mat::dense2D<emul>>,
              "... and is still rejected by the BLAS gate");

using colmaj = mat::parameters<tag::col_major>;

double vv(std::size_t i) { return 0.5 + static_cast<double>((i * 37) % 101) / 16.0; }
double ww(std::size_t i) { return -0.25 + static_cast<double>((i * 53) % 89) / 8.0; }

/// Rows a chunk holds for a GEMV of `nn` columns, and for a GEMM row of `nn*kk`
/// units -- the same arithmetic `mult_generic_par` uses, so a test can state
/// how many chunks its problem forms instead of hoping it forms more than one.
std::size_t chunks(std::size_t rows, std::size_t grain) { return rows / grain; }

} // namespace

TEST_CASE("threaded generic GEMV on a non-BLAS scalar is bit-identical, row-major",
          "[operation][mult][threading][mt][universal]") {
    const auto pool = detail::thread_pool::instance().size();
    if (pool < 2)
        WARN("pool < 2 workers: the row partition is not exercised (set MTL5_NUM_THREADS)");

    const std::size_t m = 512, n = 64;
    const std::size_t grain = interface::row_grain<emul>(n);
    REQUIRE(chunks(m, grain) >= 2);          // the problem really does split

    mat::dense2D<emul> A(m, n);
    vec::dense_vector<emul> x(n), y(m), yref(m);
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < n; ++c)
            A(r, c) = emul{vv(r * n + c)};
    for (std::size_t c = 0; c < n; ++c) x(c) = emul{ww(c)};

    detail::mult_generic(A, x, yref);        // the serial reference loop
    mult(A, x, y);                           // dispatch -> mult_generic_par

    for (std::size_t r = 0; r < m; ++r)
        REQUIRE(y(r).x == yref(r).x);        // exact: same terms, same order
}

TEST_CASE("threaded generic GEMV on a non-BLAS scalar is bit-identical, col-major",
          "[operation][mult][threading][mt][universal]") {
    const std::size_t m = 512, n = 64;
    mat::dense2D<emul, colmaj> A(m, n);
    vec::dense_vector<emul> x(n), y(m), yref(m);
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < n; ++c)
            A(r, c) = emul{vv(r + c * m)};
    for (std::size_t c = 0; c < n; ++c) x(c) = emul{ww(c)};

    detail::mult_generic(A, x, yref);
    mult(A, x, y);

    for (std::size_t r = 0; r < m; ++r)
        REQUIRE(y(r).x == yref(r).x);
}

TEST_CASE("threaded generic GEMM on a non-BLAS scalar is bit-identical",
          "[operation][mult][threading][mt][universal]") {
    const std::size_t m = 256, n = 32, k = 32;
    const std::size_t grain = interface::row_grain<emul>(n * k);
    REQUIRE(chunks(m, grain) >= 2);

    mat::dense2D<emul> A(m, k), B(k, n), C(m, n), Cref(m, n);
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < k; ++c) A(r, c) = emul{vv(r * k + c)};
    for (std::size_t r = 0; r < k; ++r)
        for (std::size_t c = 0; c < n; ++c) B(r, c) = emul{ww(r * n + c)};

    detail::mult_generic(A, B, Cref);
    mult(A, B, C);

    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < n; ++c)
            REQUIRE(C(r, c).x == Cref(r, c).x);
}

TEST_CASE("threaded generic GEMM honors a custom accumulator, bit-identically",
          "[operation][mult][threading][mt][universal][accumulator]") {
    // The mixed-precision path -- element type emul, products summed in double,
    // rounded out to emul on store -- is the one a mixed-precision sweep spends
    // its time in, so it must thread too rather than falling back to serial.
    const std::size_t m = 256, n = 32, k = 32;
    mat::dense2D<emul> A(m, k), B(k, n), C(m, n), Cref(m, n);
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < k; ++c) A(r, c) = emul{vv(r * k + c)};
    for (std::size_t r = 0; r < k; ++r)
        for (std::size_t c = 0; c < n; ++c) B(r, c) = emul{ww(r * n + c)};

    detail::mult_generic<double>(A, B, Cref);
    mult<double>(A, B, C);

    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < n; ++c)
            REQUIRE(C(r, c).x == Cref(r, c).x);
}

TEST_CASE("threaded generic GEMV distributes its rows over the pool",
          "[operation][mult][threading][mt][universal]") {
    // Bit-identity alone cannot tell a working partition from one that stayed
    // serial -- so observe the partition directly. Small enough that the mutex
    // in emul's multiply is affordable: 4096 x 8 is 32K recorded products.
    const auto pool = detail::thread_pool::instance().size();
    if (pool < 2) {
        WARN("pool < 2 workers: nothing to distribute (set MTL5_NUM_THREADS)");
        return;
    }

    const std::size_t m = 4096, n = 8;
    const std::size_t grain = interface::row_grain<emul>(n);
    REQUIRE(chunks(m, grain) >= 2);

    mat::dense2D<emul> A(m, n);
    vec::dense_vector<emul> x(n), y(m);
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < n; ++c) A(r, c) = emul{vv(r * n + c)};
    for (std::size_t c = 0; c < n; ++c) x(c) = emul{ww(c)};

    { std::lock_guard<std::mutex> lk(g_tid_mtx); g_tids.clear(); }
    g_record.store(true, std::memory_order_relaxed);
    mult(A, x, y);
    g_record.store(false, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(g_tid_mtx);
    INFO("threads observed: " << g_tids.size() << " (pool size " << pool << ")");
    REQUIRE(g_tids.size() > 1);
}
