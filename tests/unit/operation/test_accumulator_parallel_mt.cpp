// Accumulator-typed reductions on the thread pool, combined by
// accumulator_traits::merge (#582).
//
// dot, dot_real, two_norm and frobenius_norm with an explicit Accumulator used
// to run serially whatever MTL5_NUM_THREADS said: the pool's parallel_reduce
// combines partials with `+` on the result type, which a super-accumulator does
// not have. They now split across the pool when the accumulator can merge.
//
// Two claims, needing different machinery (as in test_mult_generic_mt):
//
//   1. BIT-IDENTITY for an exact accumulator. `exact_acc` sums integer-valued
//      products in an int64, so its partial sums and their merge are exact and
//      the threaded result must equal the serial loop EXACTLY.
//   2. THE WORK ACTUALLY SPLITS. A kernel that silently stayed serial passes (1)
//      trivially, so exact_acc's add_product records the executing thread, and
//      the tests require more than one.
//
// Plus: an accumulator without a merge stays serial, and a rounding
// accumulator is deterministic for the fixed thread count.
//
// Sets MTL5_NUM_THREADS before the pool's first use, the only in-process way to
// exercise the threading.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <random>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/vec/dense_vector.hpp>

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
std::unordered_set<std::thread::id> g_tids;

void record_thread() {
    std::lock_guard<std::mutex> lk(g_tid_mtx);
    g_tids.insert(std::this_thread::get_id());
}

std::size_t threads_seen_and_reset() {
    std::lock_guard<std::mutex> lk(g_tid_mtx);
    const std::size_t n = g_tids.size();
    g_tids.clear();
    return n;
}

} // namespace

namespace exactlib {

// An exact accumulator for integer-valued doubles, plugged in through the ADL
// hook as a number library's super-accumulator would be.
struct exact_acc {
    std::int64_t sum;
};

template <std::same_as<double> V>
void accumulator_assign(exact_acc& a, const V& v) { a.sum = static_cast<std::int64_t>(v); }
inline void accumulator_clear(exact_acc& a) { a.sum = 0; }
template <std::same_as<double> V>
void accumulator_add_product(exact_acc& a, const V& m, const V& v) {
    record_thread();
    a.sum += static_cast<std::int64_t>(m) * static_cast<std::int64_t>(v);
}
template <typename R>
R accumulator_value(const exact_acc& a, std::type_identity<R>) { return static_cast<R>(a.sum); }
inline void accumulator_merge(exact_acc& a, const exact_acc& b) { a.sum += b.sum; }

// The same accumulator with no accumulator_merge.
struct mergeless_acc {
    std::int64_t sum;
};
inline void accumulator_clear(mergeless_acc& a) { a.sum = 0; }
template <std::same_as<double> V>
void accumulator_assign(mergeless_acc& a, const V& v) { a.sum = static_cast<std::int64_t>(v); }
template <std::same_as<double> V>
void accumulator_add_product(mergeless_acc& a, const V& m, const V& v) {
    record_thread();
    a.sum += static_cast<std::int64_t>(m) * static_cast<std::int64_t>(v);
}
template <typename R>
R accumulator_value(const mergeless_acc& a, std::type_identity<R>) { return static_cast<R>(a.sum); }

} // namespace exactlib

using exactlib::exact_acc;
using mtl::detail::thread_pool;

namespace {

// Integer values in [-1000, 1000], as doubles.
mtl::vec::dense_vector<double> int_vector(std::size_t n, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> d(-1000, 1000);
    mtl::vec::dense_vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v(i) = static_cast<double>(d(g));
    return v;
}

std::int64_t serial_dot(const mtl::vec::dense_vector<double>& x,
                        const mtl::vec::dense_vector<double>& y) {
    std::int64_t s = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += static_cast<std::int64_t>(x(i)) * static_cast<std::int64_t>(y(i));
    return s;
}

bool pool_is_threaded() {
    if (thread_pool::instance().size() < 2) {
        WARN("pool < 2 workers: the split is not exercised (set MTL5_NUM_THREADS)");
        return false;
    }
    return true;
}

} // namespace

TEST_CASE("mergeable_accumulator says which accumulators can split",
          "[operation][accumulator][mt]") {
    using mtl::math::mergeable_accumulator;
    STATIC_REQUIRE(mergeable_accumulator<double, float>);
    STATIC_REQUIRE(mergeable_accumulator<mtl::math::fma_accumulator<double>, float>);
    STATIC_REQUIRE(mergeable_accumulator<exact_acc, double>);
    STATIC_REQUIRE_FALSE(mergeable_accumulator<exactlib::mergeless_acc, double>);
}

TEST_CASE("thread_pool::parallel_accumulate: same exact sum for every team size",
          "[detail][thread_pool][mt]") {
    const std::size_t n = 10007;                     // prime: uneven last chunk
    std::vector<std::int64_t> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = static_cast<std::int64_t>(i * i % 977) - 488;
    std::int64_t expect = 0;
    for (auto d : data) expect += d;

    for (unsigned team : {1u, 2u, 3u, 4u, 7u}) {
        thread_pool pool(team);
        for (std::size_t grain : {std::size_t{1}, std::size_t{100}, std::size_t{4000}, n}) {
            const auto got = pool.parallel_accumulate<std::int64_t>(
                n, grain, [] { return std::int64_t{0}; },
                [&](std::int64_t& acc, std::size_t lo, std::size_t hi) {
                    for (std::size_t i = lo; i < hi; ++i) acc += data[i];
                },
                [](std::int64_t& into, const std::int64_t& from) { into += from; });
            REQUIRE(got == expect);
        }
    }
    // n == 0 runs map once over an empty range and returns the empty accumulator.
    thread_pool pool(4);
    const auto empty = pool.parallel_accumulate<std::int64_t>(
        0, 1, [] { return std::int64_t{7}; },
        [](std::int64_t&, std::size_t lo, std::size_t hi) { REQUIRE(lo == hi); },
        [](std::int64_t& a, const std::int64_t& b) { a += b; });
    REQUIRE(empty == 7);
}

TEST_CASE("dot and dot_real with an exact accumulator: threaded, serial-exact",
          "[operation][accumulator][mt]") {
    const std::size_t n = 20000;                     // > 4 chunks of 1024
    const auto x = int_vector(n, 1);
    const auto y = int_vector(n, 2);
    const auto expect = static_cast<double>(serial_dot(x, y));

    threads_seen_and_reset();
    REQUIRE(mtl::dot<exact_acc, double>(x, y) == expect);
    const std::size_t dot_threads = threads_seen_and_reset();
    REQUIRE(mtl::dot_real<exact_acc, double>(x, y) == expect);
    const std::size_t dot_real_threads = threads_seen_and_reset();

    if (pool_is_threaded()) {
        REQUIRE(dot_threads > 1);
        REQUIRE(dot_real_threads > 1);
    }
}

TEST_CASE("two_norm and frobenius_norm with an exact accumulator: threaded, serial-exact",
          "[operation][accumulator][mt]") {
    const std::size_t n = 20000;
    const auto v = int_vector(n, 3);
    const double expect_v = std::sqrt(static_cast<double>(serial_dot(v, v)));

    threads_seen_and_reset();
    REQUIRE(mtl::two_norm<exact_acc, double>(v) == expect_v);
    const std::size_t norm_threads = threads_seen_and_reset();

    const std::size_t rows = 150, cols = 140;        // 21000 terms, 7 rows per chunk
    const auto flat = int_vector(rows * cols, 4);
    mtl::mat::dense2D<double> m(rows, cols);
    std::int64_t ss = 0;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            m(r, c) = flat(r * cols + c);
            ss += static_cast<std::int64_t>(m(r, c)) * static_cast<std::int64_t>(m(r, c));
        }
    REQUIRE(mtl::frobenius_norm<exact_acc, double>(m) == std::sqrt(static_cast<double>(ss)));
    const std::size_t frob_threads = threads_seen_and_reset();

    if (pool_is_threaded()) {
        REQUIRE(norm_threads > 1);
        REQUIRE(frob_threads > 1);
    }
}

TEST_CASE("an accumulator without merge stays serial", "[operation][accumulator][mt]") {
    const std::size_t n = 20000;
    const auto x = int_vector(n, 5);
    const auto y = int_vector(n, 6);
    threads_seen_and_reset();
    REQUIRE(mtl::dot<exactlib::mergeless_acc, double>(x, y) ==
            static_cast<double>(serial_dot(x, y)));
    REQUIRE(threads_seen_and_reset() == 1);
}

TEST_CASE("a rounding accumulator: deterministic for the thread count, close to serial",
          "[operation][accumulator][mt]") {
    // long double over float takes the generic accumulator path (no SIMD
    // widening kernel for it), with 65536-term chunks.
    const std::size_t n = 300000;
    std::mt19937 g(7);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    mtl::vec::dense_vector<float> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) { x(i) = d(g); y(i) = d(g); }

    const long double a = mtl::dot<long double, long double>(x, y);
    const long double b = mtl::dot<long double, long double>(x, y);
    REQUIRE(a == b);

    long double serial = 0.0L;
    for (std::size_t i = 0; i < n; ++i)
        serial += static_cast<long double>(x(i)) * static_cast<long double>(y(i));
    REQUIRE(std::abs(a - serial) <= 1e-12L * std::abs(serial) + 1e-12L);
}
