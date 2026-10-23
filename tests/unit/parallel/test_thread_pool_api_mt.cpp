// The public thread pool API (#630): mtl::parallel::thread_pool's contract,
// run() and parallel_for() allocating nothing, and the per-call team cap.
#include <catch2/catch_test_macros.hpp>

#include <mtl/parallel/thread_pool.hpp>
#include <mtl/detail/thread_pool.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <new>
#include <set>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// A sanitizer runtime defines operator new itself, so replacing it fails to
// link (TSan's libclang_rt.tsan_cxx in CI's _mt job). Under one, the allocation
// tests are compiled out and the rest of this file runs, which is what that
// job is for: the happens-before test below.
#define MTL5_TEST_COUNTS_ALLOCATIONS 1
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#  undef MTL5_TEST_COUNTS_ALLOCATIONS
#  define MTL5_TEST_COUNTS_ALLOCATIONS 0
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
      __has_feature(memory_sanitizer)
#    undef MTL5_TEST_COUNTS_ALLOCATIONS
#    define MTL5_TEST_COUNTS_ALLOCATIONS 0
#  endif
#endif

#if MTL5_TEST_COUNTS_ALLOCATIONS
// Every allocation in this test executable goes through these, so a region can
// be checked for allocating anything at all, on any thread.
namespace {
std::atomic<bool> counting{false};
std::atomic<std::size_t> allocations{0};

void* counted_alloc(std::size_t n) {
    if (counting.load(std::memory_order_relaxed))
        allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}

/// Allocations made while `f` runs.
template <typename F>
std::size_t allocations_during(F&& f) {
    allocations.store(0);
    counting.store(true);
    f();
    counting.store(false);
    return allocations.load();
}
} // namespace

void* operator new(std::size_t n) { return counted_alloc(n); }
void* operator new[](std::size_t n) { return counted_alloc(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#endif

using mtl::parallel::thread_pool;

TEST_CASE("parallel: the detail names are the public pool", "[parallel][thread_pool]") {
    STATIC_REQUIRE(std::is_same_v<mtl::detail::thread_pool, mtl::parallel::thread_pool>);
    REQUIRE(&mtl::parallel::pool() == &mtl::detail::thread_pool::instance());
}

#if MTL5_TEST_COUNTS_ALLOCATIONS
TEST_CASE("parallel: run() does not allocate, whatever the task captures",
          "[parallel][thread_pool]") {
    thread_pool pool(4);
    // 256 bytes of captures: past every standard library's std::function small
    // buffer, so a pool that stored a copy of the task in one would allocate
    // here. (The pre-#630 pool stored a lambda holding a reference to the task,
    // which fits the buffer on the major libraries without being guaranteed to;
    // this test pins the guarantee rather than reproducing a failure.)
    std::array<double, 32> big{};
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<double>(i);
    std::array<double, 4> out{};
    auto task = [big, &out](unsigned tid) { out[tid] = big[tid] + big[31]; };
    static_assert(sizeof(task) > 256);

    pool.run(4, task);   // warm up: the workers have reached their wait
    const std::size_t n = allocations_during([&] {
        for (int rep = 0; rep < 100; ++rep) pool.run(4, task);
    });
    REQUIRE(n == 0);
    for (unsigned t = 0; t < 4; ++t) REQUIRE(out[t] == static_cast<double>(t) + 31.0);
}

TEST_CASE("parallel: parallel_for does not allocate", "[parallel][thread_pool]") {
    thread_pool pool(4);
    std::vector<int> v(10000, 0);
    std::array<char, 300> pad{};   // a large capture again
    auto body = [&v, pad](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) v[i] += 1 + pad[0];
    };
    pool.parallel_for(v.size(), 64, body);
    const std::size_t n = allocations_during([&] {
        for (int rep = 0; rep < 100; ++rep) pool.parallel_for(v.size(), 64, body);
        pool.parallel_for(v.size(), 64, 2u, body);
    });
    REQUIRE(n == 0);
    for (int x : v) REQUIRE(x == 102);
}
#endif

namespace {
std::array<std::atomic<int>, 4> hits{};
void count_tid(unsigned tid) { hits[tid].fetch_add(1); }
std::array<std::atomic<std::size_t>, 1> covered{};
void count_range(std::size_t b, std::size_t e) { covered[0].fetch_add(e - b); }
} // namespace

TEST_CASE("parallel: a free function is a task", "[parallel][thread_pool]") {
    // A function name deduces as a function reference, which has no object
    // address; run() must still accept it, as the std::function pool did.
    thread_pool pool(4);
    pool.run(4, count_tid);
    void (*fp)(unsigned) = count_tid;
    pool.run(4, fp);
    for (unsigned t = 0; t < 4; ++t) REQUIRE(hits[t].load() == 2);

    pool.parallel_for(1000, 10, count_range);
    pool.parallel_for(1000, 10, 2u, count_range);
    REQUIRE(covered[0].load() == 2000);
}

TEST_CASE("parallel: a task's writes are visible when run() returns", "[parallel][thread_pool]") {
    // Plain, non-atomic writes from every tid, read by the caller after the
    // join. The contract says these happen-before; a ThreadSanitizer build
    // checks that it is so.
    thread_pool pool(8);
    std::vector<std::size_t> slot(8 * 1024, 0);
    for (std::size_t rep = 1; rep <= 20; ++rep) {
        pool.run(8, [&](unsigned tid) {
            for (std::size_t i = 0; i < 1024; ++i) slot[tid * 1024 + i] = rep * (tid + 1);
        });
        for (unsigned t = 0; t < 8; ++t)
            for (std::size_t i = 0; i < 1024; ++i) REQUIRE(slot[t * 1024 + i] == rep * (t + 1));
    }
}

TEST_CASE("parallel: a nested region runs serially on its caller", "[parallel][thread_pool]") {
    thread_pool pool(4);
    std::vector<std::vector<unsigned>> inner(4);
    std::vector<char> same_thread(4, 0);   // not vector<bool>: tids write it concurrently
    pool.run(4, [&](unsigned tid) {
        const auto me = std::this_thread::get_id();
        bool all_here = true;
        pool.run(3, [&](unsigned t) {
            inner[tid].push_back(t);
            all_here = all_here && std::this_thread::get_id() == me;
        });
        same_thread[tid] = all_here;
    });
    for (unsigned t = 0; t < 4; ++t) {
        REQUIRE(inner[t] == std::vector<unsigned>{0, 1, 2});
        REQUIRE(same_thread[t]);
    }
}

TEST_CASE("parallel: parallel_for's max_threads caps the team", "[parallel][thread_pool]") {
    thread_pool pool(8);
    const std::size_t n = 8000;

    auto chunks = [&](unsigned cap) {
        std::mutex m;
        std::set<std::pair<std::size_t, std::size_t>> seen;
        pool.parallel_for(n, 10, cap, [&](std::size_t b, std::size_t e) {
            std::lock_guard<std::mutex> lk(m);
            seen.emplace(b, e);
        });
        return seen;
    };

    SECTION("a cap below the pool splits into that many contiguous chunks") {
        const auto c = chunks(3);
        REQUIRE(c.size() == 3);
        std::size_t next = 0;
        for (const auto& [b, e] : c) { REQUIRE(b == next); next = e; }
        REQUIRE(next == n);
    }
    SECTION("cap 1 is one serial body(0, n)") {
        REQUIRE(chunks(1) == std::set<std::pair<std::size_t, std::size_t>>{{0, n}});
    }
    SECTION("0, or a cap above the pool, is the whole pool") {
        REQUIRE(chunks(0).size() == 8);
        REQUIRE(chunks(64).size() == 8);
    }
    SECTION("the grain still applies under the cap") {
        std::size_t calls = 0;
        std::mutex m;
        pool.parallel_for(40, 10, 8u, [&](std::size_t, std::size_t) {
            std::lock_guard<std::mutex> lk(m);
            ++calls;
        });
        REQUIRE(calls == 4);   // 40 units / grain 10
    }
}

TEST_CASE("parallel: a capped reduction is the reduction of a pool that size",
          "[parallel][thread_pool]") {
    // The grouping depends only on the team, so a cap of 3 on an 8-thread pool
    // must give the same bits as a 3-thread pool.
    const std::size_t n = 30001;
    auto map = [](std::size_t b, std::size_t e) {
        double s = 0.0;
        for (std::size_t i = b; i < e; ++i) s += 1.0 / static_cast<double>(i + 1);
        return s;
    };
    thread_pool eight(8), three(3);
    const double capped = eight.parallel_reduce<double>(n, 16, 3u, map);
    const double sized = three.parallel_reduce<double>(n, 16, map);
    REQUIRE(capped == sized);
    REQUIRE(eight.parallel_reduce<double>(n, 16, 0u, map) == eight.parallel_reduce<double>(n, 16, map));
}

TEST_CASE("parallel: parallel_accumulate's cap sets the number of partials",
          "[parallel][thread_pool]") {
    thread_pool pool(8);
    std::atomic<int> inits{0};
    const long long got = pool.parallel_accumulate<long long>(
        10000, 10, 2u,
        [&] { ++inits; return 0LL; },
        [](long long& acc, std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) acc += static_cast<long long>(i);
        },
        [](long long& into, const long long& from) { into += from; });
    REQUIRE(inits.load() == 2);
    REQUIRE(got == 9999LL * 10000LL / 2);
}
