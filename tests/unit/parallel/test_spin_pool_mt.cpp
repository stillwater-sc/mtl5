// spin_pool (#636): thread_pool's contract, checked on both pool types and on
// a spin_pool that never spins (every wait goes to sleep); the partitions of
// the chunked loops are identical across the two types; run() and
// parallel_for() allocate nothing on either the spinning or the sleeping path;
// and a stress test interleaves regions, nesting and concurrent callers.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <mtl/parallel/spin_pool.hpp>
#include <mtl/parallel/thread_pool.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// As in test_thread_pool_api_mt.cpp: a sanitizer runtime defines operator new
// itself, so under one the allocation tests are compiled out.
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
namespace {
std::atomic<bool> counting{false};
std::atomic<std::size_t> allocations{0};

void* counted_alloc(std::size_t n) {
    if (counting.load(std::memory_order_relaxed))
        allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}

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

using mtl::parallel::spin_pool;
using mtl::parallel::thread_pool;
namespace par = mtl::parallel;

namespace {

// The pools under test, as types for TEMPLATE_TEST_CASE.
struct cv_pool {
    using type = thread_pool;
    static std::unique_ptr<type> make(unsigned n) { return std::make_unique<type>(n); }
};
struct spinning_pool {
    using type = spin_pool;
    static std::unique_ptr<type> make(unsigned n) { return std::make_unique<type>(n); }
};
struct sleeping_pool {   // spin 0: every worker and every join sleeps at once
    using type = spin_pool;
    static std::unique_ptr<type> make(unsigned n) { return std::make_unique<type>(n, 0u); }
};

} // namespace

#define POOLS cv_pool, spinning_pool, sleeping_pool

TEMPLATE_TEST_CASE("spin_pool: every tid runs once, tid 0 on the caller",
                   "[parallel][spin_pool]", POOLS) {
    auto pool = TestType::make(4);
    for (int rep = 0; rep < 50; ++rep) {
        std::array<std::thread::id, 4> ran{};
        std::array<std::atomic<int>, 4> hits{};
        pool->run(4, [&](unsigned tid) {
            ran[tid] = std::this_thread::get_id();
            hits[tid].fetch_add(1);
        });
        REQUIRE(ran[0] == std::this_thread::get_id());
        for (unsigned t = 0; t < 4; ++t) REQUIRE(hits[t].load() == 1);
        REQUIRE(std::set<std::thread::id>(ran.begin(), ran.end()).size() == 4);
    }
    // count is clamped to size(); count 0 runs nothing.
    std::atomic<int> n{0};
    pool->run(9, [&](unsigned) { n.fetch_add(1); });
    REQUIRE(n.load() == 4);
    pool->run(0, [&](unsigned) { n.fetch_add(1); });
    REQUIRE(n.load() == 4);
}

namespace {
std::array<std::atomic<int>, 4> free_hits{};
void free_task(unsigned tid) { free_hits[tid].fetch_add(1); }
} // namespace

TEST_CASE("spin_pool: a free function is a task", "[parallel][spin_pool]") {
    spin_pool pool(4);
    pool.run(4, free_task);
    for (unsigned t = 0; t < 4; ++t) REQUIRE(free_hits[t].load() == 1);
}

TEMPLATE_TEST_CASE("spin_pool: a task's writes are visible when run() returns",
                   "[parallel][spin_pool]", POOLS) {
    // Plain writes, read by the caller after the join; TSan checks the order.
    auto pool = TestType::make(8);
    std::vector<std::size_t> slot(8 * 512, 0);
    for (std::size_t rep = 1; rep <= 40; ++rep) {
        pool->run(8, [&](unsigned tid) {
            for (std::size_t i = 0; i < 512; ++i) slot[tid * 512 + i] = rep * (tid + 1);
        });
        for (unsigned t = 0; t < 8; ++t)
            for (std::size_t i = 0; i < 512; ++i) REQUIRE(slot[t * 512 + i] == rep * (t + 1));
    }
}

TEMPLATE_TEST_CASE("spin_pool: a nested region runs serially on its caller",
                   "[parallel][spin_pool]", POOLS) {
    auto pool = TestType::make(4);
    std::vector<std::vector<unsigned>> inner(4);
    std::vector<char> same_thread(4, 0);
    pool->run(4, [&](unsigned tid) {
        const auto me = std::this_thread::get_id();
        bool all_here = true;
        pool->run(3, [&](unsigned t) {
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

TEMPLATE_TEST_CASE("spin_pool: concurrent callers each get every tid once",
                   "[parallel][spin_pool]", POOLS) {
    // Four threads share one pool; whichever finds it busy runs serially.
    auto pool = TestType::make(4);
    constexpr int callers = 4, regions = 200;
    std::vector<std::array<int, 4>> sums(callers, std::array<int, 4>{});
    std::vector<std::thread> threads;
    for (int c = 0; c < callers; ++c) {
        threads.emplace_back([&, c] {
            for (int r = 0; r < regions; ++r) {
                std::array<int, 4> once{};
                pool->run(4, [&](unsigned tid) { once[tid] += 1; });
                for (unsigned t = 0; t < 4; ++t) sums[c][t] += once[t];
            }
        });
    }
    for (auto& t : threads) t.join();
    for (int c = 0; c < callers; ++c)
        for (unsigned t = 0; t < 4; ++t) REQUIRE(sums[c][t] == regions);
}

TEMPLATE_TEST_CASE("spin_pool: the first exception is rethrown after the join",
                   "[parallel][spin_pool]", POOLS) {
    auto pool = TestType::make(4);
    std::atomic<int> finished{0};
    auto run_throwing = [&](unsigned thrower) {
        finished.store(0);
        pool->run(4, [&](unsigned tid) {
            if (tid == thrower) throw std::runtime_error("tid");
            finished.fetch_add(1);
        });
    };
    REQUIRE_THROWS_AS(run_throwing(2), std::runtime_error);
    REQUIRE(finished.load() == 3);   // every other tid completed before the rethrow
    REQUIRE_THROWS_AS(run_throwing(0), std::runtime_error);
    REQUIRE(finished.load() == 3);

    // Several throwers: exactly one exception arrives.
    REQUIRE_THROWS_AS(pool->run(4, [](unsigned) { throw std::logic_error("all"); }),
                      std::logic_error);

    // The pool is usable afterwards, and its exception slot is clear.
    std::atomic<int> n{0};
    pool->run(4, [&](unsigned) { n.fetch_add(1); });
    REQUIRE(n.load() == 4);
}

TEMPLATE_TEST_CASE("spin_pool: in_region and on_pool_worker as on thread_pool",
                   "[parallel][spin_pool]", POOLS) {
    auto pool = TestType::make(4);
    REQUIRE_FALSE(par::detail::in_region());
    std::array<char, 4> in{}, worker{};
    pool->run(4, [&](unsigned tid) {
        in[tid] = par::detail::in_region();
        worker[tid] = par::detail::on_pool_worker();
    });
    for (unsigned t = 0; t < 4; ++t) REQUIRE(in[t]);
    REQUIRE_FALSE(worker[0]);
    for (unsigned t = 1; t < 4; ++t) REQUIRE(worker[t]);
    REQUIRE_FALSE(par::detail::in_region());
}

TEMPLATE_TEST_CASE("spin_pool: the chunked loops partition as thread_pool's do",
                   "[parallel][spin_pool]", spinning_pool, sleeping_pool) {
    // One template does the partitioning for both pool types, so the chunks,
    // and with them every reduction's bits, are the same.
    for (unsigned n_threads : {2u, 3u, 8u}) {
        auto p = TestType::make(n_threads);
        thread_pool ref(n_threads);
        for (unsigned cap : {0u, 2u, 5u}) {
            auto chunks = [&](auto& pool) {
                std::mutex m;
                std::set<std::pair<std::size_t, std::size_t>> seen;
                pool.parallel_for(10007, 16, cap, [&](std::size_t b, std::size_t e) {
                    std::lock_guard<std::mutex> lk(m);
                    seen.emplace(b, e);
                });
                return seen;
            };
            REQUIRE(chunks(*p) == chunks(ref));

            auto map = [](std::size_t b, std::size_t e) {
                double s = 0.0;
                for (std::size_t i = b; i < e; ++i) s += 1.0 / static_cast<double>(i + 1);
                return s;
            };
            REQUIRE(p->template parallel_reduce<double>(30001, 16, cap, map) ==
                    ref.parallel_reduce<double>(30001, 16, cap, map));

            auto init = [] { return 0.0; };
            auto acc_map = [](double& a, std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e; ++i) a += 1.0 / static_cast<double>(i + 1);
            };
            auto combine = [](double& into, const double& from) { into += from; };
            REQUIRE(p->template parallel_accumulate<double>(30001, 16, cap, init, acc_map, combine) ==
                    ref.parallel_accumulate<double>(30001, 16, cap, init, acc_map, combine));
        }
    }
}

TEMPLATE_TEST_CASE("spin_pool: regions spaced past the spin wake sleeping workers",
                   "[parallel][spin_pool]", spinning_pool, sleeping_pool) {
    // A gap longer than any spin, so every worker has gone to sleep when the
    // next region is published.
    auto pool = TestType::make(4);
    for (int rep = 0; rep < 10; ++rep) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        std::array<std::atomic<int>, 4> hits{};
        pool->run(4, [&](unsigned tid) { hits[tid].fetch_add(1); });
        for (unsigned t = 0; t < 4; ++t) REQUIRE(hits[t].load() == 1);
    }
}

TEST_CASE("spin_pool: a slow worker puts the caller to sleep in its join",
          "[parallel][spin_pool]") {
    // spin 0: the caller sleeps at once, and the last worker must wake it.
    spin_pool pool(3, 0u);
    for (int rep = 0; rep < 10; ++rep) {
        std::atomic<int> done{0};
        pool.run(3, [&](unsigned tid) {
            if (tid == 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            done.fetch_add(1);
        });
        REQUIRE(done.load() == 3);
    }
}

TEST_CASE("spin_pool: destruction with workers spinning or asleep", "[parallel][spin_pool]") {
    { spin_pool p(4); }                         // workers may not have started
    {
        spin_pool p(4);
        p.run(4, [](unsigned) {});
    }                                           // workers spinning
    {
        spin_pool p(4, 0u);
        p.run(4, [](unsigned) {});
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }                                           // workers asleep
    { spin_pool p(1); }                         // serial: no workers
    SUCCEED();
}

TEST_CASE("spin_pool: a pool of one runs everything on the caller", "[parallel][spin_pool]") {
    spin_pool pool(1);
    REQUIRE(pool.size() == 1);
    std::vector<unsigned> order;
    pool.run(4, [&](unsigned tid) { order.push_back(tid); });
    REQUIRE(order == std::vector<unsigned>{0});   // clamped to size() == 1
}

TEST_CASE("spin_pool: stress, interleaved regions, nesting and concurrent callers",
          "[parallel][spin_pool]") {
    for (unsigned n : {2u, 3u, 5u, 8u}) {
        for (unsigned spin : {0u, 16u, spin_pool::default_spin}) {
            spin_pool pool(n, spin);
            std::atomic<long> total{0};
            long expect = 0;
            auto caller = [&](unsigned seed, int regions) {
                long local_expect = 0;
                for (int r = 0; r < regions; ++r) {
                    const unsigned count = 1 + (seed * 7u + static_cast<unsigned>(r)) % (n + 1);
                    const bool nest = (r % 5) == 0;
                    pool.run(count, [&](unsigned) {
                        total.fetch_add(1);
                        if (nest) pool.run(2, [&](unsigned) { total.fetch_add(1); });
                    });
                    const long ran = count < n ? count : n;
                    local_expect += ran + (nest ? 2 * ran : 0);
                }
                return local_expect;
            };
            std::atomic<long> other_expect{0};
            std::thread other([&] { other_expect.store(caller(3, 150)); });
            expect += caller(1, 300);
            other.join();
            expect += other_expect.load();
            INFO("n " << n << " spin " << spin);
            REQUIRE(total.load() == expect);
        }
    }
}

#if MTL5_TEST_COUNTS_ALLOCATIONS
TEMPLATE_TEST_CASE("spin_pool: run() and parallel_for allocate nothing",
                   "[parallel][spin_pool]", spinning_pool, sleeping_pool) {
    auto pool = TestType::make(4);
    std::array<double, 32> big{};
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<double>(i);
    std::array<double, 4> out{};
    auto task = [big, &out](unsigned tid) { out[tid] = big[tid] + big[31]; };
    static_assert(sizeof(task) > 256);
    std::vector<int> v(10000, 0);
    auto body = [&v](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) v[i] += 1;
    };

    pool->run(4, task);   // warm up: the workers have reached their wait
    const std::size_t n = allocations_during([&] {
        for (int rep = 0; rep < 100; ++rep) pool->run(4, task);
        for (int rep = 0; rep < 100; ++rep) pool->parallel_for(v.size(), 64, body);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));   // workers asleep
        pool->run(4, task);
    });
    REQUIRE(n == 0);
    for (unsigned t = 0; t < 4; ++t) REQUIRE(out[t] == static_cast<double>(t) + 31.0);
    for (int x : v) REQUIRE(x == 100);
}
#endif
