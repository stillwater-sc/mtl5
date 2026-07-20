// Tests for the persistent thread pool (#221).
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/thread_pool.hpp>

#include <atomic>
#include <numeric>
#include <stdexcept>
#include <vector>

using mtl::detail::thread_pool;

TEST_CASE("thread_pool: each tid runs exactly once", "[detail][thread_pool]") {
    for (unsigned n : {1u, 2u, 4u, 8u}) {
        thread_pool pool(n);
        REQUIRE(pool.size() == n);
        std::vector<int> hits(n, 0);
        pool.run(n, [&](unsigned tid) { hits[tid] += 1; });
        for (unsigned t = 0; t < n; ++t) REQUIRE(hits[t] == 1);
    }
}

TEST_CASE("thread_pool: parallel partition sums correctly (many regions)", "[detail][thread_pool]") {
    const unsigned n = 8;
    thread_pool pool(n);
    const std::size_t N = 100000;
    // Each region: partition [0,N) across tids, accumulate into per-tid partials.
    for (int rep = 0; rep < 50; ++rep) {
        std::vector<long long> partial(n, 0);
        pool.run(n, [&](unsigned tid) {
            long long s = 0;
            for (std::size_t i = tid; i < N; i += n) s += static_cast<long long>(i);
            partial[tid] = s;
        });
        long long total = std::accumulate(partial.begin(), partial.end(), 0LL);
        REQUIRE(total == static_cast<long long>(N) * (N - 1) / 2);
    }
}

TEST_CASE("thread_pool: count < size only runs that many tids", "[detail][thread_pool]") {
    thread_pool pool(8);
    std::vector<int> hits(8, 0);
    pool.run(3, [&](unsigned tid) { hits[tid] += 1; });
    REQUIRE(hits[0] == 1);
    REQUIRE(hits[1] == 1);
    REQUIRE(hits[2] == 1);
    for (unsigned t = 3; t < 8; ++t) REQUIRE(hits[t] == 0);  // idle tids untouched
}

TEST_CASE("thread_pool: size 1 is the serial path", "[detail][thread_pool]") {
    thread_pool pool(1);
    int calls = 0;
    pool.run(1, [&](unsigned tid) { REQUIRE(tid == 0); calls++; });
    REQUIRE(calls == 1);
}

TEST_CASE("thread_pool: worker exception propagates after join", "[detail][thread_pool]") {
    thread_pool pool(4);
    std::atomic<int> ran{0};
    auto call = [&] {
        pool.run(4, [&](unsigned tid) {
            ran.fetch_add(1);
            if (tid == 2) throw std::runtime_error("boom");
        });
    };
    REQUIRE_THROWS_AS(call(), std::runtime_error);
    REQUIRE(ran.load() == 4);            // every tid still ran (all joined)
    // Pool is reusable after an exception.
    int hits = 0;
    pool.run(4, [&](unsigned) { });
    pool.run(1, [&](unsigned) { hits++; });
    REQUIRE(hits == 1);
}

TEST_CASE("thread_pool: parallel_for covers [0,n) exactly once", "[detail][thread_pool]") {
    for (unsigned nt : {1u, 2u, 4u, 8u}) {
        thread_pool pool(nt);
        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7},
                              std::size_t{1000}, std::size_t{100000}}) {
            std::vector<int> hits(n, 0);
            pool.parallel_for(n, /*grain=*/8, [&](std::size_t b, std::size_t e) {
                REQUIRE(b <= e);
                for (std::size_t i = b; i < e; ++i) hits[i] += 1;
            });
            for (std::size_t i = 0; i < n; ++i) REQUIRE(hits[i] == 1);  // once, no gaps/overlap
        }
    }
}

TEST_CASE("thread_pool: parallel_for partial sums match serial", "[detail][thread_pool]") {
    const std::size_t n = 200000;
    thread_pool pool(8);
    std::atomic<long long> total{0};
    pool.parallel_for(n, 1024, [&](std::size_t b, std::size_t e) {
        long long s = 0;
        for (std::size_t i = b; i < e; ++i) s += static_cast<long long>(i);
        total.fetch_add(s);
    });
    REQUIRE(total.load() == static_cast<long long>(n) * (n - 1) / 2);
}

TEST_CASE("thread_pool: parallel_for below grain runs as one chunk", "[detail][thread_pool]") {
    thread_pool pool(8);
    int calls = 0;
    std::size_t seen_b = 999, seen_e = 999;
    pool.parallel_for(10, /*grain=*/64, [&](std::size_t b, std::size_t e) {
        calls++; seen_b = b; seen_e = e;
    });
    REQUIRE(calls == 1);            // one chunk (below grain -> serial)
    REQUIRE(seen_b == 0);
    REQUIRE(seen_e == 10);
}

TEST_CASE("thread_pool: parallel_reduce sums correctly", "[detail][thread_pool]") {
    for (unsigned nt : {1u, 2u, 4u, 8u}) {
        thread_pool pool(nt);
        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{5},
                              std::size_t{1000}, std::size_t{250000}}) {
            long long total = pool.parallel_reduce<long long>(n, /*grain=*/1024,
                [&](std::size_t b, std::size_t e) {
                    long long s = 0;
                    for (std::size_t i = b; i < e; ++i) s += static_cast<long long>(i);
                    return s;
                });
            REQUIRE(total == static_cast<long long>(n) * (n > 0 ? n - 1 : 0) / 2);
        }
    }
}

TEST_CASE("thread_pool: parallel_reduce below grain runs as one map call", "[detail][thread_pool]") {
    thread_pool pool(8);
    int calls = 0;
    long long total = pool.parallel_reduce<long long>(50, /*grain=*/1024,
        [&](std::size_t b, std::size_t e) {
            calls++;
            long long s = 0; for (std::size_t i = b; i < e; ++i) s += static_cast<long long>(i);
            return s;
        });
    REQUIRE(calls == 1);
    REQUIRE(total == 50LL * 49 / 2);
}

TEST_CASE("thread_pool: nested run falls back to serial (no deadlock)", "[detail][thread_pool]") {
    thread_pool pool(4);
    std::atomic<int> inner{0};
    pool.run(4, [&](unsigned) {
        // A nested parallel region must not deadlock; it runs serially here.
        pool.run(4, [&](unsigned) { inner.fetch_add(1); });
    });
    // 4 outer tids, each running 4 inner iterations serially = 16.
    REQUIRE(inner.load() == 16);
}
