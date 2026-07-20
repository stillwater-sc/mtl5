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
