// mtl::parallel::detail::in_region() (#640): true on every tid of a region,
// including one that runs serially because another is in progress; false
// outside a region; left as it was by the serial fast path.
#include <catch2/catch_test_macros.hpp>

#include <mtl/parallel/thread_pool.hpp>

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace par = mtl::parallel;

// A pool of four (or the hardware's size, if smaller), set before first use.
par::thread_pool& pool4() {
    par::configure(4);
    return par::pool();
}
} // namespace

TEST_CASE("parallel: in_region() is false outside a region", "[parallel][thread_pool]") {
    pool4();
    REQUIRE_FALSE(par::detail::in_region());

    // A thread the pool did not start is outside a region too.
    bool other = true;
    std::thread t([&] { other = par::detail::in_region(); });
    t.join();
    REQUIRE_FALSE(other);
}

TEST_CASE("parallel: in_region() is true on every tid of run() and parallel_for",
          "[parallel][thread_pool]") {
    auto& p = pool4();
    if (p.size() < 2) {
        WARN("one hardware thread: the pool is serial, so no region is dispatched");
        return;
    }
    const unsigned n = p.size();

    std::vector<int> seen(n, -1);
    p.run(n, [&](unsigned tid) { seen[tid] = par::detail::in_region() ? 1 : 0; });
    for (unsigned t = 0; t < n; ++t) REQUIRE(seen[t] == 1);
    REQUIRE_FALSE(par::detail::in_region());   // tid 0's flag is restored

    // Only the workers are marked as such: tid 0 is the caller's own thread.
    std::vector<int> worker(n, -1);
    p.run(n, [&](unsigned tid) { worker[tid] = par::detail::on_pool_worker() ? 1 : 0; });
    REQUIRE(worker[0] == 0);
    for (unsigned t = 1; t < n; ++t) REQUIRE(worker[t] == 1);
    REQUIRE_FALSE(par::detail::on_pool_worker());

    std::atomic<int> outside{0};
    p.parallel_for(std::size_t{4096}, std::size_t{64}, [&](std::size_t, std::size_t) {
        if (!par::detail::in_region()) ++outside;
    });
    REQUIRE(outside.load() == 0);
    REQUIRE_FALSE(par::detail::in_region());
}

TEST_CASE("parallel: a region nested in another runs serially with in_region() set",
          "[parallel][thread_pool]") {
    auto& p = pool4();
    if (p.size() < 2) {
        WARN("one hardware thread: the busy-pool fallback is not reached");
        return;
    }
    std::vector<int> inner(p.size() * 2, -1);
    p.run(2, [&](unsigned outer) {
        // The pool is busy, so this runs all tids on the caller, in order.
        p.run(2, [&](unsigned tid) {
            inner[outer * 2 + tid] = par::detail::in_region() ? 1 : 0;
        });
        // The nested region restores the outer one's flag, not false.
        if (!par::detail::in_region()) inner[outer * 2] = 2;
    });
    for (unsigned i = 0; i < 4; ++i) REQUIRE(inner[i] == 1);
}

TEST_CASE("parallel: the serial fast path leaves in_region() as it was",
          "[parallel][thread_pool]") {
    auto& p = pool4();
    // count 1 outside a region: no MTL5 parallelism, so the flag stays false.
    bool flag = true;
    p.run(1, [&](unsigned) { flag = par::detail::in_region(); });
    REQUIRE_FALSE(flag);

    bool body = true;
    p.parallel_for(std::size_t{8}, std::size_t{1024}, [&](std::size_t, std::size_t) {
        body = par::detail::in_region();
    });
    REQUIRE_FALSE(body);   // n < 2 * grain: one chunk, run directly
}

TEST_CASE("parallel: in_region() is restored when a tid throws", "[parallel][thread_pool]") {
    auto& p = pool4();
    REQUIRE_THROWS_AS(p.run(p.size(), [](unsigned) { throw std::runtime_error("tid"); }),
                      std::runtime_error);
    REQUIRE_FALSE(par::detail::in_region());
}
