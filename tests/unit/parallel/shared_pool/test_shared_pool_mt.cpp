// One thread pool across shared libraries (#639): two hidden-visibility
// modules that define MTL5_SHARED_POOL, and this executable, all use the pool
// in the pool library; configure() through one module sizes it for all. A
// third hidden module without the macro shows the problem being fixed: it has
// a pool of its own.
#include <catch2/catch_test_macros.hpp>

#include "plugin.hpp"

DECLARE_PLUGIN(shared_a)
DECLARE_PLUGIN(shared_b)
DECLARE_PLUGIN(local)

TEST_CASE("parallel: modules that define MTL5_SHARED_POOL share one pool",
          "[parallel][thread_pool][shared_pool]") {
    // Nothing has created the pool yet. Configure it through module a, then
    // module b: both write the one configuration, and the later call wins.
    REQUIRE(shared_a_configure(2));
    REQUIRE(shared_b_configure(3));
    const unsigned want = mtl::parallel::detail::clamp_pool_threads(3);

    const void* const pool = &mtl::parallel::pool();
    REQUIRE(shared_a_pool() == pool);
    REQUIRE(shared_b_pool() == pool);
    REQUIRE(mtl::parallel::pool().size() == want);
    REQUIRE(shared_a_size() == want);
    REQUIRE(shared_b_size() == want);

    // The pool exists now, for every module.
    REQUIRE_FALSE(shared_b_configure(4));
    REQUIRE_FALSE(mtl::parallel::configure(4));

    // Module a's region runs on the pool's workers.
    REQUIRE(shared_a_distinct_threads() == want);

    if (want >= 2) {
        // A region module b starts from inside a region this executable
        // started finds the pool busy, and runs serially: the modules share
        // the pool's state, not just its address.
        bool serial = false;
        mtl::parallel::pool().run(2, [&](unsigned tid) {
            if (tid == 1) serial = shared_b_region_ran_serially();
        });
        REQUIRE(serial);
    } else {
        WARN("one hardware thread: the busy-pool check is skipped");
    }

    // Without MTL5_SHARED_POOL, a hidden-visibility module has its own pool.
    REQUIRE(local_pool() != pool);
}
