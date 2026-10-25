// One thread pool across shared libraries (#639): two hidden-visibility
// modules that define MTL5_SHARED_POOL, and this executable, all use the pool
// in the pool library; configure() through one module sizes it for all. A
// third hidden module without the macro shows the problem being fixed: it has
// a pool of its own. Two default-visible modules, one per mode, check that
// the modes' exported inline definitions do not bind to each other.
#include <catch2/catch_test_macros.hpp>

#include "plugin.hpp"

DECLARE_PLUGIN(shared_a)
DECLARE_PLUGIN(shared_b)
DECLARE_PLUGIN(local)
DECLARE_PLUGIN(shared_default)
DECLARE_PLUGIN(local_default)

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

        // The in-region flag is the process's too (#640): a task this
        // executable runs sees it set through module b, while a module with a
        // pool of its own reads its own, unset, flag.
        bool b_in = false, local_in = true;
        mtl::parallel::pool().run(2, [&](unsigned tid) {
            if (tid == 1) { b_in = shared_b_in_region(); local_in = local_in_region(); }
        });
        REQUIRE(b_in);
        REQUIRE_FALSE(local_in);
        REQUIRE_FALSE(shared_a_in_region());
    } else {
        WARN("one hardware thread: the busy-pool check is skipped");
    }

    // Without MTL5_SHARED_POOL, a hidden-visibility module has its own pool.
    REQUIRE(local_pool() != pool);
}

TEST_CASE("parallel: default-visible modules of both modes keep apart",
          "[parallel][thread_pool][shared_pool]") {
    // Both modules export their inline pool functions and statics, the local
    // one first in the search order. Under one set of names the dynamic linker
    // would bind both to one definition: a reference static read as an object,
    // or one mode's configure() writing the other's configuration.
    const void* const pool = &mtl::parallel::pool();
    REQUIRE(shared_default_pool() == pool);
    REQUIRE(shared_default_size() == mtl::parallel::pool().size());
    REQUIRE_FALSE(shared_default_configure(1));   // the shared pool exists

    // The local module's configuration is its own, and still open.
    REQUIRE(local_default_configure(1));
    REQUIRE(local_default_size() == 1u);
    REQUIRE(local_default_pool() != pool);
    REQUIRE(local_default_pool() != local_pool());
}
