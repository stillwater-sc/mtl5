// mtl::parallel::configure() (#630). A test executable of its own, with a
// single test case, because configure() acts only before the process-wide
// pool's first use and nothing else here may touch the pool first.
#include <catch2/catch_test_macros.hpp>

#include <mtl/parallel/thread_pool.hpp>

#include <cstdlib>
#include <thread>

namespace {
unsigned clamped(unsigned n) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    return n < hw ? n : hw;
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
} // namespace

TEST_CASE("parallel: configure() sizes the pool before first use, and only then",
          "[parallel][thread_pool]") {
    // configure() overrides the environment, which asks for serial here.
    set_env("MTL5_NUM_THREADS", "1");

    REQUIRE_FALSE(mtl::parallel::configure(0));   // not a size
    REQUIRE(mtl::parallel::configure(100000));    // clamped, like the variable
    REQUIRE(mtl::parallel::configure(2));         // the last call before first use wins

    auto& pool = mtl::parallel::pool();
    REQUIRE(pool.size() == clamped(2));

    // After first use, configure() changes nothing and says so.
    REQUIRE_FALSE(mtl::parallel::configure(3));
    REQUIRE(mtl::parallel::pool().size() == clamped(2));
    REQUIRE(&mtl::parallel::pool() == &pool);
}
