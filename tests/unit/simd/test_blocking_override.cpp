// The blocking diagnostics, built: no other test defines the macros below, so
// these branches of simd/blocking.hpp were compiled by no test (issue 878).
//
//   - MTL5_ENABLE_BLOCKING_OVERRIDE: MTL5_BLOCKING_KC / _MC / _NC from the
//     environment, now parsed as a whole positive std::size_t (it was atol
//     into a long: 32 bits on Windows, and "96x" read as 96);
//   - MTL5_GEMM_MR_OVERRIDE: mr of the compile-time tile;
//   - MTL5_ENABLE_CACHE_DETECTION_L1: configured_detect_levels().
//
// The macros are defined here, ahead of every include, so this test is one
// translation unit with one configuration.
#define MTL5_ENABLE_BLOCKING_OVERRIDE 1
#define MTL5_GEMM_MR_OVERRIDE 5
#define MTL5_ENABLE_CACHE_DETECTION_L1 1

#include <catch2/catch_test_macros.hpp>

#include <mtl/simd/blocking.hpp>

#include <cstddef>
#include <cstdlib>
#include <string>

using namespace mtl::simd;

namespace {

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // namespace

TEST_CASE("env_block_size: a whole positive size_t, else the derived value", "[simd][blocking][878]") {
    struct cases { const char* value; bool valid; std::size_t expected; };
    for (const auto& c : {cases{"96", true, 96}, cases{"1", true, 1}, cases{"4096", true, 4096},
                          cases{"0", false, 0}, cases{"-5", false, 0}, cases{"96x", false, 0},
                          cases{" 96", false, 0}, cases{"+96", false, 0}, cases{"", false, 0},
                          cases{"99999999999999999999999", false, 0}}) {
        INFO("\"" << c.value << "\"");
        set_env("MTL5_TEST_BLOCK", c.value);
        std::size_t v = 7;
        CHECK(detail::env_block_size("MTL5_TEST_BLOCK", v) == c.valid);
        CHECK(v == (c.valid ? c.expected : std::size_t{7}));   // unchanged when invalid
    }
}

TEST_CASE("the blocking overrides reach the blocking parameters", "[simd][blocking][878]") {
    // MTL5_GEMM_MR_OVERRIDE sets mr of the compile-time tile, and nothing else.
    STATIC_REQUIRE(default_blocking<double>.mr == 5);
    STATIC_REQUIRE(default_blocking<float>.mr == 5);
    STATIC_REQUIRE(configured_detect_levels() == detect_levels::l1);

    // runtime_blocking<T>() reads the environment once per T, on first use.
    set_env("MTL5_BLOCKING_KC", "96");
    set_env("MTL5_BLOCKING_MC", "12x");   // invalid: the derived mc stays
    set_env("MTL5_BLOCKING_NC", "-5");    // invalid: the default nc stays
    const blocking_params derived = derive_blocking<double>(width<double>, detected_hw_traits());
    const blocking_params& p = runtime_blocking<double>();
    CHECK(p.kc == 96);
    CHECK(p.mc == derived.mc);
    CHECK(p.nc == default_blocking<double>.nc);
    CHECK(p.mr == 5);
    CHECK(p.nr == default_blocking<double>.nr);
}
