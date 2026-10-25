#pragma once
// One module of the shared-pool test (#639). Each plugin_*.cpp defines
// PLUGIN(name), the prefix of its exports, and includes this; the test
// executable includes it without PLUGIN to declare the exports it calls.
#include <mtl/parallel/thread_pool.hpp>
#include <mtl/parallel/spin_pool.hpp>

#include <algorithm>
#include <thread>
#include <vector>

#if defined(_WIN32) || defined(__CYGWIN__)
#  define PLUGIN_EXPORT extern "C" __declspec(dllexport)
#  define PLUGIN_IMPORT extern "C" __declspec(dllimport)
#else
#  define PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#  define PLUGIN_IMPORT extern "C"
#endif

#define DECLARE_PLUGIN(p)                                  \
    PLUGIN_IMPORT const void* p##_pool();                  \
    PLUGIN_IMPORT bool p##_configure(unsigned threads);    \
    PLUGIN_IMPORT unsigned p##_size();                     \
    PLUGIN_IMPORT unsigned p##_distinct_threads();         \
    PLUGIN_IMPORT bool p##_region_ran_serially();       \
    PLUGIN_IMPORT bool p##_in_region();                 \
    PLUGIN_IMPORT const void* p##_spin_pool();

#if defined(PLUGIN)

/// The pool this module's pool() returns.
PLUGIN_EXPORT const void* PLUGIN(pool)() { return &mtl::parallel::pool(); }

PLUGIN_EXPORT bool PLUGIN(configure)(unsigned threads) {
    return mtl::parallel::configure(threads);
}

PLUGIN_EXPORT unsigned PLUGIN(size)() { return mtl::parallel::pool().size(); }

/// Runs a region over the whole pool; returns how many threads ran it.
PLUGIN_EXPORT unsigned PLUGIN(distinct_threads)() {
    mtl::parallel::thread_pool& p = mtl::parallel::pool();
    std::vector<std::thread::id> ids(p.size());
    p.run(p.size(), [&](unsigned tid) { ids[tid] = std::this_thread::get_id(); });
    std::sort(ids.begin(), ids.end());
    return static_cast<unsigned>(std::unique(ids.begin(), ids.end()) - ids.begin());
}

/// Runs a two-tid region; returns whether both tids ran on the calling
/// thread, which is what a region started while the pool is busy does.
PLUGIN_EXPORT bool PLUGIN(region_ran_serially)() {
    const std::thread::id me = std::this_thread::get_id();
    std::thread::id ran[2];
    mtl::parallel::pool().run(2, [&](unsigned tid) { ran[tid] = std::this_thread::get_id(); });
    return ran[0] == me && ran[1] == me;
}

/// This module's view of the calling thread's in-region flag (#640).
PLUGIN_EXPORT bool PLUGIN(in_region)() { return mtl::parallel::detail::in_region(); }

/// The process spin_pool this module sees (#636).
PLUGIN_EXPORT const void* PLUGIN(spin_pool)() { return &mtl::parallel::spin_pool::instance(); }

#endif
