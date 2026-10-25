#pragma once
// MTL5 -- the definition of the process's one thread pool (#639).
//
// Include this from exactly one translation unit of one shared library,
// compiled with MTL5_SHARED_POOL and MTL5_SHARED_POOL_BUILD defined; every
// module that links that library and defines MTL5_SHARED_POOL then uses its
// pool (see <mtl/parallel/thread_pool.hpp>). The mtl5_pool library that the
// MTL5_SHARED_POOL_LIBRARY CMake option builds is this file and nothing else;
// include it yourself only when building without MTL5's CMake.
#if !defined(MTL5_SHARED_POOL)
#  error "shared_pool_definition.hpp needs MTL5_SHARED_POOL defined"
#endif
#if (defined(_WIN32) || defined(__CYGWIN__)) && !defined(MTL5_SHARED_POOL_BUILD)
#  error "shared_pool_definition.hpp needs MTL5_SHARED_POOL_BUILD defined, so the pool is exported from the DLL"
#endif

#include <mtl/parallel/thread_pool.hpp>
#include <mtl/parallel/spin_pool.hpp>

// Qualified definitions, so they land in the shared mode's inline namespace.
namespace mtl::parallel {

MTL5_SHARED_POOL_API detail::pool_config& detail::shared_pool_config_v1() {
    static pool_config c;
    return c;
}

MTL5_SHARED_POOL_API thread_pool& detail::shared_pool_v1() {
    static thread_pool pool(claim_pool_size());
    return pool;
}

// Constructed only if a module uses it: under MTL5_POOL_SPIN, or by calling
// spin_pool::instance() (#636).
MTL5_SHARED_POOL_API spin_pool& detail::shared_spin_pool_v1() {
    static spin_pool pool(claim_pool_size(), resolve_pool_spin(spin_pool::default_spin));
    return pool;
}

MTL5_SHARED_POOL_API detail::region_thread& detail::shared_region_thread_v1() noexcept {
    thread_local region_thread t{};
    return t;
}

} // namespace mtl::parallel
