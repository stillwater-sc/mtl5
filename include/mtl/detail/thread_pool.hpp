#pragma once
// MTL5 -- compatibility header for the thread pool (#221).
//
// The pool is public API since #630: <mtl/parallel/thread_pool.hpp>, as
// mtl::parallel::thread_pool, with mtl::parallel::pool() and
// mtl::parallel::configure(). These names stay so existing callers, MTL5's own
// kernels among them, keep compiling unchanged; new code should use the public
// header.
#include <mtl/parallel/thread_pool.hpp>

namespace mtl::detail {

using thread_pool = mtl::parallel::thread_pool;
using mtl::parallel::detail::resolve_pool_threads;

} // namespace mtl::detail
