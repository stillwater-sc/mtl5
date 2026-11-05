#pragma once
// MTL5 -- building with exceptions disabled (#652)
//
// MTL5 throws on contract violations (an index out of range under bounds
// checking, a resize of a fixed-size container) and the thread pool forwards a
// task's exception to the caller of run(). Firmware and other embedded targets
// build with -fno-exceptions (/EHs- on MSVC), where any `try`, `catch` or
// `throw` in a header fails to compile. Code on such a path goes through this
// header instead:
//
//   * MTL5_HAS_EXCEPTIONS is 1 when the compiler has exceptions enabled, else 0.
//     Define it yourself before including MTL5 to override the detection.
//   * detail::throw_or_abort(e) throws `e` when exceptions are enabled. When
//     they are not, it calls std::abort(), which is what an uncaught exception
//     would have done.
//
// With exceptions enabled, behaviour is unchanged.

#include <cstdlib>
#include <utility>

#if !defined(MTL5_HAS_EXCEPTIONS)
#  if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#    define MTL5_HAS_EXCEPTIONS 1
#  else
#    define MTL5_HAS_EXCEPTIONS 0
#  endif
#endif

namespace mtl::detail {

/// Throw `e`; with exceptions disabled, abort instead.
template <typename E>
[[noreturn]] inline void throw_or_abort(E&& e) {
#if MTL5_HAS_EXCEPTIONS
    throw std::forward<E>(e);
#else
    (void)e;
    std::abort();
#endif
}

} // namespace mtl::detail
