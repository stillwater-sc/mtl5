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
//   * detail::contract_abort(msg) prints msg and aborts in every build, for a
//     broken contract where throwing is not possible (a noexcept move).
//
// With exceptions enabled, behaviour is unchanged.

#include <cstdio>
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

/// A broken contract on a path that cannot throw (a noexcept move): print the
/// message and abort, in every build. For the cases that used to be an assert
/// and, under NDEBUG, carried on doing something wrong (#856).
[[noreturn]] inline void contract_abort(const char* what) noexcept {
    std::fputs("mtl5: ", stderr);
    std::fputs(what, stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    std::abort();
}

} // namespace mtl::detail
