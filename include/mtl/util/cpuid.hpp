#pragma once
// MTL5 -- minimal portable CPUID wrapper.
//
// Factored out of system_info.hpp (#222) so cache_info.hpp can issue CPUID leaves
// without either header defining its own `cpuidex`: both live in `mtl::util`, so
// two inline definitions of the same signature would collide in any translation
// unit that included both.
//
// x86 only. `MTL5_HAS_X86_CPUID` is 0 on every other ISA, where the whole block
// (including <cpuid.h> / <intrin.h>) disappears.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define MTL5_HAS_X86_CPUID 1
#else
#  define MTL5_HAS_X86_CPUID 0
#endif

#if MTL5_HAS_X86_CPUID
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif

namespace mtl::util {

/// Fill regs[eax, ebx, ecx, edx] for CPUID `leaf`/`subleaf`.
inline void cpuidex(int leaf, int subleaf, unsigned regs[4]) {
#  if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, leaf, subleaf);
    regs[0] = static_cast<unsigned>(r[0]);
    regs[1] = static_cast<unsigned>(r[1]);
    regs[2] = static_cast<unsigned>(r[2]);
    regs[3] = static_cast<unsigned>(r[3]);
#  else
    unsigned a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    regs[0] = a; regs[1] = b; regs[2] = c; regs[3] = d;
#  endif
}

} // namespace mtl::util
#endif // MTL5_HAS_X86_CPUID
