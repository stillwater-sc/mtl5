#pragma once
// MTL5 benchmark harness -- the binary's half of a result's sidecar (#442, #930)
//
// A sidecar (<result>.sysinfo) records what produced a result: the build and
// the machine, which only the binary can see, and the run (benchmark, machine
// name, class, pinning, clock policy, preflight), which the harness script adds
// (benchmarks/sidecar.sh). benchmarks/sidecar_keys.txt lists the keys every
// sidecar must carry, and check_sidecars.sh enforces it.
//
// write_build_keys writes the binary's keys, the same in every benchmark:
//   cpu_*, build_isa, os_*, compiler, compiler_version, cpp_standard,
//   build_type (system_info::to_keyvals); simd_backend; git_commit, git_dirty,
//   cxx_flags, cmake_build_type (the build); pool_size.
// pool_size is the thread pool the measured kernels used: 1 for a harness that
// runs single-threaded, which is a value, not an omission (#930).
#include <cstddef>
#include <ostream>

#include <mtl/build_info.hpp>
#include <mtl/simd/batch.hpp>
#include <mtl/util/system_info.hpp>

namespace mtl::bench {

inline void write_build_keys(std::ostream& os, std::size_t pool_size) {
    os << mtl::util::to_keyvals(mtl::util::identify())
       << "simd_backend="     << mtl::simd::backend_name()  << "\n"
       << "git_commit="       << mtl::build_git_commit      << "\n"
       << "git_dirty="        << mtl::build_git_dirty       << "\n"
       << "cxx_flags="        << mtl::build_cxx_flags       << "\n"
       << "cmake_build_type=" << mtl::build_cmake_type      << "\n"
       << "pool_size="        << pool_size                  << "\n";
}

} // namespace mtl::bench
