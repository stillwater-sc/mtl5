# Cross-compile MTL5 for aarch64 Linux, and run its tests under QEMU user mode.
#
# This lets an x86-64 Linux host build and RUN the aarch64 code paths: NEON,
# fixed-length SVE (SVE2_128, SVE_256) and scalable SVE at any vector length
# from 128 to 2048 bits -- widths no CI runner provides (the ubuntu-24.04-arm
# runners are Neoverse N2 at 128 bits). It is for CORRECTNESS only: emulated
# timings say nothing about real hardware, so mtl5_bench's numbers under it
# are meaningless (its correctness checks are not).
#
# Host requirements (Debian/Ubuntu):
#   sudo apt install g++-aarch64-linux-gnu qemu-user
#
# Use it through a preset (CMakePresets.json: aarch64-neon, aarch64-sve2-128,
# aarch64-sve-256, aarch64-sve2-scalable), or directly:
#   cmake -B build-aarch64 \
#         -DCMAKE_TOOLCHAIN_FILE=tools/cmake/toolchains/aarch64-linux-gnu.cmake \
#         -DMTL5_QEMU_CPU=neoverse-n2
#   cmake --build build-aarch64 -j4 && ctest --test-dir build-aarch64
#
# ctest runs every test through QEMU automatically: add_test() prefixes a
# target command with CMAKE_CROSSCOMPILING_EMULATOR.
#
# Cache variables:
#   MTL5_QEMU_CPU          the emulated CPU, which also fixes the SVE vector
#                          length (default neoverse-n2, 128-bit):
#                            neoverse-n2  SVE2, 128     neoverse-v1  SVE, 256
#                            a64fx        SVE, 512      max          SVE2, 512
#                            max,sve-default-vector-length=<bytes>  any width
#                          A binary built with -msve-vector-bits=N is correct
#                          ONLY at that width: pick the CPU to match.
#   MTL5_AARCH64_SYSROOT   the target sysroot (default /usr/aarch64-linux-gnu)
#   MTL5_AARCH64_CLANG     ON to use clang/clang++ --target=aarch64-linux-gnu
#                          instead of aarch64-linux-gnu-gcc/g++

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(MTL5_AARCH64_SYSROOT "/usr/aarch64-linux-gnu" CACHE PATH "aarch64 target sysroot")
set(MTL5_QEMU_CPU "neoverse-n2" CACHE STRING "CPU model QEMU emulates (fixes the SVE vector length)")
option(MTL5_AARCH64_CLANG "Cross-compile with clang instead of aarch64-linux-gnu-gcc" OFF)

if(MTL5_AARCH64_CLANG)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
    set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
else()
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
endif()

# Look for libraries and packages in the target sysroot only. In particular this
# keeps find_package(hwy) from picking up an x86-64 Highway installed on the
# host: MTL5 then fetches and cross-builds the pinned Highway instead.
set(CMAKE_FIND_ROOT_PATH "${MTL5_AARCH64_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# -L points QEMU's dynamic loader at the sysroot. Without it, running a binary
# fails with "Could not open '/lib/ld-linux-aarch64.so.1'" even when binfmt
# is registered, because the host has no aarch64 loader there.
find_program(MTL5_QEMU_AARCH64 qemu-aarch64)
if(MTL5_QEMU_AARCH64)
    set(CMAKE_CROSSCOMPILING_EMULATOR
        "${MTL5_QEMU_AARCH64};-L;${MTL5_AARCH64_SYSROOT};-cpu;${MTL5_QEMU_CPU}")
else()
    message(WARNING "qemu-aarch64 not found: the build works, but ctest cannot run "
                    "the aarch64 tests on this host (sudo apt install qemu-user)")
endif()
