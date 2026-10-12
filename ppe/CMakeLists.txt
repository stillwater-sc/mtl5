# PPE -- Platform Performance Engineering
#
# Standalone from the MTL5 library: these kernels are written from scratch so
# each optimization step is visible and attributable. Nothing here includes
# <mtl/...>; the point is the progression, not the library.
cmake_minimum_required(VERSION 3.22...4.2)
project(ppe CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(ppe_gemm src/ppe_gemm.cpp)
target_include_directories(ppe_gemm PRIVATE include)

# -march=native matters here: without it the compiler targets a baseline ISA and
# the vectorizable microkernel cannot use FMA or the wider registers, which is
# most of what v5 is measuring. Non-portable binaries, which is fine -- these are
# measurements of THIS machine.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march=native PPE_HAS_MARCH_NATIVE)
if(PPE_HAS_MARCH_NATIVE)
    target_compile_options(ppe_gemm PRIVATE -march=native)
endif()
target_compile_options(ppe_gemm PRIVATE -O3)
