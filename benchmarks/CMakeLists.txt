# MTL5 Benchmark Suite -- Policy-based backend comparison
# Built when MTL5_BUILD_BENCHMARKS=ON

# Benchmark harness headers are included via <benchmarks/harness/...>
# so we add the parent of this directory to the include path.

add_executable(bench_all bench_all.cpp)
target_link_libraries(bench_all PRIVATE MTL5::mtl5)
target_include_directories(bench_all PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
set_target_properties(bench_all PROPERTIES FOLDER "Benchmarks")

# KLU performance scoreboard: native KLU vs SuiteSparse KLU (epic #138, Phase 0).
# Links MTL5::mtl5, which propagates MTL5_HAS_KLU + the KLU library when the
# build is configured with -DMTL5_WITH_SUITESPARSE_KLU=ON; otherwise it builds
# native-only.
add_executable(bench_klu bench_klu.cpp)
target_link_libraries(bench_klu PRIVATE MTL5::mtl5)
target_include_directories(bench_klu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(bench_klu PROPERTIES FOLDER "Benchmarks")

# SuperLU performance scoreboard: native LU vs SuiteSparse SuperLU (epic #186,
# Phase 0 / #180). Links MTL5::mtl5, which propagates MTL5_HAS_SUPERLU + the
# SuperLU library when the build is configured with -DMTL5_WITH_SUPERLU=ON;
# otherwise it builds native-only.
add_executable(bench_superlu bench_superlu.cpp)
target_link_libraries(bench_superlu PRIVATE MTL5::mtl5)
target_include_directories(bench_superlu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(bench_superlu PROPERTIES FOLDER "Benchmarks")

# Sparse triangular-solve scaling scoreboard (#297 phase 2): measures the
# level-scheduled solve() phase of the native sparse Cholesky/LDLT/LU and
# supernodal solvers across MTL5_NUM_THREADS. Native-only; needs no external
# library. Run under benchmarks/run_scaling.sh with SUITE=sparse-solve.
add_executable(bench_sparse bench_sparse.cpp)
target_link_libraries(bench_sparse PRIVATE MTL5::mtl5)
target_include_directories(bench_sparse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(bench_sparse PROPERTIES FOLDER "Benchmarks")

# Cache-blocking A/B (#426 / #430 / #432): the SAME source built twice, once with
# MTL5_ENABLE_CACHE_DETECTION and once without, so the two binaries differ only
# in kc/mc. Driven by benchmarks/run_blocking_ab.sh, which pins and interleaves
# them. The `-default` arm is what MTL5 ships; the `-detected` arm is the opt-in
# under test, which measured a loss on the i7-12700K.
# FOUR arms, because the four-machine result (#430) says kc and mc do not behave
# alike -- the Ryzen run varied mc alone and cost nothing, while both machines
# whose kc moved lost throughput. `kconly` and `mconly` turn that inference into
# a measurement: L1 feeds kc, L2 feeds mc, and each arm detects one of them.
#
# All four are the SAME source; only the macro differs, so any difference in
# throughput is a difference in blocking.
set(_ab_arms      default        detected                        kconly                             mconly)
set(_ab_arm_defs  ""             MTL5_ENABLE_CACHE_DETECTION     MTL5_ENABLE_CACHE_DETECTION_L1     MTL5_ENABLE_CACHE_DETECTION_L2)
foreach(_i RANGE 3)
    list(GET _ab_arms     ${_i} _arm)
    list(GET _ab_arm_defs ${_i} _def)
    add_executable(bench_blocking_ab_${_arm} bench_blocking_ab.cpp)
    target_link_libraries(bench_blocking_ab_${_arm} PRIVATE MTL5::mtl5)
    target_include_directories(bench_blocking_ab_${_arm} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
    set_target_properties(bench_blocking_ab_${_arm} PROPERTIES FOLDER "Benchmarks")
    if(NOT _def STREQUAL "")
        target_compile_definitions(bench_blocking_ab_${_arm} PRIVATE ${_def})
    endif()
    # Provenance is regenerated per build, so the recorded commit follows the
    # working tree rather than the last configure (#442).
    add_dependencies(bench_blocking_ab_${_arm} mtl5_build_info)
endforeach()

# Future: individual focused benchmarks can be added here
# add_executable(bench_gemm bench_gemm.cpp)
# target_link_libraries(bench_gemm PRIVATE MTL5::mtl5)
# target_include_directories(bench_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
