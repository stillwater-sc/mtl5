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
# `ccap` is the #453 arm: L2 detection PLUS the runtime C-strip bound on mc. It
# is deliberately mconly + the cap, so `ccap` vs `mconly` isolates the cap and
# `ccap` vs `default` answers the shipped question.
# One if-chain rather than two parallel lists: an arm needs TWO defines, and a
# ";"-joined list element is silently FLATTENED into the outer list by CMake, so
# the parallel-list form shifted every later index and handed `ccap` only half
# its configuration. It still built and ran, reporting "L2 only" -- a
# misconfigured arm that looks like a working one, which is the failure this
# whole benchmark tree exists to make impossible.
set(_ab_arms default detected kconly mconly ccap ccap2)
foreach(_arm ${_ab_arms})
    if(_arm STREQUAL "default")
        set(_def "")
    elseif(_arm STREQUAL "detected")
        set(_def MTL5_ENABLE_CACHE_DETECTION)
    elseif(_arm STREQUAL "kconly")
        set(_def MTL5_ENABLE_CACHE_DETECTION_L1)
    elseif(_arm STREQUAL "mconly")
        set(_def MTL5_ENABLE_CACHE_DETECTION_L2)
    elseif(_arm STREQUAL "ccap")
        set(_def MTL5_ENABLE_CACHE_DETECTION_L2 MTL5_GEMM_C_STRIP_CAP)
    elseif(_arm STREQUAL "ccap2")
        # As ccap, but the budget charges the C strip ALONE. The Xeon run says
        # A does not belong in it: every win was a shape where C alone overflowed
        # L2, every loss a shape where C already fitted.
        set(_def MTL5_ENABLE_CACHE_DETECTION_L2 MTL5_GEMM_C_STRIP_CAP
                 MTL5_GEMM_C_STRIP_CAP_C_ONLY)
    else()
        message(FATAL_ERROR "no defines configured for A/B arm '${_arm}'")
    endif()
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

# Blocking response-curve sweep (#453): the same bench_blocking_ab source, built
# with the diagnostic override so kc/mc/nc can be set per run from the
# environment. Separate target so no other binary carries the knob.
# Offline enumeration of where the candidate nc models disagree (#479). Measures
# nothing -- pure functions only -- so it plans a scarce-hardware session instead
# of consuming one.
add_executable(sweep_nc_models sweep_nc_models.cpp)
target_link_libraries(sweep_nc_models PRIVATE MTL5::mtl5)
target_include_directories(sweep_nc_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_dependencies(sweep_nc_models mtl5_build_info)

# The timing half of #479: what the disagreement the sweep found is WORTH.
# Shares nc_shapes.hpp with the sweep, so the shapes nominated there are the
# shapes measured here -- by construction, not by two lists agreeing.
add_executable(bench_nc_models bench_nc_models.cpp)
target_link_libraries(bench_nc_models PRIVATE MTL5::mtl5)
target_include_directories(bench_nc_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(bench_nc_models PROPERTIES FOLDER "Benchmarks")
add_dependencies(bench_nc_models mtl5_build_info)

add_executable(bench_blocking_sweep bench_blocking_ab.cpp)
target_link_libraries(bench_blocking_sweep PRIVATE MTL5::mtl5)
target_include_directories(bench_blocking_sweep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_definitions(bench_blocking_sweep PRIVATE MTL5_ENABLE_BLOCKING_OVERRIDE)
set_target_properties(bench_blocking_sweep PROPERTIES FOLDER "Benchmarks")
add_dependencies(bench_blocking_sweep mtl5_build_info)

# Future: individual focused benchmarks can be added here
# add_executable(bench_gemm bench_gemm.cpp)
# target_link_libraries(bench_gemm PRIVATE MTL5::mtl5)
# target_include_directories(bench_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
