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

# Block-scaled integer gemv (#628): per-block reduce_dot_widen vs dequantize +
# float dot vs reduce_dot_block_scaled, at block 32 and 256. Native-only; the
# result is only meaningful with MTL5_WITH_HIGHWAY=ON and a -march that names
# the target being measured.
add_executable(bench_block_scaled_dot bench_block_scaled_dot.cpp)
target_link_libraries(bench_block_scaled_dot PRIVATE MTL5::mtl5)
target_include_directories(bench_block_scaled_dot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(bench_block_scaled_dot PROPERTIES FOLDER "Benchmarks")

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
# `tileN` and `isaregs` are the #643 register-tile arms. `tileN` sets mr = N and
# nothing else (MTL5_GEMM_MR_OVERRIDE), so N in {6,8,10,12,14,16} against `default`
# is a tile sweep at fixed nr, kc, mc and nc.
# `isaregs` is the rejected candidate, kept measurable: the target ISA's full
# register file at the 3/4 rule (MTL5_GEMM_TARGET_REGISTER_FILE), which is 12xnr
# on NEON and AVX-512 and identical to `default` on AVX2. The sweep put NEON's
# peak at 8xnr, which is what aarch64 now ships; see docs/benchmarks/systems.md.
# On aarch64, `default` is mr = 8, so `tile8` equals it there and `tile6` is the
# pre-#643 tile.
set(_ab_tile_arms tile6 tile8 tile10 tile12 tile14 tile16)
set(_ab_arms default detected kconly mconly ccap ccap2 legacymc isaregs ${_ab_tile_arms})
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
    elseif(_arm STREQUAL "legacymc")
        # The pre-#458 balanced_mc: the threaded ic step chosen by block count,
        # not by the busiest thread's micro-panels. `default` vs `legacymc` is the
        # #458 before/after, from one tree in one session.
        set(_def MTL5_GEMM_LEGACY_BALANCED_MC)
    elseif(_arm STREQUAL "isaregs")
        set(_def MTL5_GEMM_TARGET_REGISTER_FILE)
    elseif(_arm IN_LIST _ab_tile_arms)
        string(SUBSTRING "${_arm}" 4 -1 _mr)
        set(_def MTL5_GEMM_MR_OVERRIDE=${_mr})
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

# The i-k-j generic GEMM (#549) and the CRS x dense product (#550): one source
# built against the headers before and after, compared by
# docs/performance/ikj-gemm-and-crs-dense.md. fig_mult_ikj draws that page's
# non-zero structure and access-pattern figures.
add_executable(bench_mult_ikj bench_mult_ikj.cpp)
target_link_libraries(bench_mult_ikj PRIVATE MTL5::mtl5)
set_target_properties(bench_mult_ikj PROPERTIES FOLDER "Benchmarks")

add_executable(fig_mult_ikj fig_mult_ikj.cpp)
target_link_libraries(fig_mult_ikj PRIVATE MTL5::mtl5)
set_target_properties(fig_mult_ikj PROPERTIES FOLDER "Benchmarks")

# The accuracy architecture (#596): the cost of the accumulator paths of
# lu_factor and cholesky_factor, and the loop-order figures of
# docs/architecture/accuracy-architecture.md.
add_executable(bench_accumulator_factor bench_accumulator_factor.cpp)
target_link_libraries(bench_accumulator_factor PRIVATE MTL5::mtl5)
set_target_properties(bench_accumulator_factor PROPERTIES FOLDER "Benchmarks")

add_executable(fig_loop_orders fig_loop_orders.cpp)
target_link_libraries(fig_loop_orders PRIVATE MTL5::mtl5)
set_target_properties(fig_loop_orders PROPERTIES FOLDER "Benchmarks")

# A vendor BLAS called from inside an MTL5 region (#640): the batched-dgemm
# region with the vendor uncapped, capped, and called directly. Meaningful only
# with MTL5_WITH_BLAS=ON; benchmarks/run_vendor_nested.sh builds one tree per
# vendor.
add_executable(bench_vendor_nested bench_vendor_nested.cpp)
target_link_libraries(bench_vendor_nested PRIVATE MTL5::mtl5)
set_target_properties(bench_vendor_nested PROPERTIES FOLDER "Benchmarks")
add_dependencies(bench_vendor_nested mtl5_build_info)

# Dispatch and join latency of thread_pool and spin_pool (#636), as
# percentiles; benchmarks/run_pool_latency.sh runs every arm in its own process.
add_executable(bench_pool_latency bench_pool_latency.cpp)
target_link_libraries(bench_pool_latency PRIVATE MTL5::mtl5)
set_target_properties(bench_pool_latency PROPERTIES FOLDER "Benchmarks")
add_dependencies(bench_pool_latency mtl5_build_info)

# Future: individual focused benchmarks can be added here
# add_executable(bench_gemm bench_gemm.cpp)
# target_link_libraries(bench_gemm PRIVATE MTL5::mtl5)
# target_include_directories(bench_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
