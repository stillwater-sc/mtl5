# Auto-discover test sources and create Catch2 test targets per category.
# Each .cpp file becomes target "${prefix}_<filename_stem>" and a CTest entry.

file(GLOB CONCEPT_SRC "concepts/*.cpp")
compile_all("true" "concept" "Tests/concepts" "MTL5::mtl5;Catch2::Catch2WithMain" ${CONCEPT_SRC})

file(GLOB SIMD_SRC "simd/*.cpp")
compile_all("true" "simd" "Tests/simd" "MTL5::mtl5;Catch2::Catch2WithMain" ${SIMD_SRC})

file(GLOB MATH_SRC "math/*.cpp")
compile_all("true" "math" "Tests/math" "MTL5::mtl5;Catch2::Catch2WithMain" ${MATH_SRC})

file(GLOB DETAIL_SRC "detail/*.cpp")
compile_all("true" "detail" "Tests/detail" "MTL5::mtl5;Catch2::Catch2WithMain" ${DETAIL_SRC})

file(GLOB MAT_SRC "mat/*.cpp")
compile_all("true" "mat" "Tests/mat" "MTL5::mtl5;Catch2::Catch2WithMain" ${MAT_SRC})

file(GLOB VEC_SRC "vec/*.cpp")
compile_all("true" "vec" "Tests/vec" "MTL5::mtl5;Catch2::Catch2WithMain" ${VEC_SRC})

file(GLOB OP_SRC "operation/*.cpp")
compile_all("true" "op" "Tests/operation" "MTL5::mtl5;Catch2::Catch2WithMain" ${OP_SRC})

file(GLOB FUNCTOR_SRC "functor/*.cpp")
compile_all("true" "functor" "Tests/functor" "MTL5::mtl5;Catch2::Catch2WithMain" ${FUNCTOR_SRC})

file(GLOB ITL_SRC "itl/*.cpp")
compile_all("true" "itl" "Tests/itl" "MTL5::mtl5;Catch2::Catch2WithMain" ${ITL_SRC})

file(GLOB IO_SRC "io/*.cpp")
compile_all("true" "io" "Tests/io" "MTL5::mtl5;Catch2::Catch2WithMain" ${IO_SRC})

file(GLOB RECURSION_SRC "recursion/*.cpp")
compile_all("true" "recursion" "Tests/recursion" "MTL5::mtl5;Catch2::Catch2WithMain" ${RECURSION_SRC})

file(GLOB GENERATOR_SRC "generators/*.cpp")
compile_all("true" "generators" "Tests/generators" "MTL5::mtl5;Catch2::Catch2WithMain" ${GENERATOR_SRC})

file(GLOB INTERFACE_SRC "interface/*.cpp")
compile_all("true" "iface" "Tests/interface" "MTL5::mtl5;Catch2::Catch2WithMain" ${INTERFACE_SRC})

file(GLOB SPARSE_SRC "sparse/*.cpp")
compile_all("true" "sparse" "Tests/sparse" "MTL5::mtl5;Catch2::Catch2WithMain" ${SPARSE_SRC})

# The threaded solve tests assert BIT-IDENTITY against their serial counterparts
# (#381). That claim is about ACCUMULATION ORDER -- the level-scheduled gather
# visits each row's entries in the same increasing-column order the serial CSC
# scatter does -- and it is true at the source level.
#
# FP contraction does not change that order; it fuses a multiply and an add. But
# it is applied per-expression, so the compiler may contract one of the two loop
# nests and not the other, and the results then differ in the last ulp. Measured
# under -O3 -march=native: half the entries differ, by ~1 eps relative to the
# solution norm. With contraction off, 0 entries differ on both GCC and Clang.
#
# So these targets pin contraction, which is what makes the order claim testable
# rather than a claim about the optimizer. MSVC needs nothing: /fp:precise is the
# default and does not contract across statements.
#
# The invariants that hold under ANY flags -- determinism across repeated
# threaded runs, and agreement with the serial solve to a measured bound -- are
# asserted inside the tests themselves and are not affected by this.
#
# op_test_mult_ikj makes the same kind of claim for the i-k-j GEMM and the CRS x
# dense product (#549, #550): bit-identical to the i-j-k loop because each C(r,c)
# sees the same terms in the same order. Under -march=x86-64-v3 or on AArch64 the
# i-k-j loop contracts every term to an FMA while the i-j-k reference contracts
# only some, so the bits differ even though the order does not.
#
# op_test_cholesky_ordering compares cholesky_factor's term-by-term ordering
# (#556) with a reference loop written the same way, bit for bit. That claim is
# about the order of operations too, and needs the same flag.
#
# op_test_householder_textbook does the same for the textbook Householder
# reflector and qr_factor (#559).
#
# op_test_mult_acc_panel_mt asserts that the packed-panel accumulator GEMM (#587)
# is bit-identical to the dot-product kernel it replaced. The packed kernel
# vectorizes across columns, so on an FMA target it can contract acc += a*b
# where the scalar reference does not.
#
# op_test_accumulation_configs_mt compares configuration 1 (the product rounds,
# then the sum) with configuration 2 (fma_accumulator) (#586). Contracting
# acc += a*b turns the first into the second, and the comparison measures nothing.
# op_test_accumulation_solvers_mt runs the same configurations through the solvers.
#
# op_test_dense_direct_accumulator asserts that a plain accumulator in the
# element type gives the void dense factorizations bit for bit (#585): the same
# products in the same order, which contraction can fuse in one and not the other.
#
# sparse_test_solve_accumulator_policy asserts that the level-scheduled solves,
# with the default and with an `Acc == Value` accumulator (#535), are bit-identical
# to the dense_* scatter references -- the same order-of-operations claim as
# trisolve_level_schedule_mt, broken the same way by contraction.
if(NOT MSVC)
    foreach(mt_target sparse_test_supernodal_ldlt_mt
                      sparse_test_supernodal_lu_mt
                      sparse_test_trisolve_level_schedule_mt
                      op_test_mult_ikj
                      op_test_cholesky_ordering
                      op_test_householder_textbook
                      op_test_mult_acc_panel_mt
                      op_test_accumulation_configs_mt
                      op_test_accumulation_solvers_mt
                      op_test_dense_direct_accumulator
                      sparse_test_solve_accumulator_policy)
        if(TARGET ${mt_target})
            target_compile_options(${mt_target} PRIVATE -ffp-contract=off)
        endif()
    endforeach()
endif()

file(GLOB ARRAY_SRC "array/*.cpp")
compile_all("true" "array" "Tests/array" "MTL5::mtl5;Catch2::Catch2WithMain" ${ARRAY_SRC})

file(GLOB TENSOR_SRC "tensor/*.cpp")
compile_all("true" "tensor" "Tests/tensor" "MTL5::mtl5;Catch2::Catch2WithMain" ${TENSOR_SRC})

file(GLOB UTIL_SRC "util/*.cpp")
compile_all("true" "util" "Tests/util" "MTL5::mtl5;Catch2::Catch2WithMain" ${UTIL_SRC})

# -- Provenance escaping probe (#443) ----------------------------------------
# util/test_provenance_escaping.cpp asserts that a value survives cmake/
# GitInfo.cmake byte for byte. Generate the header it checks, from a value
# carrying every hazard at once: Windows backslashes (which become bogus \f and
# \b escapes), an embedded quote (which ends the literal early), and CRLF plus a
# bare newline (which an earlier implementation discarded).
#
# The expected bytes are handed to the test through string(HEX), which has no
# backslash handling of its own, so the expectation cannot be dragged along by a
# broken escaper.
#
# This ran as a `cmake -P` script that invoked ${CMAKE_CXX_COMPILER} directly.
# That cannot work under MSVC -- cl.exe needs the developer environment, which
# the ctest step does not have -- so it failed on the one platform whose flags
# and CRLF line endings motivate the escaping in the first place. Compiling it as
# an ordinary test target uses whatever toolchain the build already configured.
set(_prov_dir ${CMAKE_CURRENT_BINARY_DIR}/provenance_probe)
set(_prov_hostile "/I C:\\foo\\bar /D S=\"q\"\r\nsecond\nthird -Wall")
execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DSRC_DIR=${PROJECT_SOURCE_DIR}
            -DTEMPLATE=${PROJECT_SOURCE_DIR}/include/mtl/build_info.hpp.in
            -DOUTPUT=${_prov_dir}/build_info.hpp
            -DCXX_FLAGS=${_prov_hostile}
            -DCMAKE_TYPE=Release
            -P ${PROJECT_SOURCE_DIR}/cmake/GitInfo.cmake
    RESULT_VARIABLE _prov_rc
    OUTPUT_QUIET ERROR_VARIABLE _prov_err)
if(NOT _prov_rc EQUAL 0)
    message(FATAL_ERROR "GitInfo.cmake failed for the provenance probe: ${_prov_err}")
endif()

string(HEX "${_prov_hostile}" _prov_hex)
string(REGEX REPLACE "(..)" "0x\\1," _prov_bytes "${_prov_hex}")
file(WRITE ${_prov_dir}/expected.hpp
     "// Generated by tests/unit/CMakeLists.txt -- do not edit.\n"
     "#pragma once\n"
     "#define MTL5_TEST_PROVENANCE_EXPECTED_BYTES ${_prov_bytes}\n")

target_include_directories(util_test_provenance_escaping PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# -- Compile-failure tests ---------------------------------------------------
# Each source in compile_fail/ MUST NOT compile. The targets are
# EXCLUDE_FROM_ALL and are built on demand by ctest, so a normal build is
# unaffected.
#
# These use PASS_REGULAR_EXPRESSION rather than WILL_FAIL deliberately: WILL_FAIL
# passes whenever the build returns non-zero, so a typo in the test source would
# make it pass green while proving nothing. Matching the expected diagnostic
# instead means the test only passes when compilation fails for the RIGHT reason.
# Each source declares that diagnostic in an "// EXPECT-ERROR: <regex>" line.
if(CMAKE_CONFIGURATION_TYPES)
    set(_cf_config --config $<CONFIG>)
else()
    set(_cf_config "")
endif()

file(GLOB COMPILE_FAIL_SRC "compile_fail/*.cpp")
foreach(cf_src ${COMPILE_FAIL_SRC})
    get_filename_component(cf_stem ${cf_src} NAME_WE)
    set(cf_target compile_fail_${cf_stem})

    file(STRINGS ${cf_src} cf_expect REGEX "^// EXPECT-ERROR: ")
    if(NOT cf_expect)
        message(FATAL_ERROR
            "${cf_src} has no '// EXPECT-ERROR: <regex>' line; a compile-failure "
            "test without one cannot verify why it failed.")
    endif()
    string(REGEX REPLACE "^// EXPECT-ERROR: " "" cf_expect "${cf_expect}")

    add_executable(${cf_target} EXCLUDE_FROM_ALL ${cf_src})
    target_link_libraries(${cf_target} PRIVATE MTL5::mtl5)
    set_target_properties(${cf_target} PROPERTIES FOLDER "Tests/compile_fail")

    add_test(NAME ${cf_target}
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                     --target ${cf_target} ${_cf_config})
    set_tests_properties(${cf_target} PROPERTIES
        PASS_REGULAR_EXPRESSION "${cf_expect}"
        LABELS "compile_fail")
endforeach()
