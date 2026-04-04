# Regression tests — large-scale solver correctness tests.
#
# These tests exercise dense and sparse solvers with thousands to tens of
# thousands of DOF.  They are slower than unit tests (~10-15 min total)
# and are gated behind MTL5_BUILD_REGRESSION_TESTS (OFF by default).
#
# CTest labels:
#   ctest -L regression         — run ONLY regression tests
#   ctest -LE regression        — run ONLY unit tests (excludes regression)
#   ctest                       — run ALL tests (unit + regression)
#
# Default timeout per regression test: 120 seconds.

# Helper: compile regression tests with the "regression" CTest label.
# Same pattern as compile_all() in unit tests but adds the label.
function(add_regression_tests prefix folder sources)
    foreach(src ${sources})
        get_filename_component(stem ${src} NAME_WE)
        set(target "regression_${prefix}_${stem}")
        add_executable(${target} ${src})
        target_link_libraries(${target} PRIVATE MTL5::mtl5 Catch2::Catch2WithMain)
        set_target_properties(${target} PROPERTIES
            FOLDER "Tests/regression/${folder}"
        )
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES
            LABELS "regression"
            TIMEOUT 300
        )
    endforeach()
endfunction()

# Dense solver regression tests
file(GLOB DENSE_REG_SRC "dense/*.cpp")
if(DENSE_REG_SRC)
    add_regression_tests("dense" "dense" "${DENSE_REG_SRC}")
endif()

# Sparse solver regression tests
file(GLOB SPARSE_REG_SRC "sparse/*.cpp")
if(SPARSE_REG_SRC)
    add_regression_tests("sparse" "sparse" "${SPARSE_REG_SRC}")
endif()

# Iterative solver regression tests
file(GLOB ITL_REG_SRC "itl/*.cpp")
if(ITL_REG_SRC)
    add_regression_tests("itl" "itl" "${ITL_REG_SRC}")
endif()
