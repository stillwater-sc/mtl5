include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch2Testing OPTIONAL)

add_subdirectory(unit)
add_subdirectory(integration)

# -- exceptions and RTTI disabled (#652) --------------------------------------
# Firmware builds with -fno-exceptions -fno-rtti, so dense_vector and dense2D
# (fixed-size, on-stack, and the thread pool they include) must compile that
# way. Built once per pool, so MTL5_POOL_SPIN's spin_pool is covered too. GCC
# and Clang only: MSVC and Clang-CL spell the flags differently.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    foreach(_pool default spin)
        set(_tgt build_test_no_exceptions_${_pool})
        add_executable(${_tgt} no_exceptions/test_no_exceptions.cpp)
        target_link_libraries(${_tgt} PRIVATE MTL5::mtl5)
        target_compile_options(${_tgt} PRIVATE -fno-exceptions -fno-rtti -Wall -Wextra -Werror)
        if(_pool STREQUAL "spin")
            target_compile_definitions(${_tgt} PRIVATE MTL5_POOL_SPIN)
        endif()
        set_target_properties(${_tgt} PROPERTIES FOLDER "Tests/build")
        add_test(NAME ${_tgt} COMMAND ${_tgt})
    endforeach()
endif()

# -- benchmark preflight contract (#442) -------------------------------------
# preflight.sh and preflight.ps1 are one contract with two implementations that
# run on disjoint machines, so they drift silently unless each platform checks
# its own half. Windows picks the .ps1, everything else the .sh; both are held to
# the same key list in tools/cmake/test_preflight.cmake.
if(WIN32)
    find_program(MTL5_PREFLIGHT_INTERP NAMES pwsh powershell)
    set(_preflight_script ${PROJECT_SOURCE_DIR}/benchmarks/preflight.ps1)
else()
    find_program(MTL5_PREFLIGHT_INTERP NAMES bash)
    set(_preflight_script ${PROJECT_SOURCE_DIR}/benchmarks/preflight.sh)
endif()
if(MTL5_PREFLIGHT_INTERP)
    add_test(NAME bench_preflight_contract
             COMMAND ${CMAKE_COMMAND}
                     -DINTERP=${MTL5_PREFLIGHT_INTERP}
                     -DSCRIPT=${_preflight_script}
                     -DREPO=${PROJECT_SOURCE_DIR}
                     -P ${PROJECT_SOURCE_DIR}/tools/cmake/test_preflight.cmake)
endif()

# The PowerShell harnesses run on one machine each, by hand, ten minutes into a
# session -- so a syntax error surfaces at the worst moment on the one machine
# nobody else has. Parse them wherever a PowerShell is present, which on the
# hosted runners includes Linux.
#
# Not on Apple: on the macos-latest runner image, launching pwsh now fails in
# LaunchServices (_LSOpenURLsWithCompletionHandler error -1712, a timeout)
# before it parses anything, and leaves orphaned pwsh processes. The same
# scripts are still parsed on the Linux and Windows runners, and they only
# run on Windows.
find_program(MTL5_POWERSHELL NAMES pwsh powershell)
if(MTL5_POWERSHELL AND NOT APPLE)
    file(GLOB MTL5_PS1_HARNESSES "${PROJECT_SOURCE_DIR}/benchmarks/*.ps1")
    if(MTL5_PS1_HARNESSES)
        add_test(NAME bench_powershell_parse
                 COMMAND ${CMAKE_COMMAND}
                         -DINTERP=${MTL5_POWERSHELL}
                         "-DSCRIPTS=${MTL5_PS1_HARNESSES}"
                         -P ${PROJECT_SOURCE_DIR}/tools/cmake/test_powershell_parse.cmake)
    endif()
endif()

if(MTL5_BUILD_REGRESSION_TESTS)
    add_subdirectory(regression)
endif()
