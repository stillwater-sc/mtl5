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

if(MTL5_BUILD_REGRESSION_TESTS)
    add_subdirectory(regression)
endif()
