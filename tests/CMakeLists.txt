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

# The PowerShell harnesses run on one machine each, by hand, ten minutes into a
# session -- so a syntax error surfaces at the worst moment on the one machine
# nobody else has. Parse them wherever a PowerShell is present, which on the
# hosted runners includes Linux.
find_program(MTL5_POWERSHELL NAMES pwsh powershell)
if(MTL5_POWERSHELL)
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
