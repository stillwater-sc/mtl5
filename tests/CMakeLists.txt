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

if(MTL5_BUILD_REGRESSION_TESTS)
    add_subdirectory(regression)
endif()
