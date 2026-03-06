cmake_minimum_required(VERSION 3.22)

project(MTL5
    VERSION 5.1.0
    LANGUAGES CXX
    DESCRIPTION "Matrix Template Library 5 — C++20 header-only linear algebra"
)

# ── CMake infrastructure ────────────────────────────────────────────────
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake")
include(banners)
include(compiler_helpers)
print_header()

# ── Options ──────────────────────────────────────────────────────────────
option(MTL5_BUILD_TESTS    "Build the test suite"       ON)
option(MTL5_BUILD_EXAMPLES "Build the examples"         ON)
option(MTL5_ENABLE_OPENMP  "Enable OpenMP parallelism"  OFF)
option(MTL5_ENABLE_BLAS    "Enable BLAS acceleration"   OFF)
option(MTL5_ENABLE_LAPACK  "Enable LAPACK acceleration" OFF)
option(MTL5_CMAKE_TRACE    "Trace CMake target creation" OFF)

# ── Global settings ─────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Configure version header ────────────────────────────────────────────
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/mtl/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/mtl/version.hpp"
    @ONLY
)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ── Header-only INTERFACE library ────────────────────────────────────────
add_library(mtl5 INTERFACE)
add_library(MTL5::mtl5 ALIAS mtl5)

target_include_directories(mtl5 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(mtl5 INTERFACE cxx_std_20)

if(MTL5_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(mtl5 INTERFACE OpenMP::OpenMP_CXX)
endif()

if(MTL5_ENABLE_BLAS)
    find_package(BLAS REQUIRED)
    target_compile_definitions(mtl5 INTERFACE MTL5_HAS_BLAS)
    target_link_libraries(mtl5 INTERFACE ${BLAS_LIBRARIES})
endif()

if(MTL5_ENABLE_LAPACK)
    find_package(LAPACK REQUIRED)
    target_compile_definitions(mtl5 INTERFACE MTL5_HAS_LAPACK)
    target_link_libraries(mtl5 INTERFACE ${LAPACK_LIBRARIES})
endif()

# ── Subdirectories ───────────────────────────────────────────────────────
if(MTL5_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MTL5_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Install rules ────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS mtl5
    EXPORT MTL5Targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/mtl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/mtl/version.hpp"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mtl
)

install(EXPORT MTL5Targets
    FILE MTL5Targets.cmake
    NAMESPACE MTL5::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MTL5
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MTL5Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MTL5Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MTL5
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MTL5ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MTL5Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MTL5ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MTL5
)

# ── Configuration summary ───────────────────────────────────────────────
include(summary)
mtl5_print_configuration_summary()
