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
option(MTL5_ENABLE_UMFPACK "Enable UMFPACK sparse solver" OFF)
option(MTL5_ENABLE_SUPERLU "Enable SuperLU sparse solver" OFF)
option(MTL5_ENABLE_KLU     "Enable KLU sparse solver (SuiteSparse)" OFF)
option(MTL5_ENABLE_CHOLMOD "Enable CHOLMOD sparse Cholesky (SuiteSparse)" OFF)
option(MTL5_ENABLE_SPQR   "Enable SuiteSparseQR sparse QR" OFF)
option(MTL5_BUILD_BENCHMARKS "Build the benchmark suite" OFF)
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

if(MTL5_ENABLE_UMFPACK)
    find_path(UMFPACK_INCLUDE_DIR umfpack.h
        PATH_SUFFIXES suitesparse SuiteSparse)
    find_library(UMFPACK_LIBRARY NAMES umfpack)
    if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
        target_compile_definitions(mtl5 INTERFACE MTL5_HAS_UMFPACK)
        target_include_directories(mtl5 INTERFACE ${UMFPACK_INCLUDE_DIR})
        target_link_libraries(mtl5 INTERFACE ${UMFPACK_LIBRARY})
        message(STATUS "UMFPACK found: ${UMFPACK_LIBRARY}")
    else()
        message(FATAL_ERROR "UMFPACK requested but not found. "
            "Install libsuitesparse-dev or set UMFPACK_INCLUDE_DIR and UMFPACK_LIBRARY.")
    endif()
endif()

if(MTL5_ENABLE_SUPERLU)
    find_path(SUPERLU_INCLUDE_DIR slu_ddefs.h
        PATH_SUFFIXES superlu SuperLU)
    find_library(SUPERLU_LIBRARY NAMES superlu)
    if(SUPERLU_INCLUDE_DIR AND SUPERLU_LIBRARY)
        target_compile_definitions(mtl5 INTERFACE MTL5_HAS_SUPERLU)
        target_include_directories(mtl5 INTERFACE ${SUPERLU_INCLUDE_DIR})
        target_link_libraries(mtl5 INTERFACE ${SUPERLU_LIBRARY})
        message(STATUS "SuperLU found: ${SUPERLU_LIBRARY}")
    else()
        message(FATAL_ERROR "SuperLU requested but not found. "
            "Install libsuperlu-dev or set SUPERLU_INCLUDE_DIR and SUPERLU_LIBRARY.")
    endif()
endif()

if(MTL5_ENABLE_KLU)
    find_path(KLU_INCLUDE_DIR klu.h
        PATH_SUFFIXES suitesparse SuiteSparse)
    find_library(KLU_LIBRARY NAMES klu)
    find_library(BTF_LIBRARY NAMES btf)
    find_library(AMD_LIBRARY NAMES amd)
    find_library(COLAMD_LIBRARY NAMES colamd)
    find_library(SUITESPARSE_CONFIG_LIBRARY NAMES suitesparseconfig)
    if(KLU_INCLUDE_DIR AND KLU_LIBRARY)
        target_compile_definitions(mtl5 INTERFACE MTL5_HAS_KLU)
        target_include_directories(mtl5 INTERFACE ${KLU_INCLUDE_DIR})
        target_link_libraries(mtl5 INTERFACE ${KLU_LIBRARY})
        foreach(_lib BTF_LIBRARY AMD_LIBRARY COLAMD_LIBRARY SUITESPARSE_CONFIG_LIBRARY)
            if(${_lib})
                target_link_libraries(mtl5 INTERFACE ${${_lib}})
            endif()
        endforeach()
        message(STATUS "KLU found: ${KLU_LIBRARY}")
    else()
        message(FATAL_ERROR "KLU requested but not found. "
            "Install libsuitesparse-dev or set KLU_INCLUDE_DIR and KLU_LIBRARY.")
    endif()
endif()

if(MTL5_ENABLE_CHOLMOD)
    find_path(CHOLMOD_INCLUDE_DIR cholmod.h
        PATH_SUFFIXES suitesparse SuiteSparse)
    find_library(CHOLMOD_LIBRARY NAMES cholmod)
    if(CHOLMOD_INCLUDE_DIR AND CHOLMOD_LIBRARY)
        target_compile_definitions(mtl5 INTERFACE MTL5_HAS_CHOLMOD)
        target_include_directories(mtl5 INTERFACE ${CHOLMOD_INCLUDE_DIR})
        target_link_libraries(mtl5 INTERFACE ${CHOLMOD_LIBRARY})
        message(STATUS "CHOLMOD found: ${CHOLMOD_LIBRARY}")
    else()
        message(FATAL_ERROR "CHOLMOD requested but not found. "
            "Install libsuitesparse-dev or set CHOLMOD_INCLUDE_DIR and CHOLMOD_LIBRARY.")
    endif()
endif()

if(MTL5_ENABLE_SPQR)
    find_path(SPQR_INCLUDE_DIR SuiteSparseQR_C.h
        PATH_SUFFIXES suitesparse SuiteSparse)
    find_library(SPQR_LIBRARY NAMES spqr)
    if(SPQR_INCLUDE_DIR AND SPQR_LIBRARY)
        target_compile_definitions(mtl5 INTERFACE MTL5_HAS_SPQR)
        target_include_directories(mtl5 INTERFACE ${SPQR_INCLUDE_DIR})
        target_link_libraries(mtl5 INTERFACE ${SPQR_LIBRARY})
        message(STATUS "SPQR found: ${SPQR_LIBRARY}")
    else()
        message(FATAL_ERROR "SPQR requested but not found. "
            "Install libsuitesparse-dev or set SPQR_INCLUDE_DIR and SPQR_LIBRARY.")
    endif()
endif()

# ── Subdirectories ───────────────────────────────────────────────────────
if(MTL5_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MTL5_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(MTL5_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
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
