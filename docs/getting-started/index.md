# Getting Started

MTL5 is a header-only C++20 linear algebra library. This guide will get you up and running quickly.

## Requirements

- **CMake** 3.22+
- **C++20 compiler**: GCC 11+, Clang 14+, Apple Clang 15+, MSVC 2022+
- **Catch2 v3** (fetched automatically for tests)

## Quick Install

```bash
git clone https://github.com/stillwater-sc/mtl5.git
cd mtl5
cmake --preset dev
cmake --build build -j$(nproc)
ctest --test-dir build
```

## CMake Integration

### As a subdirectory

```cmake
add_subdirectory(mtl5)
target_link_libraries(my_app PRIVATE mtl5::mtl5)
```

### With find_package

```cmake
find_package(mtl5 REQUIRED)
target_link_libraries(my_app PRIVATE mtl5::mtl5)
```

### With FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
  mtl5
  GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(mtl5)
target_link_libraries(my_app PRIVATE mtl5::mtl5)
```
