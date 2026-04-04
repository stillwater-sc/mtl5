# Build Options

MTL5 provides several CMake options to control the build and enable optional features.

## Core Options

| Option | Default | Description |
|---|---|---|
| `MTL5_BUILD_TESTS` | ON | Build the Catch2 test suite |
| `MTL5_BUILD_EXAMPLES` | ON | Build example programs |
| `MTL5_ENABLE_OPENMP` | OFF | Enable OpenMP parallelism |

## External Library Bindings

These options link optional external libraries for hardware-accelerated or specialized solvers. MTL5 has native C++ implementations of all algorithms — these bindings are for users who want to delegate to optimized external libraries.

| Option | Default | Description |
|---|---|---|
| `MTL5_WITH_BLAS` | OFF | Link BLAS library for dense acceleration |
| `MTL5_WITH_LAPACK` | OFF | Link LAPACK library for factorizations |
| `MTL5_WITH_UMFPACK` | OFF | Link UMFPACK library (SuiteSparse) |
| `MTL5_WITH_SUPERLU` | OFF | Link SuperLU library |
| `MTL5_WITH_SUITESPARSE_KLU` | OFF | Link KLU sparse solver (SuiteSparse) |
| `MTL5_WITH_SUITESPARSE_CHOLMOD` | OFF | Link CHOLMOD Cholesky (SuiteSparse) |
| `MTL5_WITH_SUITESPARSE_SPQR` | OFF | Link SuiteSparseQR (SuiteSparse) |

## Build Presets

The project includes CMake presets for common configurations:

```bash
# Development (Debug, tests ON)
cmake --preset dev

# Or configure manually
cmake -B build -DCMAKE_CXX_STANDARD=20 -DMTL5_WITH_BLAS=ON
cmake --build build -j$(nproc)
```

## Running Tests

```bash
# Run all tests
ctest --test-dir build

# Run a specific test
ctest --test-dir build -R test_concepts

# Run with verbose output
ctest --test-dir build --output-on-failure
```
