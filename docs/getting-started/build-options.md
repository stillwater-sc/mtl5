# Build Options

MTL5 provides several CMake options to control the build and enable optional features.

## Core Options

| Option | Default | Description |
|---|---|---|
| `MTL5_BUILD_TESTS` | ON | Build the Catch2 test suite |
| `MTL5_BUILD_EXAMPLES` | ON | Build example programs |

## Acceleration Options

| Option | Default | Description |
|---|---|---|
| `MTL5_ENABLE_OPENMP` | OFF | Enable OpenMP parallelism |
| `MTL5_ENABLE_BLAS` | OFF | Enable BLAS acceleration for dense multiply/norms |
| `MTL5_ENABLE_LAPACK` | OFF | Enable LAPACK for LU/QR/Cholesky/SVD/eigenvalue |
| `MTL5_ENABLE_UMFPACK` | OFF | Enable UMFPACK sparse direct solver (requires SuiteSparse) |

## Build Presets

The project includes CMake presets for common configurations:

```bash
# Development (Debug, tests ON)
cmake --preset dev

# Or configure manually
cmake -B build -DCMAKE_CXX_STANDARD=20 -DMTL5_ENABLE_BLAS=ON
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
