# Build Options

MTL5 provides several CMake options to control the build and enable optional features.

## Core Options

| Option | Default | Description |
|---|---|---|
| `MTL5_BUILD_TESTS` | ON | Build the Catch2 test suite |
| `MTL5_BUILD_EXAMPLES` | ON | Build example programs |

## External Library Bindings

These options link optional external libraries for hardware-accelerated or specialized solvers. MTL5 has native C++ implementations of all algorithms — these bindings are for users who want to delegate to optimized external libraries.

| Option | Default | Description |
|---|---|---|
| `MTL5_WITH_BLAS` | OFF | Link BLAS library for dense acceleration |
| `MTL5_WITH_LAPACK` | OFF | Link LAPACK library for factorizations |
| `MTL5_WITH_ZLIB` | OFF | Link zlib for transparent gzip (`.mtx.gz`) Matrix Market reading |
| `MTL5_WITH_UMFPACK` | OFF | Link UMFPACK library (SuiteSparse) |
| `MTL5_WITH_SUPERLU` | OFF | Link SuperLU library |
| `MTL5_WITH_SUITESPARSE_KLU` | OFF | Link KLU sparse solver (SuiteSparse) |
| `MTL5_WITH_SUITESPARSE_CHOLMOD` | OFF | Link CHOLMOD Cholesky (SuiteSparse) |
| `MTL5_WITH_SUITESPARSE_SPQR` | OFF | Link SuiteSparseQR (SuiteSparse) |

### Matrix Market I/O: gzip and large files

`mtl::io::mm_read` / `mm_read_dense` read SuiteSparse `.mtx` files. Two features
support circuit5M-scale workloads:

**Transparent gzip.** Built with `-DMTL5_WITH_ZLIB=ON` (defines `MTL5_HAS_ZLIB`),
`mm_read("matrix.mtx.gz")` decompresses on the fly — no need to pre-extract a
downloaded `.tar.gz`. Without the flag, passing a `.gz` path throws a clear error
pointing at the option. Plain `.mtx` reading is unaffected and needs no zlib.

```bash
cmake -B build -DCMAKE_CXX_STANDARD=20 -DMTL5_WITH_ZLIB=ON
```

**Large-file load path.** The sparse coordinate reader assembles CRS directly from
a single triplet buffer sized from the header `nnz` and sorts it in place,
avoiding the extra full-size copy that the generic `coordinate2D::compress()`
makes. This roughly halves transient peak memory on very large matrices while
producing identical output (same `(row, col)` order, same duplicate
accumulation).

**circuit5M (`Freescale/circuit5M`, ~5.56M × 5.56M, ~59.5M nnz).** The loader is
built to handle this scale, but it is an opt-in large run (multi-GB file and RAM):
the matrix is not downloaded or factored in CI. To validate locally, fetch the
matrix, then load and factor it with native KLU in `double`, recording wall-clock
load/factor time and peak RSS. Expect the load to be dominated by I/O and the
single triplet buffer (~`nnz × 24` bytes) plus the CRS output; factor cost is
governed by the per-block ordering (see the native-KLU notes).

## Native Performance & Development

These tune or instrument **in-tree** builds (tests, benchmarks). They are
`BUILD_INTERFACE`-only and never leak into the installed/exported package.

| Option | Default | Description |
|---|---|---|
| `MTL5_WITH_HIGHWAY` | OFF | Use Google Highway for SIMD-accelerated native kernels (found or fetched at a pinned tag); scalar fallback otherwise |
| `MTL5_NATIVE_ARCH` | OFF | Tune for the host CPU (`-march=native`); lets the SIMD layer pick the widest ISA (non-portable binaries) |
| `MTL5_NATIVE_FAST_GEMM` | OFF | Route `mtl::mult` through the native blocked GEMM / SIMD GEMV path instead of the generic loop |
| `MTL5_SANITIZE` | *(empty)* | Comma-separated sanitizers for in-tree builds, e.g. `-DMTL5_SANITIZE=address,undefined` (GCC/Clang) |

```bash
# Native fast dense path, host-tuned (what benchmarks/run_sweeps.sh builds):
cmake -B build -DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON

# ASan + UBSan over the kernels:
cmake -B build-asan -DMTL5_SANITIZE=address,undefined -DMTL5_WITH_HIGHWAY=ON
```

### Multithreaded GEMM

The native blocked GEMM parallelizes its row (`ic`) loop with the C++ standard
concurrency runtime (`std::thread`) — **no OpenMP dependency**. It is controlled
at runtime by the `MTL5_NUM_THREADS` environment variable (clamped to the
hardware concurrency); unset or `1` keeps the single-thread path unchanged.

```bash
MTL5_NUM_THREADS=8 ./your_program   # use up to 8 threads for native GEMM
```

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
