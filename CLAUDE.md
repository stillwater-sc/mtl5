# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MTL5 (Matrix Template Library Version 5) is a C++20 header-only linear algebra library. It modernizes MTL4 by replacing all Boost dependencies with C++20 standard equivalents (concepts, constexpr, std::span, ranges). The library is designed for mixed-precision algorithm design with custom number types (posits, LNS, etc.).

## Build Commands

```bash
# Configure with dev preset (Debug, tests ON)
cmake --preset dev
# Or manually:
cmake -B build -DCMAKE_CXX_STANDARD=20

# Build all (tests + examples)
cmake --build build -j$(nproc)

# Run all tests
ctest --test-dir build

# Run a single test
ctest --test-dir build -R <test_name>   # e.g. -R test_concepts

# Run the example
./build/examples/hello_mtl5
```

### CMake Options

- `-DMTL5_BUILD_TESTS=ON` (default ON) — build Catch2 test suite
- `-DMTL5_BUILD_EXAMPLES=ON` (default ON) — build examples
- `-DMTL5_ENABLE_OPENMP=ON` — enable OpenMP parallelism
- `-DMTL5_ENABLE_BLAS=ON` — enable BLAS acceleration for dense mult/norms
- `-DMTL5_ENABLE_LAPACK=ON` — enable LAPACK acceleration for LU/QR/Cholesky/SVD/eigenvalue
- `-DMTL5_ENABLE_UMFPACK=ON` — enable UMFPACK sparse direct solver (requires SuiteSparse)

## Architecture

### Source Layout

All library headers live under `include/mtl/`:

- **`concepts/`** — C++20 concepts: `Scalar`, `Field`, `Matrix`, `Vector`, `Collection`, `LinearOperator`, `Preconditioner`
- **`tag/`** — Compile-time tags: orientation, sparsity, shape, traversal, storage
- **`traits/`** — Type traits: `category`, `ashape`, `transposed_orientation`
- **`math/`** — Algebraic identities: `zero<T>()`, `one<T>()`, operation tags
- **`detail/`** — Internal: memory blocks, index types
- **`mat/`** — Matrix types: `dense2D`, `compressed2D`, `coordinate2D`, `ell_matrix`, plus expressions and views
- **`vec/`** — Vector types: `dense_vector`, `sparse_vector`, plus expressions
- **`operation/`** — Free functions: decompositions (LU, QR, LQ, Cholesky, SVD), norms, solvers, element-wise ops
- **`functor/`** — Scalar functors (plus, times, abs, ...) and typed functors (scale, rscale, ...)
- **`recursion/`** — Block-recursive matrix infrastructure
- **`io/`** — Matrix Market I/O
- **`itl/`** — Iterative Template Library: Krylov solvers (CG, BiCGSTAB, GMRES, ...), preconditioners, smoothers
- **`interface/`** — Optional external library bindings: `blas.hpp` (L1/L2/L3), `lapack.hpp` (factorizations, eigensolvers), `umfpack.hpp` (sparse direct solver), `dispatch_traits.hpp` (compile-time dispatch decisions). Operations in `operation/` auto-dispatch to BLAS/LAPACK when `MTL5_HAS_BLAS`/`MTL5_HAS_LAPACK` is defined and types are `dense2D<float/double>`
- **`mtl.hpp`** — Kitchen-sink umbrella include
- **`mtl_fwd.hpp`** — Forward declarations

### Test Layout

Under `tests/`:

- `unit/concepts/` — Concept satisfaction tests
- `unit/mat/` — Matrix type tests
- `unit/vec/` — Vector type tests
- `unit/operation/` — Operation tests
- `unit/math/` — Math utility tests
- `unit/itl/` — ITL solver tests
- `integration/` — Integration tests (future)

### Namespaces

- `mtl::` — top-level namespace
- `mtl::mat` — matrix types and operations
- `mtl::vec` — vector types and operations
- `mtl::math` — algebraic identities and operation tags
- `mtl::tag` — compile-time dispatch tags
- `mtl::traits` — type traits and metafunctions
- `mtl::ashape` — algebraic shape classification
- `mtl::detail` — implementation details
- `mtl::itl` — iterative solvers
- `mtl::itl::pc` — preconditioners
- `mtl::itl::smoother` — multigrid smoothers
- `mtl::operation` — free-function operations
- `mtl::functor::scalar` — scalar functors
- `mtl::functor::typed` — typed functors
- `mtl::recursion` — block-recursive infrastructure
- `mtl::io` — I/O utilities
- `mtl::interface` — external library bindings

### Key Patterns

- **C++20 concepts** replace Boost.MPL enable_if chains
- **`if constexpr`** replaces tag dispatch where appropriate
- **`constexpr`** for math identities and dimension types
- **CRTP** for static polymorphism in expressions
- **Expression templates** for lazy evaluation
- **`#pragma once`** for include guards

### Porting from MTL4

When porting an MTL4 file to MTL5:
1. Replace `boost::enable_if<is_X<T>>` with `requires X<T>` concept
2. Replace `boost::mpl::if_<cond, A, B>::type` with `std::conditional_t<cond, A, B>`
3. Replace `BOOST_STATIC_ASSERT` with `static_assert`
4. Replace `boost::shared_ptr` with `std::shared_ptr`
5. Replace `math::zero(ref)` / `math::one(ref)` with `math::zero<T>()` / `math::one<T>()`
6. Move namespace from `boost::numeric::mtl` to `mtl::`
7. Move ITL from `itl::` to `mtl::itl::`

### Cross-Platform Development

This library targets Linux (x64/ARM64), macOS (ARM64), and Windows (MSVC/Clang-CL). All code MUST be portable:

- **File paths**: NEVER use hardcoded paths like `/tmp/`, `/home/`, or `C:\`. Use `std::filesystem::temp_directory_path()` for temp files, `std::filesystem::path` for path construction
- **Path separators**: Use `std::filesystem::path` and `/` operator, never hardcode `/` or `\\`
- **POSIX-only APIs**: Avoid `unistd.h`, `sys/`, `dlfcn.h` etc. without `#ifdef` guards. Prefer C++ standard library equivalents
- **Math constants**: NEVER use `M_PI`, `M_E`, etc. (POSIX extensions, not defined by MSVC). Use C++20 `std::numbers::pi`, `std::numbers::e` from `<numbers>`
- **Compiler differences**: Test with GCC, Clang, Apple Clang, and MSVC. Avoid compiler-specific extensions without `#ifdef` guards
- **Line endings**: Use `.gitattributes` to handle CRLF/LF. Don't assume `\n` in file parsing
- **Integer types**: Use `std::size_t` for sizes, not platform-specific types

### Adding a New File

- **Operation**: create header in `include/mtl/operation/`, include from `mtl.hpp` when ready
- **Test**: create `test_<name>.cpp` in appropriate `tests/unit/` subdirectory, register in `tests/unit/CMakeLists.txt`
