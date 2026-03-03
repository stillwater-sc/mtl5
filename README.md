# MTL5 — Matrix Template Library 5

C++20 header-only linear algebra library for mixed-precision algorithm design and optimization.

MTL5 is a modernized successor to [MTL4](https://github.com/stillwater-sc/mtl4), preserving the proven architecture for custom number types (posits, LNS, etc.) while leveraging C++20 features for cleaner, safer, and more maintainable code.

## Features

- **Header-only** — no linking required
- **C++20 concepts** — clear constraints replacing SFINAE/enable_if
- **Zero Boost dependency** — pure C++20 standard library
- **Expression templates** — lazy evaluation with CRTP
- **Mixed-precision support** — designed for custom arithmetic types
- **Dense & sparse** — CSR/CSC, COO, ELLPACK, dense row/column-major
- **ITL included** — Krylov solvers (CG, BiCGSTAB, GMRES, IDR(s), ...) and preconditioners
- **Type-safe math** — `math::zero<T>()` and `math::one<T>()` for generic algorithms

## Requirements

- CMake 3.22+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- Catch2 v3 (fetched automatically for tests)

## Quick Start

```bash
# Clone
git clone https://github.com/stillwater-sc/mtl5.git
cd mtl5

# Build (uses dev preset: Debug, tests ON)
cmake --preset dev
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build

# Run example
./build/examples/hello_mtl5
```

## Build Options

| Option | Default | Description |
|---|---|---|
| `MTL5_BUILD_TESTS` | ON | Build the Catch2 test suite |
| `MTL5_BUILD_EXAMPLES` | ON | Build example programs |
| `MTL5_ENABLE_OPENMP` | OFF | Enable OpenMP parallelism |

## Project Structure

```
include/mtl/
├── concepts/       # C++20 concepts (Scalar, Matrix, Vector, ...)
├── tag/            # Compile-time tags (orientation, sparsity, ...)
├── traits/         # Type traits and metafunctions
├── math/           # Algebraic identities (zero, one)
├── detail/         # Internal implementation
├── mat/            # Matrix types, expressions, views
├── vec/            # Vector types and expressions
├── operation/      # Free-function operations (dot, norms, LU, QR, ...)
├── functor/        # Scalar and typed functors
├── recursion/      # Block-recursive infrastructure
├── io/             # I/O (Matrix Market)
├── itl/            # Iterative solvers and preconditioners
└── interface/      # Optional BLAS/LAPACK bindings
```

## Key Boost-to-C++20 Replacements

| MTL4 (Boost) | MTL5 (C++20) |
|---|---|
| `boost::enable_if<is_matrix<T>>` | `requires Matrix<T>` |
| `boost::mpl::if_<cond, A, B>::type` | `std::conditional_t<cond, A, B>` |
| `boost::is_same<A,B>` | `std::is_same_v<A,B>` |
| `BOOST_STATIC_ASSERT` | `static_assert` |
| `boost::shared_ptr<T>` | `std::shared_ptr<T>` |

## Installation

```bash
cmake --preset release
cmake --build build-release
cmake --install build-release --prefix /usr/local
```

Then in your project:
```cmake
find_package(MTL5 REQUIRED)
target_link_libraries(myapp PRIVATE MTL5::mtl5)
```

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

MTL5 builds on the foundational work of MTL4 by Peter Gottschling and the Simunova team.
