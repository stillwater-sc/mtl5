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
- **`sparse/`** — Sparse direct solver infrastructure: orderings (RCM, AMD, COLAMD), analysis (elimination tree, postorder), factorization (triangular solve), utilities (CSC, permutations, scatter)
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
- `unit/sparse/` — Sparse direct solver infrastructure tests
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
- `mtl::sparse` — sparse direct solver infrastructure
- `mtl::sparse::ordering` — fill-reducing orderings
- `mtl::sparse::analysis` — symbolic analysis (elimination trees, postorder)
- `mtl::sparse::factorization` — numeric factorization algorithms
- `mtl::sparse::util` — permutations, CSC format, sparse accumulators

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
- **Documentation page**: create `.md` in `docs/<category>/`, add to `FILE_MAP` in `docs-site/sync-content.mjs`

### Documentation Site Architecture

The docs site (`docs-site/`) is an Astro/Starlight site deployed to GitHub Pages.

**CRITICAL RULE: `docs-site/src/content/docs/` is 100% GENERATED — NEVER write or edit files there.**

All documentation content lives in `docs/`:

- `docs/site/` — Starlight-specific MDX pages (landing page with Hero component, etc.)
- `docs/getting-started/` — Installation, build options
- `docs/architecture/` — Architecture overview, concepts
- `docs/design/` — Design documents
- `docs/examples/` — Example walkthroughs
- `docs/generators/` — Test matrix generators
- `docs/img/` — Images (copied to `docs-site/public/img/`)

The build pipeline (`npm run build` in `docs-site/`):
1. `sync-content.mjs` wipes `src/content/docs/` entirely
2. Copies `docs/site/*.mdx` verbatim (already have frontmatter)
3. Transforms `docs/**/*.md` → adds YAML frontmatter from H1, rewrites links/images
4. Copies `docs/img/` → `public/img/`
5. Astro builds the static site from the generated content

To add a new doc page:
1. Write the `.md` file in `docs/<category>/` (pure markdown, H1 heading, no frontmatter needed)
2. Add the mapping to `FILE_MAP` in `docs-site/sync-content.mjs`
3. If it's a new sidebar section, add to `sidebar` in `docs-site/astro.config.mjs`

```bash
# Preview locally
cd docs-site && npm install && npm run dev
```

## Git Workflow

### Branch Protection

The `main` branch is protected:
- All changes go through pull requests (no direct pushes)
- CI must pass before merge
- CodeRabbit AI review is triggered on every PR

### Branch Naming

Use conventional prefixes matching the commit type:

- `feat/<topic>` — new features (e.g., `feat/sparse-cholesky`)
- `fix/<topic>` — bug fixes (e.g., `fix/csc-transpose-empty`)
- `refactor/<topic>` — code restructuring
- `test/<topic>` — adding or fixing tests
- `docs/<topic>` — documentation changes
- `perf/<topic>` — performance improvements
- `chore/<topic>` — build, CI, tooling changes

### Conventional Commits

All commit messages MUST follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat` — a new feature (correlates with MINOR in semver)
- `fix` — a bug fix (correlates with PATCH in semver)
- `refactor` — code change that neither fixes a bug nor adds a feature
- `test` — adding or correcting tests
- `docs` — documentation only changes
- `perf` — performance improvement
- `chore` — build process, CI, tooling, dependencies
- `style` — formatting, whitespace (no code change)

**Scopes** (optional, in parentheses): `sparse`, `mat`, `vec`, `itl`, `interface`, `operation`, `concepts`, `ci`, `build`

**Examples:**
```
feat(sparse): add sparse Cholesky symbolic and numeric factorization
fix(interface): correct CRS-to-CCS row index ordering for UMFPACK
refactor(mat): simplify compressed2D inserter finalization
test(sparse): add edge case tests for empty matrix permutation
docs: update design doc with Phase 2 implementation notes
perf(operation): use BLAS dispatch for sparse triangular solve
chore(ci): add MSVC 2025 to CI matrix
```

**Breaking changes:** add `!` after type/scope and a `BREAKING CHANGE:` footer:
```
refactor(concepts)!: rename Scalar concept to ArithmeticScalar

BREAKING CHANGE: All code using `Scalar<T>` must update to `ArithmeticScalar<T>`
```

### PR Workflow

1. Create a feature branch from `main`
2. Make commits following conventional commit format
3. Push branch and create PR targeting `main`
4. CodeRabbit reviews automatically; address feedback
5. CI must pass on all platforms (Linux GCC/Clang, macOS Apple Clang, Windows MSVC/Clang-CL)
6. Merge via squash-merge or regular merge (maintainer preference)

### Releases

Releases follow [Semantic Versioning](https://semver.org/) and are triggered by pushing a git tag:

```bash
git tag v5.2.0
git push --tags
```

This triggers the release workflow (`.github/workflows/release.yml`) which:
1. Runs the full CI matrix (8 platforms) on the tagged commit
2. Verifies the CMake fallback version matches the tag
3. Generates release notes from conventional commit messages
4. Creates a GitHub Release with the changelog

The CMake build system extracts the version from the git tag automatically (`git describe --tags`). When building outside a git repo or without a tag, it falls back to `MTL5_FALLBACK_VERSION` in `CMakeLists.txt`.

**Version bumping checklist:**
1. Update `MTL5_FALLBACK_VERSION` in `CMakeLists.txt` to the new version
2. Update `CHANGELOG.md` with the release date and changes
3. Commit, push, create and merge the PR
4. Tag the merge commit: `git tag v<major>.<minor>.<patch>`
5. Push the tag: `git push --tags`
