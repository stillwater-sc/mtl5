# Changelog

All notable changes to MTL5 are documented in this file.
Format follows [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

#### Sparse Direct Solver Infrastructure (Phases 1-6)
- **Phase 1: Infrastructure** — CSC format (`csc_matrix`), permutation utilities, sparse accumulator, elimination tree (O(nnz) via path compression), postorder traversal, sparse/dense triangular solves (Gilbert-Peierls reach + solve), Reverse Cuthill-McKee ordering, `FillReducingOrdering` and `SparseDirectSolver` concepts
- **Phase 2: Sparse Cholesky** (`sparse_cholesky.hpp`) — Up-looking LL^T factorization for SPD matrices with symbolic/numeric phase separation, pluggable fill-reducing orderings, automatic permutation handling in solve
- **Phase 3: Sparse LU** (`sparse_lu.hpp`) — Left-looking PA=LU factorization with threshold partial pivoting, `requires OrderedField<Value>` concept constraint, sparse workspace tracking for efficient column processing
- **Phase 4: Sparse QR** (`sparse_qr.hpp`) — Householder QR for square and overdetermined systems, least-squares solve (min ||Ax-b||), compact V+beta storage, handles rectangular matrices (m >= n)
- **Phase 5: AMD/COLAMD orderings** — Approximate Minimum Degree for symmetric fill reduction (Cholesky), Column AMD for unsymmetric fill reduction (LU/QR) via A^T*A column intersection graph
- **Phase 6: External solver interfaces** — RAII wrappers for SuperLU (`superlu_solver`), KLU (`klu_solver`), CHOLMOD (`cholmod_solver`), SPQR (`spqr_solver`), with CMake find logic, `#ifdef` guards, and CRS-to-CCS conversion

#### Development Workflow
- Conventional commit format (`feat:`, `fix:`, `test:`, `docs:`, `chore:`)
- CodeRabbit AI review configuration (`.coderabbit.yaml`) with C++20-specific review instructions
- Branch protection on `main` (PRs required, CI must pass)

#### Documentation
- Sparse direct solvers design document (`docs/sparse-direct-solvers-design.md`)
- Position paper: MTL5 + Universal for accelerated mixed-precision linear algebra (`docs/position-mixed-precision-acceleration.md`)

#### Testing
- Exhaustive cross-solver validation: Cholesky/LU/QR consistency on systems up to 100x100
- UMFPACK vs native solver comparison
- Edge cases: block diagonal, permutation matrices, identity, rectangular CSC, 1x1
- 90 total tests, all passing across GCC, Clang, Apple Clang, MSVC

### Changed
- `include/mtl/mtl.hpp` — added sparse solver and external interface umbrella includes
- `CMakeLists.txt` — added `MTL5_ENABLE_SUPERLU`, `MTL5_ENABLE_KLU`, `MTL5_ENABLE_CHOLMOD`, `MTL5_ENABLE_SPQR` options
- `CLAUDE.md` — documented `sparse/` namespace, conventional commits, branch workflow, PR process
