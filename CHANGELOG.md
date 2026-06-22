# Changelog

All notable changes to MTL5 are documented in this file.
Format follows [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

#### Mixed-precision tensor operations (epic #157)
- **`mtl::math::accumulator_traits<Acc, Value>`** — a shared, cross-cutting accumulator policy with a generalized `value<Result>` round-out, expressing the three independent precisions of a mixed-precision op: element (storage), accumulator (compute), result (serialize). The accumulate→output conversion is fused into the final store (#158)
- **Accumulator/result policy on the dense operations**: `dot`/`dot_real` (#159), `gemm`/`mult` with the result type inferred from `C` (#161), `gemv` (#160), and the sum-of-squares norms `two_norm`/`frobenius_norm` (#162). E.g. `mult<float>(A_bf16, B_bf16, C_bf16)` accumulates in fp32 and writes bf16 once. Default `Accumulator = void` is byte-identical to prior behavior
- **Dispatch guarantee** `interface::accumulator_allows_blas_v` — any non-default accumulator forces the native kernel even for float/double (external BLAS cannot honor a custom accumulator); proven via a counting accumulator (#163)
- **`mtl::convert`** — standalone element-wise tensor re-quantization for non-fused re-typing (distinct from the fused accumulate→store epilogue) (#164)
- **SIMD widening dot** — `batch::load_widen` (Highway `Rebind`+`PromoteTo`) + `simd::reduce_dot_widen` for float→double; `dot` routes its mixed path to it (~2.6× over scalar) (#165)

#### Sparse direct solvers
- **`sparse_lu_refactor` + `native_klu_refactor`** — analyze/factor/refactor: refactorize a same-pattern matrix by reusing the symbolic structure + pivot sequence (no BTF/ordering/reach/pivot-search), ~2.2× faster than a full factor; the SPICE-transient path (#153, #154)
- **`mtl::sparse::iterative_refine`** — generic, Universal-free iterative refinement through any factorization, with a templated residual precision, an optional scaled variant (rescues narrow-exponent low-precision factors), patience-based termination, and best-iterate return (#119, #167)

#### Documentation
- **"Measuring Solver Accuracy"** algorithm page — residuals, norms, absolute vs relative error, and backward-vs-forward error / conditioning (#152)

#### Benchmark suite
- **Size-N sweep** (`--sweep START:STOP:STEP` and `:xFACTOR`) plus BLAS-level suite groups `l1`/`l2`/`l3`/`blas`; default sizes now bracket powers of two with odd/1.5x neighbours to expose padding overhead (#77)
- **GFLOP/s-vs-N plotting** (`benchmarks/plot_results.py`, matplotlib) and committed example data + rendered plots under `benchmarks/data/` with provenance (#78, #79, #80)
- **One-executable-per-backend methodology** — `bench_all` calls only the public `mtl::` API; the build flags choose the backend (native / OpenBLAS / MKL); `run_sweeps.sh` builds all variants, pins to a P-core, and emits one CSV per backend (#81)

#### Documentation & API
- **Doxygen C++ API reference** generated into the docs site (`docs-site/Doxyfile`, `npm run api`, sidebar link, CI step) (#73)
- Public `eigenvalue_symmetric_generic()` — the generic (LAPACK-free) symmetric eigensolver, extracted so it can be called regardless of `MTL5_HAS_LAPACK` (#78)

#### Tooling / CI
- `.github/dependabot.yml` for the `github-actions` ecosystem (#64)

### Changed
- **`mtl::dot` / `dot_real` now dispatch to BLAS `?dot`** when types qualify (consistency with `two_norm`); both `dot` and `two_norm` BLAS paths guard the `int` length cast against overflow (#81)
- **Benchmarks rewritten** to the single-path public-API model; deleted the `Native/Blas/Lapack` policy-tag harness (#81)
- **CI hardening**: all GitHub Actions pinned to commit SHAs with `persist-credentials: false` (#64); sccache gated to trusted runs to prevent GHA cache poisoning (#74); Dependabot action bumps (#66–#71)
- Benchmark README: corrected CMake option names (`MTL5_WITH_BLAS/LAPACK`) and added Intel MKL (`BLA_VENDOR=Intel10_64lp`) instructions (#75)
- `.gitignore`: ignore Claude Code per-user/runtime files (#76)

### Fixed
- `antisymmetric_tensor::set` wrote out of bounds for diagonal indices (`i == j`) under `NDEBUG`, where the guarding `assert` is compiled out; now a safe no-op (#63)
- Benchmark `native` eigenvalue backend was silently dispatching to LAPACK; it now uses the generic C++ solver so `native` vs `lapack` is a genuine comparison (#78)

### Planned
- **Epic #82 — native dense BLAS performance** (sub-issues #83–#93, milestone v0.6): bring the native GEMM/GEMV/L1 kernels to within 10–20% of OpenBLAS/MKL via a SIMD abstraction layer, register-blocked micro-kernel, GotoBLAS/BLIS-style cache blocking + packing, and multithreading

## [5.2.0] — 2026-03-16

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
