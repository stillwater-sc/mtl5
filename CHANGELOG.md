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
- **SIMD widening GEMM** — the blocked GEMM generalized to `<TC accumulator, TAB operand>` (default `TAB = TC` ⇒ same-type path byte-identical): the micro-kernel widens narrow operands on load (`batch<TC>::load_widen<TAB>`) into `TC` accumulator registers. `mult<double>(A_float, B_float, C_double)` routes to it; **10–16× over the scalar generic kernel** (Highway), matching the wide-accumulator reference to rounding (#176)

#### Sparse direct solvers
- **`sparse_lu_refactor` + `native_klu_refactor`** — analyze/factor/refactor: refactorize a same-pattern matrix by reusing the symbolic structure + pivot sequence (no BTF/ordering/reach/pivot-search), ~2.2× faster than a full factor; the SPICE-transient path (#153, #154)
- **`mtl::sparse::iterative_refine`** — generic, Universal-free iterative refinement through any factorization, with a templated residual precision, an optional scaled variant (rescues narrow-exponent low-precision factors), patience-based termination, and best-iterate return (#119, #167)

#### Native supernodal LU (SuperLU epic #186)
- **`mtl::sparse::analysis::analyze_unsymmetric`** — column elimination tree (etree of AᵀA without forming AᵀA), column counts, and the unsymmetric supernode partition + LU fill bound, in the postorder that makes supernode columns contiguous (#181)
- **`supernodal_lu_numeric`** — native left-looking Gilbert–Peierls LU that groups columns into **supernodes** and applies each as a dense block update, with **threshold partial pivoting**, Eisenstat–Liu symmetric pruning, and dynamic supernode formation. Generic over the **`accumulator_traits` accumulator**, so a low-precision factor can accumulate in higher precision — the mixed-precision capability the fixed-precision SuiteSparse library cannot offer. Matches scalar `sparse_lu` to machine precision (#182)
- **`supernodal_lu_refactor`** — numeric-only recompute that reuses a prior factorization's order + pivot sequence + L/U pattern; **1.9–3.2× faster** than a full factor (the transient-SPICE / mp-spice path) (#184)
- **Row equilibration** — opt-in `scale=true` factors `R·A` (`r=1/max|row|`) for pivot stability in low/mixed precision; RHS row-scaled in `solve()`, `x` unchanged (#185)
- **`bench_superlu`** — native-vs-SuiteSparse-SuperLU scoreboard on an unsymmetric suite (#180)
- Mixed-precision iterative refinement integrates end-to-end via `iterative_refine` (low-precision supernodal factor + high-precision residual)
- **Note:** FP64 single-factor speed parity with SuiteSparse SuperLU is **out of scope / not planned** (#183). Profiling showed the panel GEMM is only ~14% of factor time (the bottleneck is scalar/serial work), so parity would require a full SuperLU-style reimplementation; MTL5's differentiator is mixed precision, which is delivered

#### Documentation
- **"Measuring Solver Accuracy"** algorithm page — residuals, norms, absolute vs relative error, and backward-vs-forward error / conditioning (#152)
- **"Mixed-Precision Kernels: Why, What, and How"** algorithm page — an introduction to mixed-precision algorithm design: store-narrow/accumulate-wide, the Element → Accumulate → Result model, and the SIMD widening GEMM as a worked optimization example (#200)

#### Benchmark suite
- **Size-N sweep** (`--sweep START:STOP:STEP` and `:xFACTOR`) plus BLAS-level suite groups `l1`/`l2`/`l3`/`blas`; default sizes now bracket powers of two with odd/1.5x neighbours to expose padding overhead (#77)
- **GFLOP/s-vs-N plotting** (`benchmarks/plot_results.py`, matplotlib) and committed example data + rendered plots under `benchmarks/data/` with provenance (#78, #79, #80)
- **One-executable-per-backend methodology** — `bench_all` calls only the public `mtl::` API; the build flags choose the backend (native / OpenBLAS / MKL); `run_sweeps.sh` builds all variants, pins to a P-core, and emits one CSV per backend (#81)

#### Documentation & API
- **Doxygen C++ API reference** generated into the docs site (`docs-site/Doxyfile`, `npm run api`, sidebar link, CI step) (#73)
- Public `eigenvalue_symmetric_generic()` — the generic (LAPACK-free) symmetric eigensolver, extracted so it can be called regardless of `MTL5_HAS_LAPACK` (#78)

#### Tooling / CI
- `.github/dependabot.yml` for the `github-actions` ecosystem (#64)

#### Eigenvalue/eigenvector solvers (epic #202)
- **`mtl::eigen`** — general (non-symmetric) eigenvalues **and right eigenvectors**, returned as a structured-bindable `{ eigenvalues, eigenvectors }` (complex), mirroring `eigen_symmetric`. In the in-house path, eigenvalues come from the general QR path and each eigenvector is recovered by **inverse iteration** on `A - lambda_k*I` (partial-pivot complex LU with a pivot floor); cluster-aware Gram-Schmidt deflation yields an **independent basis for repeated eigenvalues**. When LAPACK is available and the type qualifies, `eigen` instead dispatches to `geev` (which returns the eigenvectors directly — see #204). Eigenvectors are unit-norm with a canonical phase (#203)
- **LAPACK `geev` dispatch** for the general eigenproblem — `eigenvalue`/`eigen` route to `sgeev_`/`dgeev_` when `MTL5_HAS_LAPACK` is defined and the matrix is a column-major `dense2D<float/double>` (mirrors the symmetric `syev` dispatch); custom number types and other orientations use the in-house path (#204)
- **Matrix-free iterative eigensolvers** in `mtl::itl` (`include/mtl/itl/eigen/`), operating through the `LinearOperator` concept (`A * x`) so they apply to `dense2D`, `compressed2D`, and user matrix-free operators: `power_iteration` (dominant pair), `lanczos` (symmetric, k extremal Ritz pairs via a tridiagonal projection), `arnoldi` (general, k Ritz pairs via a Hessenberg projection). Each solves the small projected problem with the dense eigensolvers; an `eigen_which` selector picks the wanted end of the spectrum (#205)
- **Sparse eigensolver with shift-invert** in `mtl::sparse` — `sparse_eigs` (largest-magnitude, Arnoldi directly on the sparse operator), `sparse_eigs_shift_invert` (k eigenpairs nearest `sigma` via `(A - sigma*I)^{-1}` applied inside Arnoldi, mapping `lambda = sigma + 1/theta`), and the reusable `shift_invert_operator` (factor once with sparse LU, apply many; tiny pivots perturbed so a shift on an eigenvalue stays solvable) (#206)
- **Eigenvalue/eigenvector solver guide** — `docs/algorithms/eigenvalues.md`: a decision guide plus a runnable snippet for every public eigen API across dense/iterative/sparse, the LAPACK dispatch conditions, and the custom-number-type story (#203, #207)

### Changed
- **`mtl::dot` / `dot_real` now dispatch to BLAS `?dot`** when types qualify (consistency with `two_norm`); both `dot` and `two_norm` BLAS paths guard the `int` length cast against overflow (#81)
- **Benchmarks rewritten** to the single-path public-API model; deleted the `Native/Blas/Lapack` policy-tag harness (#81)
- **CI hardening**: all GitHub Actions pinned to commit SHAs with `persist-credentials: false` (#64); sccache gated to trusted runs to prevent GHA cache poisoning (#74); Dependabot action bumps (#66–#71)
- Benchmark README: corrected CMake option names (`MTL5_WITH_BLAS/LAPACK`) and added Intel MKL (`BLA_VENDOR=Intel10_64lp`) instructions (#75)
- `.gitignore`: ignore Claude Code per-user/runtime files (#76)
- **CI now exercises the LAPACK dispatch paths** — a `lapack` job (Linux GCC + Clang, `-DMTL5_WITH_LAPACK=ON`) builds and runs the external-library `geev`/`syev` paths, which the default LAPACK-free matrix never compiled (#212)

### Fixed
- **`mtl::eigenvalue` single-shift QR stalled on strongly non-normal matrices** whose complex eigenvalues need a double-shift (Francis) step — it fell through to reading the diagonal and **silently returned wrong eigenvalues** (e.g. the Forsythe companion matrix returned all shift value, which the old trace-only test accepted). Replaced with the **Francis implicit double-shift QR** (EISPACK `hqr`): real Schur form via 1×1/2×2 block deflation, exceptional shifts to break stagnation, and a `std::runtime_error` on non-convergence instead of a wrong result. Discovered while implementing the eigenvector generator (#203); tightened the Forsythe test to compare the full spectrum (#209)
- AMD/COLAMD minimum-degree garbage-collection compaction mis-restored each element's first entry, corrupting the quotient-graph pointers once fill exhausted the elbow room (e.g. the AᵀA pattern of a 2-D 5-point grid at n ≥ 64); it now follows the CSparse compaction order. Surfaced while validating the supernodal-LU column ordering; regression test added (#189, #191)
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
