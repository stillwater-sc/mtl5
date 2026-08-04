# Changelog

All notable changes to MTL5 are documented in this file.
Format follows [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

#### Complex (Hermitian) dense factorizations (#353)
`cholesky`, `qr` and `lq` did not compile for `std::complex`, failing first on relational operators applied to a complex where a *magnitude* was meant. Repairing only those comparisons would have produced routines that compile and compute the **wrong** result, by a different mechanism in each: `cholesky`'s unconjugated inner product yields `L·Lᵀ` rather than the `L·Lᴴ` a Hermitian matrix has, while `qr`/`lq` additionally need `Σ|z|²` in place of `Σz²` and a reflector written `I − τ·v·vᴴ` rather than `I − β·v·vᵀ`. Neither fix repairs a comparison.
- **`mtl::cholesky_h_factor` / `mtl::cholesky_h_solve`** (`operation/cholesky.hpp`) — `A = L·Lᴴ` for Hermitian positive definite input. The pivot is accumulated in the **magnitude** type rather than the element type, which is what makes the positivity test well-formed: an ordering is exactly what the element type lacks. Rejects a non-real diagonal with `CHOLESKY_NOT_HERMITIAN`. `cholesky_factor` is now restricted to real element types with a diagnostic naming the alternative — unlike `ldlt` there is no complex-symmetric variant to offer, since "positive definite" is a statement about an ordering (#360)
- **Complex Householder, QR and LQ** (`operation/householder.hpp`, `qr.hpp`, `lq.hpp`) — the reflector is now `H = I − τ·v·vᴴ` with `H·x = β·e₁` and β real, built as in LAPACK `zlarfg`. `H` is unitary but **not Hermitian** for complex τ, so `H⁻¹ = Hᴴ`; that lands as `conj(tau)` in `qr_extract_Q`, plain `tau` in `lq_extract_Q`, and a conjugated row plus `Hᴴ` on the LQ factor side. Verified `‖QᴴQ−I‖ ≤ 7.8e-16` and `‖QR−A‖ ≤ 7.0e-16` over square, tall and wide shapes (#361)
- `hessenberg_factor` and `eigen_symmetric` now **reject** complex element types. They apply `H·A·H`, a similarity transform only because a real reflector is Hermitian and involutory; making `householder()` complex-capable would otherwise have let them compute a non-similar matrix. The Hermitian reduction is tracked in #362 (#361)

#### Compile-failure test harness
- **`tests/unit/compile_fail/`** — sources that must *not* compile, built on demand by ctest (`EXCLUDE_FROM_ALL`) so ordinary builds are unaffected. Each declares an `// EXPECT-ERROR: <regex>` line and is matched with `PASS_REGULAR_EXPRESSION` rather than CMake's `WILL_FAIL`, which passes on *any* non-zero build exit and would go green on a typo in the test source. The distinction is not theoretical: removing the `cholesky` guard leaves the file still failing to compile via the old `operator<=` error, so `WILL_FAIL` would have been green while the regex catches the regression to the worse diagnostic (#358)

#### Platform Performance Engineering lab
- **`ppe/`** — a progression of GEMM implementations (naive → loop order → blocking/packing → register tiling → compile-time microtile) measured across `int8/16/32/64` and `fp16/32/64`, with hypotheses recorded before the experiments that test them. Documents the *vectorization cliff*: the compile-time-unrolled microtile is **5× slower** than a plain loop reorder for `fp32`, because full unrolling removes the loop structure the vectorizer needs and 32 scalar accumulators then spill — while being the *best* kernel for `int32`, whose SIMD headroom is small (#354)

### Fixed
- **`ldlt` returned a wrong answer for Hermitian complex input** under `info == 0`. Added `ldlt_h_factor` / `ldlt_h_solve` for `A = L·D·Lᴴ`, and guards in both directions: `LDLT_NOT_SYMMETRIC` when Hermitian input reaches the LDLᵀ form, `LDLT_NOT_HERMITIAN` when a non-real diagonal reaches the LDLᴴ form. Both tests are **scale-relative** (`n · eps · scale`, via LAPACK's `CABS1` to avoid an O(n²) sweep of `hypot`), because an exact test only catches matrices that are bit-exactly Hermitian — a one-ULP perturbation slipped past the first version and produced an answer wrong in the first significant digit. Shared thresholds live in `detail/structure_tol.hpp` (#352, #356)
- **`compressed2D` accepted `tag::col_major` and ignored it.** The container is CSR unconditionally, so a `col_major` instance was byte-for-byte a CSR matrix and genuine CSC input came back as its **transpose** while the constructor reported success. Now rejected at compile time; `ell_matrix` carried the same inert-orientation hole and is rejected too. `coordinate2D` is deliberately untouched — it stores explicit `(row, col, value)` triplets, so orientation cannot change a result (#355, #358)
- **`householder` broke its own real-β guarantee** for a complex vector with a vanishing tail and a complex leading entry, returning `tau = 0` and leaving a complex `R`/`L` diagonal — silent, since `Q` stayed unitary and `Q·R` still reproduced `A`. The identity shortcut is now taken only when it is exact (#361)

### Performance
- **Blocked GEMM parallel efficiency 79% → 91%** on 8 physical cores. The shared B panel was packed by each jc-team's leader while the rest of the team waited on the barrier — a serial region proportional to `kc·nc` per `pc` step (at N=2048 with a team of 8, one thread packed ~67 MB while seven idled). The team now splits the NR-column panels; packing is pure data movement into disjoint offsets, so the packed bytes and the result are unchanged (#348)

### Changed
- `docs/sparse-direct-solvers-design.md` no longer advertises a "zero-cost CSC view over `compressed2D` with col-major parameters" — never implemented; `sparse/util/csc.hpp` uses a standalone owning `csc_matrix`. `docs/architecture/aggregate-types.md` no longer claims "Full support (CSR, CSC, COO, ELLPACK)"; CSC exists only as a conversion utility (#358)

## [5.8.0] - 2026-08-02

### Added
- **`mtl::math::accumulator_round_type<Acc, Mag>`** (`math/accumulator_traits.hpp`) — names the arithmetic type an accumulator should be rounded out to before a scalar post-processing step such as `sqrt`: the accumulator itself for a plain arithmetic `Acc`, `T` for `fma_accumulator<T>`, and the magnitude type for a custom/quire accumulator. Lets the sum-of-squares norms accept all three configurations of the accumulator model without narrowing the sum before the root, and requires nothing new of external specializations (#324)

#### Dense mixed-precision iterative refinement (BLAS extraction from Universal)
- **`mtl::lu_iterative_refine<Working>(A, b, x, opt)`** (`operation/lu_iterative_refine.hpp`) — the dense counterpart of `sparse::iterative_refine`: factor A once in a low `Working` precision via `lu_factor`/`lu_solve`, then correct with a residual formed in the (deduced, higher) `Residual` precision. Universal-free and generic over the arithmetic type. Shares the sibling's `refine_options`/`refine_result` shape: best-iterate return, patience for a noisy low-precision residual, `rel_tol` convergence, and an opt-in `scaled` (scale-and-round) variant that carries the correction magnitude in `Residual` precision so narrow-exponent `Working` factors don't underflow.
- **`mtl::normwise_backward_error(A, x, b)`** (`operation/backward_error.hpp`) — the normwise backward error `||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf)` (Higham, "Accuracy and Stability of Numerical Algorithms", Thm 7.1), the natural quality/termination metric for refinement.
- Ported (Universal-free) from Universal's `blas/ext/solvers/luir.hpp` (`SolveIRLU`) + `blas/utes/nbe.hpp` as the first library piece of the Universal->MTL5 linear-algebra extraction; the posit/precision *experiments* driving it live in mp-ir (Universal epic #1204, phase #1207).

#### Range and test-matrix generators (BLAS extraction from Universal)
- **`mtl::generators::{arange, linspace, logspace, geomspace}`** (`generators/ranges.hpp`) — NumPy-style spacing vectors returning `vec::dense_vector<T>`, generic over the scalar type (Universal-free). `geomspace` computes a true geometric progression between its endpoints, fixing the historical Universal implementation that aliased `logspace` and treated the endpoints as exponents (`geomspace(1, 1000, 4) == {1, 10, 100, 1000}`).
- **`mtl::generators::magic`** (`generators/magic.hpp`) — magic square of order N (odd via the Siamese method, doubly-even via the complement method); singly-even orders throw `std::invalid_argument` pending support.
- Migrated from Universal's `sw::blas` as the first step of extracting the linear-algebra layer into MTL5 (Universal epic #1204, phase #1206).

#### Matrix/vector/tensor property predicate module (#244)
A cohesive set of runtime property and predicate queries as free functions in `namespace mtl`, built on the existing primitives (cholesky/lu/svd/eigenvalue/norms) with no new dependencies. Consistent tolerance policy: structural checks are exact-by-default and NaN-safe (`!(dev <= tol)`), while norm/factorization/spectral-backed checks use relative or scale-aware tolerances; verified on both the in-house and LAPACK paths.
- **Structural + vector predicates** (`operation/matrix_properties.hpp`, `operation/vector_properties.hpp`) — `is_square`, `is_empty`, `is_symmetric`, `is_hermitian`, `is_upper/lower/is_triangular`, `is_diagonal`, `is_banded`, `is_diagonally_dominant`; `is_zero`, `is_finite`/`has_nan`/`has_inf`, `is_normalized`/`is_unit`, `is_orthogonal_to`. O(n)/O(nnz), no factorization (#245)
- **Factorization-backed predicates** (`operation/factorization_properties.hpp`) — `is_spd`/`is_positive_definite` (symmetric + Cholesky), `is_singular`/`is_nonsingular`/`is_invertible` and `determinant` (LU), each on a copy so the caller's matrix is unchanged (#246)
- **Spectral / condition / rank** (`operation/spectral_properties.hpp`) — `spectral_radius`, `condition_number`, `rcond`, `numerical_rank`, `nullity`, wrapping the dense SVD and eigensolver (#247)
- **Rank-2 tensor predicates** (`tensor/properties.hpp`) — `is_symmetric`, `is_antisymmetric` (#248)
- **Orthogonality + inertia** — `is_orthogonal`/`is_unitary` (`AᴴA == I`) and `is_normal` (`AAᴴ == AᴴA`), plus `inertia` (the congruence-invariant `(positive, negative, zero)` eigenvalue-sign triple, Sylvester's law) and `is_indefinite`; inertia is backed by the symmetric eigensolver so it is robust for singular/semidefinite inputs (#249)

#### On-node threading (#221)
- **`mtl::detail::thread_pool`** — a persistent worker pool (no OpenMP/TBB, just the C++ standard concurrency runtime) with `run`, `parallel_for` (bit-identical contiguous chunking), and `parallel_reduce` (chunked, deterministic-per-thread-count). Threading is **off by default**; `MTL5_NUM_THREADS` (read once, clamped to hardware concurrency) sizes the pool, and `=1`/unset creates no workers and runs the original serial paths (#239)
- **Threaded kernels** on the pool — blocked GEMM (#239), row-/col-major GEMV and `axpy`/`scal` (#240), `dot`/`nrm2` and column-major GEMV (#241), and sparse SpMV `compressed2D * vector` (#242). The iterative and eigensolvers inherit the SpMV/L1 threading with no solver-code changes. Every kernel except the reductions is bit-identical across thread counts
- **Documentation** — an on-node threading reference (`docs/algorithms/on-node-threading.md`) and a performance-engineering case study on multi-core scaling (`docs/design/multicore-scaling-investigation.md`), including the SMT-affinity measurement pitfall and the corrected 6.3–6.9× GEMM scaling on 8 physical cores (#243)

#### On-node threading — factorizations, triangular solves, multi-loop GEMM, element-wise (#297)
Extends the #221 pool from the BLAS kernels to **every substantial dense and sparse kernel**, each **bit-identical across thread counts** (the parallel result equals the serial one exactly, `==`, not to a tolerance) and serial-by-default. Rolled out in ten batches; validated per batch against the serial path and under ThreadSanitizer.
- **Dense factorizations** — parallel trailing/column updates in LU and Cholesky (#298); parallel Householder reflector application (apply-left over columns, apply-right over rows) in QR (#300)
- **Sparse triangular solves via level scheduling** — a reusable, **value-agnostic** schedule (`sparse/factorization/level_schedule.hpp`) that stores *positions* into the factor's arrays (not values), so it survives a same-pattern in-place refactorization and reproduces the serial accumulation order exactly. Applied to sparse Cholesky forward + transpose (#301, #302), sparse LU lower/upper (#304), sparse LDLᵀ unit-lower + transpose (#305), and the supernodal LDLᵀ (#306) and supernodal LU (#309) solves
- **BLIS multi-loop GEMM** — the blocked GEMM's ic-only parallelism generalized to a 2D `jc_nt × ic_nt` thread grid: each jc-team's leader packs the shared B panel once and the team splits the ic-blocks, synchronized by a per-team barrier. Scales tall, wide, and square shapes; every C macro-block gets the same FMAs in the same order regardless of grid shape. The per-team barrier keeps a throwing worker in the barrier protocol so a team can never hang, rethrowing after the region joins (#311)
- **Element-wise expression sweeps** — the dense vector/matrix expression-template assignments (`y = a + b`, `C += A + B`, construct-from-expression, `+=`/`-=`) route through a `detail::parallel_ewise` helper; each output element is independent, so contiguous chunking is bit-identical (#312)
- **Encapsulation** — the numeric factor types privatize `L`/`U`/`D` and their coupled solve schedules behind a validating, strongly-exception-safe `set_factor(...)` with read-only accessors, so the schedule can never drift from the factor pattern; `solve()` rejects a missing factor with `std::logic_error` (#305–#309, #307/#308)
- **CI** — `threaded` (GCC+Clang, `MTL5_NUM_THREADS=4`) and `ThreadSanitizer` lanes on push and PRs, plus the design field guide `docs/design/parallelization-patterns-and-pitfalls.md` (#299/#303, #301)

#### Iterative solvers
- **`cg` accumulator policy** — the conjugate-gradient solver routes its two `dot` calls and the `mult` through an optional `accumulator_traits` accumulator (#158); default (`void`) behavior is unchanged, while `posit32`+quire shows a consistent accuracy gain over naive `posit32` (#238)

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
- **BLIS backend + expanded BLAS routine coverage** — `run_sweeps.sh`/`run_scaling.sh` gain a `blis` variant (CMake `BLA_VENDOR=FLAME`, auto-skipped if absent); the harness now benchmarks all core BLAS routines MTL5 implements (adds `axpy`/`scal` at L1, and L2/L3 as they land); `analyze_gate.py --reference` baselines against OpenBLAS, BLIS, or MKL (#227, #228)

#### Documentation & API
- **Capability assessment & expansion analysis** — `docs/design/capability-assessment-and-expansion.md`: a source-grounded assessment across functionality, performance (single/multi-thread), distributed-memory and hardware-accelerator readiness, with a maturity scorecard and a prioritized expansion roadmap (seeded the roadmap epic #220 and issues #221–#227) (#219)
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

#### Core BLAS Level-2 / Level-3 operators (#229)
- **Level 2**: `ger` (rank-1 update), `symv` (symmetric matrix-vector), `trmv` (triangular matrix-vector), `trsv` (triangular solve) (#230)
- **Level 3 triangular**: `trmm` (`B = alpha*A*B`), `trsm` (solve `A*X = alpha*B`) — left side, no transpose (#231)
- **Level 3 symmetric**: `symm` (`C = alpha*A*B + beta*C`, A symmetric), `syrk` (`C = alpha*A*Aᵀ + beta*C`), `syr2k` (`C = alpha*(A*Bᵀ + B*Aᵀ) + beta*C`); `syrk`/`syr2k` produce the full symmetric result (both triangles) (#232)
- Each is a generic templated function (any Matrix/Vector type and orientation, and custom number types) with optional external-BLAS dispatch for **column-major dense float/double**, mirroring the existing `gemv`/`gemm` gating. New `s/d` bindings added to `interface/blas.hpp` (`ger`, `symv`, `trmv`, `trmm`, `trsm`, `symm`, `syrk`, `syr2k`). BLAS leading dimensions are clamped to `max(1,m)` for empty inputs
- With this, MTL5 covers the core BLAS surface: **L1** dot/nrm2/axpy/scal, **L2** gemv/ger/symv/trmv/trsv, **L3** gemm/trmm/trsm/symm/syrk/syr2k

#### Benchmark results site, and CI coverage for code no runner compiled
- **Benchmark results site** (`docs/benchmarks/`) — a systems landing page indexing the machines (topology, vendor library versions, pinning policy and why it is mandatory on a hybrid P/E-core CPU) linking to a full per-system result page with figures and written assessments for the dense BLAS/LAPACK sweeps, the #82 gate, #108 GEMM scaling, the #297 kernel families and the sparse scoreboards. Adding a machine costs one page and one table row. The harness README is published alongside it at `/mtl5/benchmarks/` (#320)
- **CI: benchmark targets are now compiled on every push and PR** in two configurations — native, and with BLAS/LAPACK + KLU + SuperLU so the `#ifdef MTL5_HAS_KLU` / `MTL5_HAS_SUPERLU` comparison paths are exercised. `MTL5_BUILD_BENCHMARKS` defaults `OFF` and no workflow enabled it, so `benchmarks/*.cpp` was compiled by no runner; that is how the `lu_numeric` breakage below survived. Build-only: timed runs need pinned cores and a quiet machine, which shared runners cannot provide (#329)
- **CI: a Highway SIMD lane** (`-DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_FAST_GEMM=ON`, GCC + Clang) that builds *and runs* the suite. No workflow set `MTL5_WITH_HIGHWAY`, so every lane built the `size == 1` scalar fallback in `simd/batch.hpp` and the Highway specialization — the SIMD register type under the blocked GEMM — was compiled nowhere, while that flag is what the `native-fast` variant and every published benchmark number use. The fetched Highway build is cached on the pinned tag. Two guards against the lane passing while covering nothing: the workflow asserts a `libhwy` artifact was built, and `simd/test_batch.cpp` static-asserts `batch::size > 1` under `MTL5_HAS_HIGHWAY` on x86-64 (the previous assertion was `>= 1`, which passes in both configurations) (#330)

### Changed
- **The epic #82 acceptance gate now asserts the MEDIAN of the per-size ratios**, with a new `--floor` (default 0.70) that no individual size may fall below. It previously failed if *any* size fell under the threshold — gating on the minimum, the noisiest statistic available. Two runs of the identical protocol on the same idle machine failed at disjoint sets of sizes; the median is reproducible to 0.2 points where per-size values swing up to 6. Raising the iteration count does not help: within-run stddev is already 0.3–2.2%, so the variance is *between* runs, not within them (#327)
- **`run_scaling_297.sh` defaults to `SPARSE_SIZES=100,160`** (was `200,320`, which did not complete — a `T=1` run was killed after 3 h 28 min without finishing its first size). Sparse cost is dominated by the untimed factorization and grows steeply with the grid side: 11 s at 100, 3 m 38 s at 160, 8 m 46 s at 200, *per thread count*. The cost table is recorded next to the default so it is raised deliberately (#321)
- **CI runs when a draft PR is promoted to ready.** `pull_request` declared no `types`, so the default `[opened, synchronize, reopened]` applied and `ready_for_review` fired nothing. Since the Tier 2 `regression` job gates on `draft == false`, a PR opened as a draft only ever had runs in which it skipped — silently, because `gh pr checks` reports green when a job skips rather than fails (#333)
- **`cmake --preset release` builds something.** It set tests and examples `OFF` with benchmarks defaulting `OFF`, leaving a configured tree with zero buildable targets, since the only target is an `INTERFACE` header-only library (#319)
- **`mtl::svd` now uses one-sided Jacobi instead of an alternating-QR iteration, and its default `tol` is machine epsilon rather than `1e-10`.** Results change for existing callers: they are correct where they previously were not, and accurate to ~1e-15 against LAPACK rather than ~1e-10 at best. The old default would have capped accuracy several orders short of what the method achieves; pass `1e-10` explicitly for the previous looser behaviour. Singular values are non-negative and returned in descending order, and `U` is completed to a genuine orthogonal basis for rank-deficient or `m > n` input (#337)
- **`mtl::ldlt_bk_factor`'s stored factor now matches LAPACK `dsytrf` entry for entry.** Code reading the raw factor from a build before this release will see different values in the L columns whenever a pivot interchange occurred; `ipiv` is unchanged and was always LAPACK-compatible (#335)
- **`mtl::dot` / `dot_real` now dispatch to BLAS `?dot`** when types qualify (consistency with `two_norm`); both `dot` and `two_norm` BLAS paths guard the `int` length cast against overflow (#81)
- **Benchmarks rewritten** to the single-path public-API model; deleted the `Native/Blas/Lapack` policy-tag harness (#81)
- **CI hardening**: all GitHub Actions pinned to commit SHAs with `persist-credentials: false` (#64); sccache gated to trusted runs to prevent GHA cache poisoning (#74); Dependabot action bumps (#66–#71)
- Benchmark README: corrected CMake option names (`MTL5_WITH_BLAS/LAPACK`) and added Intel MKL (`BLA_VENDOR=Intel10_64lp`) instructions (#75)
- `.gitignore`: ignore Claude Code per-user/runtime files (#76)
- **CI now exercises the LAPACK dispatch paths** — a `lapack` job (Linux GCC + Clang, `-DMTL5_WITH_LAPACK=ON`) builds and runs the external-library `geev`/`syev` paths, which the default LAPACK-free matrix never compiled (#212)

### Fixed
- **`bench_klu` / `bench_superlu` did not compile** — they referenced `blk.L` / `blk.U` on `lu_numeric`, whose members are private (`L_` / `U_`). Nothing caught it because no CI job built the benchmark targets; they had not been compiled since `lu_numeric` was encapsulated. Now use the public `factorL()` / `factorU()` accessors (#319)
- **`bench_sparse`'s 3-D Laplacian grid is clamped.** `g3 = cbrt(g^2)` sized it so its DOF count matched the 2-D case, but `max(8, ...)` bounds `g3` only from *below* despite the comment claiming otherwise. Equal DOF is not equal cost — a 3-D Laplacian has far worse fill growth under Cholesky — so raising the 2-D size implicitly selected an intractable 3-D problem. Clamped to 32; grid side 200 went from *killed after 3 h 28 min* to 8 m 46 s (#322)
- **`mtl::svd` returned all-NaN singular values for ~30% of ordinary symmetric matrices**, and values wrong by up to several hundred percent for many of the rest; `condition_number`, `rcond`, `numerical_rank` and `nullity` all inherited it through `detail::singular_values`. Two independent defects. (a) `householder` formed `sum(x(i)^2)` unscaled and guarded only *exact* zero, so a tiny sigma reached `beta = 2*v0*v0/(sigma + v0*v0)` with both terms flushed to zero (0/0), while `v(i) /= v0` overflowed; `apply_householder_left/right` then evaluated `beta * v(i) * w` as `0 * inf`. Since `qr_factor` stores `v` below the diagonal, the poison spread into the factor. (b) The alternating-QR iteration converged to ~1e-9 and then **corrupted its own answer** — 136% wrong by iteration 1000 on a 10×10 SPD case — while its convergence test (off-diagonal mass / diagonal mass) kept shrinking and so read as converged; `max_iter = 100*max(m,n)` meant the corrupted value was the one returned. Replaced with **one-sided Jacobi**. Verified against LAPACK `dgesvd` over 1800 random matrices: 0 NaN, worst relative deviation 3.0e-15 (#337)
- **`mtl::itl::pc::ilu_0::solve` counted the diagonal in its back-substitution off-diagonal sum**, computing `x(i) = (y(i) - sum_{j>=i} U(i,j)*x(j)) / U(i,i)` — subtracting `U(i,i)*x(i)` and then dividing the difference by that same `U(i,i)`. Wrong for every input: on a diagonal matrix, where ILU(0) is exact, it returned `(b - d*b)/d` instead of `b/d`. The diagonal is now stored only in `u_diag_`, making `u_rows` strictly upper by construction so no consumer can forget to skip it. On a tridiagonal system where the factorization is exact, preconditioned BiCGSTAB now converges in 1 iteration instead of 11 (#323)
- **`mtl::ldlt_bk_factor` produced a wrong solution whenever a pivot interchange occurred**, under `info == 0` — normwise backward error ~1e-1 instead of ~1e-16, in 4560 of 5700 random symmetric matrices. `symmetric_swap` permuted the L columns written by *earlier* steps, putting the stored factor in the "single global P" convention while `ldlt_bk_solve` replays the interchanges one step at a time (as LAPACK's `dsytrs` does); the two agree only when nothing is interchanged, which is why the n=2 and n=3 tests passed. The swap is now restricted to columns `>= k`. The pivot *search* was always correct — `ipiv` matched `dsytrf` before and after — and the stored factor now matches `dsytrf` entry for entry (#335)
- **`mtl::two_norm<Acc>` / `frobenius_norm<Acc>` did not compile for either non-trivial accumulator configuration.** They rounded the accumulator out to the *accumulator type* and took its square root — a no-op for a plain arithmetic accumulator, which is why `two_norm<double>` worked and hid it, but an `fma_accumulator<T>` or a quire for configurations 2 and 3, neither of which has a `sqrt`. Now rounds out to the accumulator's own arithmetic precision via the new `accumulator_round_type` (#324)
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
