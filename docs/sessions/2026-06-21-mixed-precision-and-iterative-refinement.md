# Session: Mixed-precision tensor BLAS, native-KLU refactor, and iterative refinement

**Date**: 2026-06-21
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Finish the native-KLU performance work (the analyze/factor/refactor split), run
the mixed-precision SPICE study that the native KLU unlocked, distill its findings
into a reusable, dependency-free `iterative_refine` core, and design + build a
**mixed-precision tensor-BLAS** capability (the *Element → Accumulate → Result*
model) across MTL5's dense operations.

## Context

MTL5 stays free of any external number library; the Universal (number systems)
composition lives in the sister **mp-spice** repo. The native KLU solver had
reached competitive one-shot parity with SuiteSparse; this session pushed into
the *workload* win (refactor) and the *accuracy* story (mixed precision), then
generalized the lessons back into MTL5 as first-class, type-agnostic machinery.

## Work Completed

### Native KLU — analyze/factor/refactor (Phase 4, epic #138)
- **`sparse_lu_refactor`** (#153): refactorize a same-pattern matrix by reusing a
  prior factorization's symbolic structure + pivot sequence, recomputing only
  values — no BTF/ordering, no reach DFS, no pivot search. Enabling change:
  `sparse_lu_numeric` now stores L/U columns in ascending row order (topological
  replay; diagonal-first-L / diagonal-last-U preserved). **2.1× faster than
  factor** (Poisson 200²).
- **`native_klu_refactor`** (#154): drives each BTF block through the 4a engine,
  reusing per-block orderings + pivots; recomputes row scaling + B. **~2.2×
  faster than factor** (add32 7.2→3.4 ms). The SPICE-transient win.
- **Residual-fill tuning — negative result** (#150): hypothesized a looser pivot
  threshold would cut rajat30's residual 1.7× fill; *measured no improvement*
  (row scaling already aligns pivoting). Recorded the negative result and
  re-scoped the lever to ordering quality (#151).

### Generic iterative refinement (#119)
- **`mtl::sparse::iterative_refine`** (#166): refine `A x = b` through any
  factorization exposing `solve`, with the residual in a (typically higher)
  `Residual` precision. Universal-free; optional **scaled** variant (normalize the
  residual before the low-precision correction solve); `rel_tol` early stop;
  returns the best iterate.
- **Patience-based termination** (#167 → #168): tolerate a few non-improving steps
  so a noisy low-precision residual reaches its floor (posit<16,2> IR reached
  2.7e-12 vs 6.8e-12 with a strict break), while genuinely stalled types still
  exit early.
- Reconciled the older sparse follow-ups: **#118 row scaling** closed (shipped in
  v5.5.0); **#119** delivered as the reusable core above.

### Mixed-precision tensor-BLAS (epic #157)
The headline new capability — three independent precisions per operation:
*element* (storage), *accumulator* (compute), *result* (serialize), with the
accumulate→output conversion **fused into the store** (no separate downcast pass).

- **#158 / #169 — foundation**: `mtl::math::accumulator_traits<Acc, Value>`
  (shared by sparse + operation), generalized round-out **`value<Result>`**;
  `sparse_lu`'s trait inherits it (non-breaking across repos).
- **#159 / #170 — `dot` / `dot_real`**: `Accumulator`/`Result` policy.
- **#161 / #171 — `gemm`/`mult`** (headline): accumulator policy, **result type
  inferred from `C`**, conversion fused into the store.
- **#160, #162 / #173 — `gemv` and the sum-of-squares norms** (nrm2, frobenius).
- **#163 / #172 — dispatch rule**: `interface::accumulator_allows_blas_v` — any
  non-default accumulator forces the native kernel *even for float/double*
  (external BLAS can't honor it); proven with a counting accumulator (dot→5 not
  135, gemm→k not 42).
- **#164 / #174 — standalone `mtl::convert`**: element-wise re-quantization for
  *non-fused* re-typing (explicitly *not* the accumulate→store path).
- **#165 / #175 — SIMD widening dot**: `batch::load_widen` (Highway
  `Rebind`+`PromoteTo`) + `reduce_dot_widen` for float→double; **~2.6× over
  scalar** (1.66→4.33 Gelem/s), verified on both the scalar and Highway backends.

Default `Accumulator = void` is byte-identical everywhere; existing BLAS/SIMD/
regression suites unchanged; ASan/UBSan clean throughout. Epic closed; the one
remaining optimization — **GEMM blocked-kernel SIMD widening** — split to #176.

### mp-spice mixed-precision study (the motivation, in the sister repo)
Driven on `add32`; full write-up in mp-spice `docs/mixed-precision-klu-study.md`:
- **Posits are the best 16/32-bit carriers** (best direct-solve accuracy).
- **For iterative refinement, dynamic range beats mantissa width**: `float`,
  `bfloat16`, and posits refine to ~1e-13; IEEE `half` **stalls** (5.7e-5) despite
  a better direct solve and 3× the mantissa — its 5-bit exponent underflows the
  small corrections. This explained the earlier `cfloat<16,5>` non-convergence.
- **Scaled IR rescues `half`** (→ 2.8e-14): the stall is a residual *representation*
  problem, fixed by carrying the correction magnitude in double.
- **The quire (exact factorization accumulation) is washed out by IR** — wrong
  lever; IR already absorbs the factorization's accumulation error.
- The MTL5 **`#122` accumulator seam** (#155) + **`native_klu_factor` accumulator
  threading** (#156) made the quire injectable from mp-spice without MTL5
  depending on Universal; mp-spice's `mixed_refine`/`mixed_refine_scaled` now
  **delegate to MTL5's `iterative_refine`** core.

### Documentation
- **"Measuring Solver Accuracy"** (#152): published reference on residuals, norms,
  absolute vs relative error, and backward-vs-forward error / conditioning — the
  cross-cutting metric every solver and benchmark reports.
- KLU optimization tracker updated with the residual-fill negative result.

## Key Decisions

- **Element → Accumulate → Result** with a **fused-convert epilogue** (output type
  from `C`), not a separate convert pass — chosen specifically to avoid wasting
  output/NoC bandwidth (writing fp32 then re-streaming to downcast).
- **`add_product` (not `sub_product`)** as the trait primitive — a dot product /
  quire is a *sum* of products; the caller negates for a subtraction.
- **MTL5 stays Universal-free**: the canonical trait lives in `mtl::math`, the
  quire adapter and all mixed-precision *experimentation* live in mp-spice.
- mp-spice **tracks MTL5 `main`** (tight coupling during co-development of custom
  algorithms) rather than pinning a release tag.

## Outcome

Epic #138 (native KLU performance) and epic #157 (mixed-precision tensor BLAS)
both complete; #119 (iterative refinement) delivered as a reusable core. Across
MTL5, mp-spice, and mixed-precision-dsp: no open PRs at session end. Tracked
follow-ups: **#176** (GEMM SIMD widening) and **#151** (KLU ordering quality).
