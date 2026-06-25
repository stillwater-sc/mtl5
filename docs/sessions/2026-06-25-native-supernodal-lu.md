# Session: Native supernodal LU — core kernel, refactor/scaling, and the honest performance verdict

**Date**: 2026-06-25
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Drive the native supernodal LU (SuperLU) epic (#186) from its symbolic foundations
through the numeric core and into the performance work — building a general
unsymmetric supernodal factorization that is a genuine alternative to the external
SuiteSparse SuperLU binding, with the mixed-precision capability the fixed-precision
C library can never offer. Then, crucially, **measure** whether the supernodal BLAS-3
machinery actually closes the speed gap — and report the result honestly.

## Context

MTL5 stays free of any external number library; the Universal composition lives in
the sister **mp-spice** repo. SuperLU is a *different algorithm class* from the
existing scalar Gilbert–Peierls LU: supernodal left-looking with dense BLAS-3 panel
updates and threshold partial pivoting. SuiteSparse SuperLU is the proof of
existence — its FP64 speed is the bar — but the *ceiling* is higher: a native,
number-type-generic implementation gains mixed precision. The epic deliberately
follows a build-then-tune arc.

## Work Completed

### Phase 2 — supernodal numeric kernel with threshold partial pivoting (#182 → #190)
The core and the bulk of the value. Native left-looking Gilbert–Peierls LU that:
- groups columns into **supernodes** (dynamically, since pivoting precludes a fully
  static structure), capped by the Phase-1 relaxed-supernode boundaries;
- applies each closed supernode as a dense block update through the
  **`accumulator_traits` accumulator** boundary (the mixed-precision seam);
- does **threshold partial pivoting** + Eisenstat–Liu symmetric pruning + per-column
  reach DFS, reusing the scalar `sparse_lu` machinery.

Validated bit-for-result against scalar `sparse_lu` to machine precision across
dense / banded / convection-diffusion / random, plus mixed-precision accumulator
and iterative-refinement tests. A nesting bug in dynamic supernode formation
(`cand_k == cand_{k-1} \ {current pivot}`, not the previous pivot) was found and
fixed during bring-up.

### COLAMD garbage-collection bug (#189 → #191)
While validating the supernodal column ordering, the AMD/COLAMD minimum-degree GC
compaction was found to mis-restore each element's first entry, corrupting the
quotient-graph pointers — but only once fill exhausts the elbow room (the AᵀA
pattern of a 2-D 5-point grid at n ≥ 64 segfaulted; dense/small inputs never hit
it). Fixed to the CSparse compaction order; regression test over growing grids
added. A real pre-existing bug, surfaced by the new workload.

### Phase 4 — analyze/factor/refactor split (#184 → #193)
`supernodal_lu_refactor`: a numeric-only recompute that reuses a prior
factorization's column order, pivot sequence, and L/U pattern — no ordering, reach
DFS, pivot search, or supernode detection. **1.9–3.2× faster** than a full factor
(the transient-SPICE / mp-spice path). Mirrors the proven `sparse_lu_refactor`.

### Phase 5 — row equilibration + mixed-precision IR (#185 → #193)
Opt-in `scale=true` factors `R·A` (`r=1/max|row|`) for pivot stability in low/mixed
precision; RHS row-scaled in `solve()`, `x` unchanged. The mixed-precision
iterative-refinement half of Phase 5 already shipped in Phase 2
(`supernodal_lu_solve_refined`). Tests assert correctness + equivalence to unscaled
(equilibration improves *worst-case* stability, not per-instance monotonic accuracy
— a non-robust assertion was caught and removed).

### Phase 3 — the performance pursuit, measured and parked (#183, closed not-planned)
This is the session's real story. Three measured BLAS-3 iterations on the panel
kernel (groundwork preserved on `perf/superlu-block-kernel`):
- **3a** per-column BLAS-2 block update (TRSV+GEMV) — ~scalar parity.
- **3b** panel BLAS-3: batched cross-supernode TRSM+GEMM over W-column panels
  (PR #192). Correct (found + fixed a subtle `Lp[kp]`-before-`close_supernode`
  bug), but ~scalar parity. Wider panels (16→64) didn't help.
- **3c** subset tracking (apply each supernode only to reached columns) — still
  parity; banded slightly worse.

Then the decisive step: **profiling** before any further rewrite. `perf` is locked
down in this environment, so the kernel was instrumented with `#ifdef`-guarded
phase timers. The breakdown of dense-400 factorization:

| phase | share |
|---|--:|
| per-column scalar (intra-panel SAXPY + pivot + emit + EL-prune + supernode detect) | **34%** |
| batch overhead (many tiny per-column TRSV calls, sort/unique) | **25%** |
| reach DFS | **18%** |
| **batch GEMM — the BLAS-3 "win"** | **14%** |
| scatter memory (the planned `Pd` aligned-storage target) | **9%** |

Conclusions, all data-backed:
1. The entire 3a/3b/3c BLAS-3 effort optimized the **wrong 14%**; the planned `Pd`
   aligned-storage retarget targets only the 9% scatter — it would not have paid off.
2. **No cheap win exists** — removing the per-supernode buffer zero-fill
   (`assign`→`resize`) produced no measurable change.
3. The 77% majority is scalar/serial work. Reaching SuperLU parity needs a full
   reimplementation (BLAS within-panel diagonal-block factorization "Mechanism B" +
   batched TRSV + single panel symbolic) — a dedicated multi-day project.

Decision: **park FP64 parity as not-planned (#183).** MTL5's differentiator is
mixed precision (delivered and working); matching a 25-year-tuned C library's raw
FP64 speed is low-ROI. The correct, mixed-precision-capable kernel is merged on
`main`; the profiling target list is preserved on #183 for any future revival.

## Decisions & Rationale

- **Measure before rewriting.** The most valuable output of the perf work was not
  code but the profile that overturned the strategy — it proves the next effort
  shouldn't repeat the BLAS-3-panel dead end.
- **Don't merge correct-but-not-faster code.** PR #192 (panel BLAS-3) was closed
  unmerged; it adds complexity with no net speedup. Kept as a referenced branch.
- **Don't assert non-robust benefits.** The float-equilibration "always better"
  test (and earlier an accumulator/ARM test) was reframed to correctness +
  equivalence — equilibration helps worst-case, not every instance.
- **Scope honesty in the epic.** #186 updated to mark Phase 3 parked and Phase 5's
  parity-sweep DoD parked while ticking the delivered robustness work; #183 closed
  `not_planned`.

## Outcome

The native supernodal LU epic is **functionally complete and merged**: column etree
+ unsymmetric supernode symbolic, supernodal numeric with threshold partial
pivoting, analyze/factor/**refactor** (1.9–3.2× reuse), row equilibration, and
mixed-precision iterative refinement — all matching SuiteSparse SuperLU to machine
precision — plus a benchmark scoreboard and a real COLAMD bug fixed along the way.
FP64 single-factor speed parity is the one item deliberately left as documented,
data-backed future work rather than half-built.

## Issues / PRs

- Merged today: #190 (Phase 2 numeric), #191 (COLAMD GC fix), #193 (Phase 4 refactor + Phase 5 scaling)
- Closed: #192 (panel BLAS-3, unmerged groundwork), #183 (FP64 parity — not planned)
- Epic: #186 (status updated — correctness complete, parity parked)
- Branch retained: `perf/superlu-block-kernel` (panel/BLAS-3 groundwork)
