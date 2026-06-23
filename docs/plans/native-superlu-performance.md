# Native Supernodal LU Performance: Reaching SuiteSparse SuperLU Parity

## Status

**Planning.** MTL5 has an external SuperLU binding (`include/mtl/interface/superlu.hpp`)
but **no native SuperLU** — no native supernodal unsymmetric LU. Its native sparse
LU is scalar, non-supernodal Gilbert–Peierls (`sparse_lu.hpp`, and per-block in
`native_klu.hpp`); the only native *supernodal* factorization is the symmetric
LDLᵀ landed in #178. A native SuperLU must become a genuine alternative to the
external binding for general unsymmetric matrices, especially the fill- and
flop-heavy ones where SuperLU's dense panels pay off.

This document is the plan to build native supernodal unsymmetric LU and close the
gap to **SuiteSparse SuperLU parity**.

## Principle: SuiteSparse SuperLU is the floor, not the ceiling

SuperLU is a proof of existence. The performance it delivers is the **bare
minimum** native SuperLU must match. The *ceiling* is higher than SuperLU can
reach: a native implementation that is generic over the number type gains
**mixed precision** (store the supernode panels low, accumulate the dense block
updates high, recover accuracy with iterative refinement, and run with custom
types like posit/LNS) — which the fixed-precision C library cannot offer. That
mixed-precision capability is the *reason* to own a native SuperLU and is MTL5's
whole thesis.

## This is build-then-tune, not just tune — the key difference from #138

The native-KLU epic (#138) was a *parity* effort: native KLU already existed and
competes in the **same algorithm class** as SuiteSparse KLU (scalar,
non-supernodal Gilbert–Peierls LU, no BLAS), so its gap was pure constant-factor
implementation deficit.

SuperLU is a **different algorithm class**: supernodal left-looking LU with
**dense BLAS-3 panel updates** and **threshold partial pivoting**. Today's native
LU is scalar and column-at-a-time, so on matrices with dense structure it is not
merely a constant factor behind — it is the wrong kernel. Closing the gap
therefore requires *building* the supernodal machinery for the unsymmetric,
pivoting case, not just removing overhead.

The good news: most of that machinery now exists.

- **#178** delivered native supernodal symmetric LDLᵀ: supernode detection
  (`include/mtl/sparse/analysis/supernodes.hpp`), the dense-panel left-looking
  numeric kernel, and the accumulator-policy boundary (`accumulator_traits`).
- The **mixed-precision dense kernels** are in place: widening dot/GEMV, the
  accumulator policy, and the custom-accumulator dispatch rule (#160, #162, #163,
  #164, #165).
- The generic **iterative-refinement** loop (`sparse/iterative_refine.hpp`) and
  the **CHOLMOD/SuperLU/UMFPACK** bindings exist as oracles.

So the unsymmetric case mainly adds the **column elimination tree**, **unsymmetric
supernode detection**, and the hard part — **panel partial pivoting interleaved
with supernodal updates**. The dense block update reuses the existing
mixed-precision GEMM path.

## Measured baseline (the gap to establish)

Unlike #138, there is no native supernodal LU to measure yet. **Phase 0**
establishes the scoreboard by benchmarking the *existing* native LU
(`sparse_lu` / `native_klu`) against SuiteSparse SuperLU on an unsymmetric suite,
quantifying both the constant-factor gap and the kernel-class gap (which grows
with dense structure). Those numbers become the baseline every later phase is
gated against.

## Where native loses to SuperLU (the gap to close)

1. **No supernodes / no BLAS-3** — native LU updates one scalar column at a time;
   SuperLU groups columns into dense panels and does the trailing update as
   GEMM/TRSM. This is the dominant gap on matrices with dense fill structure.
2. **No column elimination tree / unsymmetric supernode symbolic** — native LU
   has no supernodal symbolic for the unsymmetric (column-etree) case; #178's
   detection is symmetric-pattern only.
3. **Pivoting × supernodes** — threshold partial pivoting perturbs the supernode
   structure; SuperLU handles this with relaxed/amalgamated supernodes and panel
   pivoting. This is the core algorithmic risk.
4. **No analyze/factor/refactor split** — native LU re-searches the pivot pattern
   every solve; SuperLU separates analyze from factor and supports numeric
   refactorization of the same pattern (the transient-SPICE / mp-spice win).
5. **Constant-factor / data-structure overhead** — `compressed2D`/`inserter` in
   the hot path and per-block allocation, as catalogued in #138.

## Definition of done

On a fixed unsymmetric suite — 2D convection–diffusion grids (scalable synthetic)
and a SuiteSparse unsymmetric set (`wang3`, `wang4`, `raefsky3`, `ecl32`,
`twotone`, `bbmat`, plus circuit `rajat*`/`ASIC_*`) — **single-threaded** native
supernodal LU is:

- within **1.5×** of SuiteSparse SuperLU in factor+solve time, and
- within **1.2×** in fill (nnz of L+U), and
- within **1.2×** in peak memory,

**and**, beyond what SuperLU offers, a mixed-precision mode (low-precision
supernode storage + high-precision accumulation) that recovers full accuracy via
iterative refinement on the suite. SuperLU is single-threaded, so the time
comparison is honest apples-to-apples; thread-level parallelism is a separate,
beyond-SuperLU effort. 1.5× acknowledges the long tail of matching a tuned C
library; tighten afterward if warranted.

## Plan — six phases, each gated against SuiteSparse SuperLU

### Phase 0 — Benchmark scoreboard (`bench_superlu`, mandatory first)
A SuperLU benchmark under `benchmarks/` running the existing native LU vs external
SuperLU on the unsymmetric suite, reporting per-phase time (order / symbolic /
numeric / solve), fill (nnz L+U), peak memory, and residual, with the
native/SuperLU ratio per matrix. Mirrors `bench_klu`. Nothing else starts until
we can measure every change against the floor.
*Gate:* scoreboard emitted; baseline ratios (and the kernel-class gap vs dense
structure) recorded.

### Phase 1 — Column elimination tree + unsymmetric supernode symbolic
Build the column elimination tree and unsymmetric (column-etree) supernode
detection, extending `analysis/supernodes.hpp` beyond the symmetric case from
#178. Produce the supernode partition, per-supernode row structure, and an L/U
nnz prediction.
*Gate:* supernode partition and predicted fill validated against a symbolic
reference and SuperLU's reported fill on the suite.

### Phase 2 — Supernodal numeric kernel with threshold partial pivoting
The core. Left-looking supernode-panel numeric with **panel partial pivoting**
and **relaxed supernodes**, the dense block update routed through the
mixed-precision GEMM path (`mult<Accumulator>` / `accumulator_traits`). Emits a
standard CSC L/U so the existing solve and iterative refinement are reused.
*Gate:* factorization correct vs SuperLU on the suite (residual parity); native
factor within ~3× of SuperLU on a dense-structured matrix.

### Phase 3 — 2D panel blocking & supernode-panel update efficiency
2D blocking of the supernode-panel update, cache/BLAS-3 tuning of the dense
kernels, gather/scatter and relative-index efficiency.
*Gate:* native within ~1.5–2× of SuperLU factor+solve time across the suite.

### Phase 4 — analyze / factor / refactor split (the SPICE win)
Separate analyze (column order + symbolic + supernodes) from factor (numeric +
pivoting) and add a numeric-only **refactor** reusing the pivot/supernode
pattern, for transient SPICE and the mp-spice study.
*Gate:* refactor far faster than factor, matching SuperLU's factor/refactor ratio.

### Phase 5 — Scaling, pivot robustness, mixed precision, final parity sweep
Row/column scaling and pivot strategy for hard unsymmetric matrices; integrate
the mixed-precision accumulator + iterative refinement end-to-end; full suite
sweep proving the Definition of Done, locked behind a regression.
*Gate:* within 1.5× time / 1.2× fill / 1.2× memory across the suite, and the
mixed-precision low+IR mode recovers full accuracy.

## Relationship to existing issues

- Builds directly on **#178** (native supernodal LDLᵀ — supernode detection,
  dense-panel kernel, accumulator-policy boundary). This is the symmetric →
  unsymmetric supernodal progression.
- Complements **#138** (native KLU parity): KLU is the scalar/circuit path,
  SuperLU is the supernodal/general-unsymmetric path.
- Reuses the mixed-precision dense-kernel work (#160, #162, #163, #164, #165) and
  the iterative-refinement core (#119).

## Highest leverage

Phase 2 (supernodal numeric + panel pivoting) is the core risk and the bulk of
the value. Phase 4 (analyze/refactor split) is highest leverage for the SPICE /
mp-spice workload. The mixed-precision mode (Phases 2 and 5) is the unique
capability native SuperLU has that the external library never will.

## References

- Demmel, Eisenstat, Gilbert, Li, Liu, "A Supernodal Approach to Sparse Partial
  Pivoting", SIAM J. Matrix Anal. Appl., 20(3), 1999 (SuperLU).
- Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006.
- Ng & Peyton, "Block sparse Cholesky algorithms on advanced uniprocessor
  computers", SIAM J. Sci. Comput., 14(5), 1993.
- Gilbert & Peierls, "Sparse Partial Pivoting in Time Proportional to Arithmetic
  Operations", SIAM J. Sci. Stat. Comput., 1988.
- Eisenstat & Liu, "Exploiting Structural Symmetry in Unsymmetric Sparse Symbolic
  Factorization", SIAM J. Matrix Anal. Appl., 1992.
