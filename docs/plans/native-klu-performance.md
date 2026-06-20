# Native KLU Performance: Reaching SuiteSparse Parity

## Status

**Planning.** Native KLU is correct and no longer O(n²) (the O(flops) factorization
landed in #127/#129; near-linear AMD/COLAMD ordering in #128/#117). But it is not
yet performance-competitive with SuiteSparse KLU, and it must be: a native sparse
direct solver has to be a genuine alternative to the external binding for every
sparse matrix, fill-heavy circuit matrices included.

This document is the plan to close the gap to **SuiteSparse KLU parity**.

## Principle: SuiteSparse KLU is the floor, not the ceiling

SuiteSparse KLU is a proof of existence. The performance it delivers is the
**bare minimum** our native KLU must match. Crucially, KLU is a **scalar,
non-supernodal, left-looking Gilbert–Peierls LU** — it uses no BLAS and no
supernodes (that is its design point for circuit matrices, which have little
dense structure). Therefore parity is achievable in the *same algorithm class*;
there is no inherent kernel disadvantage to hide behind. Any slowdown is an
implementation deficit we can remove.

## Measured baseline (the gap to close)

On `Rajat/rajat30` (643,994 × 643,994, 6,175,377 nnz):

| stage | native | external KLU |
|-------|--------|--------------|
| load (`mm_read`) | 1.7 s | — |
| BTF | 1.7 s (nblocks=11,705, **largest block=632,196**) | (part of the 7.4 s) |
| factor+solve | **did not finish in 600 s** | **7.4 s** |

On 2D Poisson 256×256 (n=65,536, single block, benign fill):

| | native | external KLU | ratio |
|---|--------|--------------|-------|
| factor+solve | 0.679 s | 0.190 s | **3.6×** |

The 3.6× on a benign block is pure constant-factor waste (same algorithm). The
>80× on rajat30 is that **plus fill explosion** from the wrong block ordering.

## Where we lose to KLU (same algorithm)

1. **Fill explosion on unsymmetric blocks** — we order each block with
   COLAMD (AᵀA); KLU defaults to **AMD on A+Aᵀ** of the block. On indefinite,
   highly-unsymmetric circuit blocks, threshold pivoting deviates from the
   COLAMD column order and actual fill blows up. (The rajat30 killer.)
2. **No symmetric pruning** — our left-looking factorization computes each
   column's reach with a full Gilbert–Peierls DFS over previously-computed L
   columns. KLU uses **Eisenstat–Liu symmetric pruning** so the DFS traverses a
   pruned structure. This is the single biggest reason KLU's scalar GP-LU is
   fast, and it dominates the benign-block constant factor.
3. **Heavyweight data structures in the hot path** — we build `B = P·A·Q` and
   extract each block through `compressed2D` + `inserter`, and allocate fresh
   `std::vector`s per block. KLU works on raw CSC integer arrays with
   pre-sized, chunk-grown storage.
4. **No analyze/factor/refactor split** — we re-search pivots on every solve.
   KLU separates `klu_analyze` (once) from `klu_factor` and **`klu_refactor`**
   (numeric reuse of the pivot pattern). Transient SPICE performs one analyze
   and thousands of refactors of the *same pattern*.
5. **BTF + scaling** — our BTF (Kuhn matching + Tarjan SCC) has constant-factor
   room versus KLU's `maxtrans`/`strongcomp`, and we apply no row scaling.

## Definition of done

On a fixed suite — 2D Poisson 32²…512² and a SuiteSparse circuit set
(rajat14, rajat30, circuit_4, circuit5M, ASIC_*) — **single-threaded** native
KLU is:

- within **1.5×** of SuiteSparse KLU in factor+solve time, and
- within **1.2×** in fill (nnz of L+U), and
- within **1.2×** in peak memory,

**including rajat30**. KLU is single-threaded, so this is honest apples-to-apples
(thread-level parallelism is a separate, beyond-KLU effort). 1.5× (not 1.0×)
acknowledges the long tail of matching a 25-year-tuned C library; we tighten to
1.2× afterward if warranted.

## Plan — six phases, each gated against SuiteSparse KLU

### Phase 0 — Benchmark scoreboard (mandatory first)
A KLU benchmark under `benchmarks/` running native vs external on the suite,
reporting per-phase time (BTF / order / symbolic / numeric / solve), fill, and
peak memory, with the native/KLU ratio per matrix. Nothing else starts until we
can measure every change against the floor.
*Gate:* scoreboard emitted; baseline ratios recorded.

### Phase 1 — Ordering: AMD-per-block (fix the fill)
Switch per-block ordering from COLAMD(AᵀA) to **AMD on A+Aᵀ** of each block
(near-linear via #128), matching KLU's default; keep COLAMD selectable.
*Gate:* native fill within ~1.3× of KLU on the circuit suite; **rajat30
completes** in sane time.

### Phase 2 — Symmetric pruning in the left-looking solve
Implement Eisenstat–Liu symmetric pruning so the per-column reach DFS traverses
pruned structure. Applies to every matrix; closes most of the benign-block
constant factor.
*Gate:* native factor within ~2× of KLU on 2D Poisson.

### Phase 3 — Kernel & data-structure efficiency
Remove `compressed2D`/`inserter` from the hot path; build B and extract blocks
into raw CSC integer arrays; pre-size and chunk-grow L/U; tighten gather/scatter
and BTF constants.
*Gate:* native within ~1.3× of KLU on Poisson and the circuit suite.

### Phase 4 — analyze / factor / refactor split (the SPICE win)
Separate `klu_analyze` (BTF + ordering + symbolic) from `klu_factor` (numeric +
pivoting) and add **`klu_refactor`** (numeric only, reusing the pivot pattern).
*Gate:* refactor far faster than factor, matching KLU's factor/refactor ratio.

### Phase 5 — Scaling, pivot robustness, final parity sweep
Row/column scaling and pivot strategy for indefinite circuit blocks, then a full
suite sweep proving the Definition of Done, locked behind a regression so it
cannot backslide.
*Gate:* native within 1.5× time / 1.2× fill / 1.2× memory across the suite,
rajat30 included.

## Relationship to existing issues

- Supersedes #131 (native KLU fill explosion — that was the symptom).
- Absorbs #118 (scaling) and #123 (pivot robustness) into Phase 5.
- Builds on #127 (O(flops)), #128 (near-linear ordering), #117 (per-block COLAMD).

## Highest leverage

Phases 1 and 2 (ordering + symmetric pruning) recover most of the gap for
one-shot factorization; Phase 4 (analyze/refactor split) is the highest leverage
for the actual SPICE workload and for the mp-spice mixed-precision study.

## References

- Davis & Palamadai Natarajan, "Algorithm 907: KLU", ACM TOMS 37(3), 2010.
- Gilbert & Peierls, "Sparse Partial Pivoting in Time Proportional to Arithmetic
  Operations", SIAM J. Sci. Stat. Comput., 1988.
- Eisenstat & Liu, "Exploiting Structural Symmetry in Unsymmetric Sparse
  Symbolic Factorization", SIAM J. Matrix Anal. Appl., 1992 (symmetric pruning).
- Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.
