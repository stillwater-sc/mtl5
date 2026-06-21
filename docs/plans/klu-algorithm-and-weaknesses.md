# KLU: Algorithm and Implementation Weaknesses

> **Note:** the durable, implementation-independent characterization of the KLU
> algorithm is published in the docs site under **Linear Algebra Algorithms**
> (`docs/algorithms/klu.md`). *This* document is the internal optimization
> tracker: it maps each weakness to a phase and records MTL5's measured
> native-vs-SuiteSparse numbers as the effort progresses.

A reference for the native KLU performance effort ([epic #138](native-klu-performance.md)).
It describes what KLU does, why it is fast, and the specific places a naive
implementation (ours included) loses performance. Each weakness is tagged with
the optimization phase that targets it; the **Quantified** column is filled in as
the benchmark scoreboard ([Phase 0, #132](native-klu-performance.md)) measures
each one.

## What KLU is

KLU (Davis & Palamadai Natarajan, *Algorithm 907*, ACM TOMS 2010) is a sparse
direct solver specialized for **circuit-simulation** matrices (Modified Nodal
Analysis). Its defining property: it is a **scalar, non-supernodal, left-looking
Gilbert–Peierls LU** with partial pivoting. It uses **no BLAS and no
supernodes** — deliberately, because circuit matrices are extremely sparse and
have almost no dense sub-structure to amortize a supernodal/BLAS kernel over.

This matters for us: a correct native implementation competes in the **same
algorithm class**. There is no kernel-class disadvantage to excuse a slowdown —
parity is an implementation problem, not an algorithmic one.

## The KLU pipeline

```
A  ──►  [1] BTF          ──►  [2] order each block  ──►  [3] factor each block  ──►  [4] solve
        (permute to                (AMD on A+Aᵀ of            (left-looking GP-LU         (block back-
         block triangular           the block)                + partial pivoting          substitution
         form)                                                 + symmetric pruning)        across blocks)
```

1. **BTF (Block Triangular Form)** — `maxtrans` (maximum transversal / matching
   for a zero-free diagonal) + `strongcomp` (strongly connected components).
   Only the diagonal blocks need factorization; off-diagonal coupling is handled
   in the solve. Near-linear.
2. **Ordering** — for each diagonal block, a fill-reducing ordering. KLU's
   **default is AMD on the symmetric structure A + Aᵀ** of the block (COLAMD and
   user orderings are options). The ordering choice dominates fill on
   unsymmetric blocks.
3. **Numeric factorization** — left-looking Gilbert–Peierls LU with **threshold
   partial pivoting** and **Eisenstat–Liu symmetric pruning**. Scalar; works on
   raw CSC integer arrays. Optionally preceded by **row scaling**.
4. **Three-way split** — `klu_analyze` (BTF + ordering + symbolic, done once),
   `klu_factor` (numeric + pivot search), and `klu_refactor` (numeric only,
   **reusing the pivot pattern**). Transient SPICE performs one analyze and
   thousands of refactors of the same pattern.

## Why KLU is fast (the techniques)

- **BTF** shrinks the problem to the diagonal blocks.
- **Good per-block ordering (AMD on A+Aᵀ)** keeps fill near-minimal for the
  unsymmetric circuit blocks.
- **Symmetric pruning** (Eisenstat–Liu) makes the per-column reach computation —
  the heart of left-looking GP-LU — touch a pruned structure instead of the full
  previously-computed columns of L. This is the dominant constant-factor win.
- **Tight CSC kernels** — integer-array gather/scatter with pre-sized,
  chunk-grown storage; no per-column heap traffic.
- **analyze/factor/refactor split** — the symbolic work and the pivot search are
  not repeated when only values change.

## Implementation weaknesses (and where a naive port loses)

The table below is the working scorecard. "Native today" describes the MTL5
implementation as of the start of the performance effort; "Quantified" is filled
from the Phase 0 scoreboard and updated as phases land.

| # | Weakness | KLU's technique | Native today | Phase | Quantified (native ÷ KLU) |
|---|----------|-----------------|--------------|-------|---------------------------|
| W1 | **Fill explosion on unsymmetric blocks** | AMD on A+Aᵀ per block | ~~COLAMD on AᵀA~~ → **AMD on A+Aᵀ (Phase 1)**; pivoting still inflates fill on the unscaled indefinite block | [1 (#133)](native-klu-performance.md) | **Measured fill (native ÷ KLU):** add32 **1.0×** (28902 = 28902), rajat14 1.1×, rajat30 **13.5×** (437M vs 32.3M). On benign blocks AMD fill *matches* KLU — explosion is specific to rajat30's indefinite block (→ scaling, Phase 5). rajat30 time 43.9× ≈ **13.5× fill × 3.3× constant** (→ pruning Phase 2 + kernels Phase 3). |
| W2 | **Expensive reach / DFS** | Eisenstat–Liu symmetric pruning | ~~full GP-LU DFS~~ → **symmetric pruning (Phase 2)** | [2 (#134)](native-klu-performance.md) | **Phase 2 (pruning):** Poisson 256² **3.6× → 1.5×** (0.68s → 0.29s; **meets the 1.5× DoD**), 128² 3.5× → 1.7×. rajat30 43.9× → **28.7×** (309s → 204s); its constant factor 3.3× → **2.1×** (residual 28.7× = 13.5× fill × 2.1× constant). Fill unchanged (= KLU); same residuals; ASan-clean. |
| W3 | **Heavy data structures in hot path** | raw CSC int arrays, pre-sized/chunk-grown | `compressed2D` + `inserter`, fresh `std::vector`s per block | [3 (#135)](native-klu-performance.md) | _TBD_ |
| W4 | **Repeated symbolic + pivot search** | analyze / factor / **refactor** split | pivots re-searched every solve; no refactor | [4 (#136)](native-klu-performance.md) | _TBD (refactor/factor ratio)_ |
| W5 | **No scaling / pivot strategy for indefinite blocks** | optional row scaling + threshold pivoting | ~~no scaling~~ → **row equilibration r=1/max\|row\| (Phase 5)** | [5 (#137)](native-klu-performance.md) | **Phase 5 (scaling):** rajat30 fill **13.5× → 1.7×** (437M → 55M) and time **28.7× → 4.5×** (204s → 32s) — equilibration lets pivoting follow the AMD order. Poisson unchanged (already well-scaled). Residuals tiny; correctness preserved. |
| W6 | **BTF constant factor** | tuned `maxtrans`/`strongcomp` | Kuhn matching + Tarjan SCC | [3 (#135)](native-klu-performance.md) | rajat30 BTF: 1.7s (of 7.4s KLU total) |

### Notes per weakness

- **W1 (fill).** The single biggest one-shot problem. Verified: on the SPD 2D
  Poisson block (where pivoting follows the order) native is only 3.6× slower
  (W2 territory), but on rajat30's indefinite 632k block the actual fill
  explodes far beyond that. **Phase 1 result:** switching to AMD on A+Aᵀ made
  rajat30 *complete* (309s, 437M fill) where it previously did not finish, and
  cut circuit fill 12–15%. But its fill is still ~10× KLU's and time 43× —
  because partial pivoting on the **unscaled, indefinite** block deviates from
  AMD's prediction. So AMD is *necessary but not sufficient*: closing the
  remaining fill needs **scaling (W5 / Phase 5)**, and the constant factor needs
  **symmetric pruning (W2 / Phase 2)**. rajat30 parity is genuinely multi-phase.
- **W2 (pruning).** Applies to *every* matrix, including symmetric ones; it is
  what makes the scalar left-looking solve competitive. Most of the benign-block
  3.6× is expected to live here.
- **W3 (data structures).** `compressed2D`/`inserter` do hashing/sorting and heap
  allocation unsuited to the inner loop; per-block `std::vector` churn adds up
  over the ~11.7k blocks of a matrix like rajat30.
- **W4 (refactor).** Not a one-shot-factor issue, but the dominant cost in the
  real SPICE workload (Newton iterations refactor the same pattern) and required
  by the mp-spice mixed-precision study.
- **W5 (scaling/pivoting).** Circuit matrices are badly scaled and indefinite;
  scaling reduces off-diagonal pivoting, which in turn reduces fill (couples back
  to W1).
- **W6 (BTF).** Smaller, but our BTF at 1.7s is a sizable fraction of KLU's
  entire 7.4s on rajat30, so it has room.

## How this doc is used

Phase 0 establishes the scoreboard and fills the **Quantified** column for W1, W2,
W6 (measurable today) and records baselines for W3–W5. Each subsequent phase
re-runs the scoreboard and updates its row, so this table tracks the closing of
the gap to the [Definition of Done](native-klu-performance.md) (within 1.5× time
/ 1.2× fill / 1.2× memory of single-threaded SuiteSparse KLU across the suite).

## References

- Davis & Palamadai Natarajan, "Algorithm 907: KLU", ACM TOMS 37(3), 2010.
- Gilbert & Peierls, "Sparse Partial Pivoting in Time Proportional to Arithmetic
  Operations", SIAM J. Sci. Stat. Comput., 1988.
- Eisenstat & Liu, "Exploiting Structural Symmetry in Unsymmetric Sparse Symbolic
  Factorization", SIAM J. Matrix Anal. Appl., 1992.
- Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006 (Ch. 6–7).
