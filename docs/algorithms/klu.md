# KLU: Sparse LU for Circuit-Simulation Matrices

KLU (Davis & Palamadai Natarajan, *Algorithm 907*, ACM TOMS 2010) is a sparse
direct solver specialized for the matrices that arise in **circuit simulation**
(Modified Nodal Analysis). It is the reference implementation against which any
native sparse-LU solver for these problems should be measured.

This page characterizes the algorithm and the engineering that makes it fast,
and catalogs the performance weaknesses a naive implementation exhibits. It is
implementation-independent; MTL5 ships its own native KLU
(`mtl::sparse::factorization::native_klu`) and an external SuiteSparse binding
(`mtl::interface::klu_solver`).

> Accuracy of a solve is judged by the **residual** `‖A x̂ − b‖`; see
> [Measuring Solver Accuracy](measuring-solver-accuracy.md) for residuals, norms,
> and the residual–vs–forward-error distinction used throughout these pages.

## What kind of solver it is

KLU is a **scalar, non-supernodal, left-looking Gilbert–Peierls LU** with
threshold partial pivoting. The "scalar, non-supernodal" part is a deliberate
design choice: circuit matrices are extremely sparse and have almost no dense
sub-structure, so the supernodal/BLAS-3 approach that powers solvers like
CHOLMOD and UMFPACK has nothing to amortize over. KLU therefore uses tight
scalar integer kernels instead — and is faster than supernodal solvers *on this
class of matrix* precisely because it does not pay for machinery it cannot use.

A practical consequence: a from-scratch solver competes with KLU **in the same
algorithm class**. There is no kernel-class advantage to attribute a slowdown
to; matching KLU is an engineering problem, not an algorithmic one.

## The pipeline

```text
A ─► [1] BTF ──────────► [2] order each ──► [3] factor each ───────► [4] solve
        permute to            diagonal block      diagonal block          block
        block triangular      (AMD on A+Aᵀ)       (left-looking GP-LU      back-
        form                                       + partial pivoting       substitution
                                                   + symmetric pruning)
```

1. **Block Triangular Form (BTF).** A maximum transversal (bipartite matching
   for a zero-free diagonal) followed by a strongly-connected-components
   decomposition permutes `A` so that only the diagonal blocks require
   factorization; the off-diagonal coupling is resolved during the solve. Many
   circuit matrices are highly reducible, so BTF can shrink the work
   dramatically. (Some are not — see "When BTF does not help" below.)
2. **Ordering.** Each diagonal block gets a fill-reducing ordering. KLU's default
   is **AMD on the symmetric structure `A + Aᵀ`** of the block. The ordering
   choice dominates fill, and for unsymmetric blocks the *symmetrized* ordering
   is markedly better than ordering `AᵀA` (COLAMD-style), because partial
   pivoting otherwise deviates from the column order and fill explodes.
3. **Numeric factorization.** Left-looking Gilbert–Peierls: each column `k` of
   `L`/`U` is computed by a sparse triangular solve against the
   already-computed columns, whose nonzero pattern (the *reach*) is found by a
   depth-first traversal of the elimination structure. Threshold partial
   pivoting provides stability; **Eisenstat–Liu symmetric pruning** keeps the
   reach traversal cheap. Optional row **scaling** improves robustness on the
   badly-scaled, indefinite blocks circuit matrices produce.
4. **Solve.** Block back-substitution across the BTF structure, with a
   triangular solve per diagonal block using its stored factors.

## Why it is fast

| Technique | What it buys |
|-----------|--------------|
| BTF | factor only the diagonal blocks |
| AMD on A+Aᵀ per block | near-minimal fill on unsymmetric blocks |
| Eisenstat–Liu **symmetric pruning** | the per-column reach DFS touches a *pruned* structure, not the full computed columns of L — the dominant constant-factor win in scalar left-looking LU |
| Tight CSC integer kernels | gather/scatter on raw arrays, pre-sized and chunk-grown — no per-column heap traffic |
| **analyze / factor / refactor** split | symbolic analysis and pivot search are done once; transient simulation refactors the same pattern thousands of times via `klu_refactor` (numeric only) |

That last point is central to the circuit-simulation workload: a transient
analysis performs one `klu_analyze` and then a `klu_factor`/`klu_refactor` per
Newton step, reusing the symbolic structure and (for refactor) the pivot
sequence. Amortizing the symbolic work is much of KLU's real-world speed.

## Complexity

- **Time:** O(flops), i.e. proportional to the arithmetic operations of the
  factorization — *not* a function of `n` independent of sparsity. The
  Gilbert–Peierls reach formulation is what guarantees this; an implementation
  that scans all previous columns per step is accidentally O(n²) regardless of
  sparsity.
- **Fill / memory:** determined by the ordering. For 2D-grid-like structure,
  optimal orderings give O(n log n) fill and O(n^1.5) flops — so even a perfect
  implementation is super-linear on such matrices; the goal is to match the
  *constant*, not to be linear.

## Implementation pitfalls (where naive versions lose)

These are the recurring ways a from-scratch KLU underperforms the reference.
Each is a concrete, measurable target.

1. **Accidental O(n²) factorization.** Computing each column's structure by
   scanning all previously-factored columns (instead of via the Gilbert–Peierls
   *reach*) is quadratic in `n` even for a tridiagonal matrix. Symptom: time
   quadruples when `n` doubles, independent of nonzeros.
2. **Wrong block ordering → fill explosion.** Ordering unsymmetric blocks by
   `AᵀA` (COLAMD) rather than `A+Aᵀ` (AMD) lets partial pivoting diverge from the
   predicted order; actual fill — and thus flops and memory — blows up on
   indefinite circuit blocks while staying fine on symmetric/SPD ones.
3. **No symmetric pruning.** Without Eisenstat–Liu pruning, the per-column reach
   DFS re-traverses full L columns; the factorization is still O(flops) but with
   a large constant. This is usually the biggest single constant-factor gap.
4. **Heavyweight data structures in the hot path.** Building and slicing the
   matrix through general containers (hashed/sorted inserters, per-block
   reallocation) instead of raw CSC integer arrays adds allocation and indirection
   to the innermost loops.
5. **No analyze/factor/refactor split.** Re-running symbolic analysis and pivot
   search on every solve is wasteful for the dominant workload (repeated
   factorization of the same pattern with changing values).
6. **No scaling / weak pivot strategy.** Badly-scaled, indefinite circuit blocks
   provoke excessive off-diagonal pivoting (which feeds pitfall 2). Row/column
   scaling and a tuned threshold reduce both instability and fill.
7. **Quadratic BTF.** The matching and SCC passes must be near-linear; a per-row
   full-array reset in the matching, or recursive SCC that overflows the stack,
   reintroduces quadratic time or crashes on large graphs.

### When BTF does not help

BTF only reduces work when the matrix is reducible. Some circuit matrices are
essentially irreducible — e.g. `Rajat/rajat30` (≈644k unknowns) decomposes into
one block of ≈632k plus ~11.7k singletons. There, performance rests entirely on
the per-block ordering (pitfall 2) and factorization quality (pitfalls 1, 3),
not on BTF.

## Characterizing an implementation against KLU

When a from-scratch solver is slower than KLU, the first question is *which*
pitfall dominates. The durable technique is to **measure fill and time as two
separate axes** against the reference, because they have different cures:

- **fill** = nnz(L) + nnz(U) of the factors (the structural cost),
- **time** = factor + solve wall-clock (the structural cost × per-flop cost).

They compose multiplicatively:

```text
time_ratio  ≈  fill_ratio  ×  constant_factor
            (ordering/scaling)   (kernels: pruning, data structures)
```

So the two ratios localize the bottleneck:

- **fill_ratio ≈ 1 but time_ratio > 1** → the ordering is already good; the gap
  is **constant factor** — missing symmetric pruning and/or heavyweight data
  structures (pitfalls 3, and the reach cost of no pruning).
- **fill_ratio ≫ 1** → an **ordering/scaling** problem (pitfalls 2, 6): the
  factorization is doing genuinely more arithmetic. Fix fill *before* chasing
  the constant factor, since it multiplies everything.

### Worked example

Measuring a native left-looking implementation (AMD-per-block, no symmetric
pruning yet) against SuiteSparse KLU:

| matrix | fill_ratio | time_ratio | reading |
|--------|-----------:|-----------:|---------|
| `add32` (well-scaled circuit block) | **1.0×** (identical) | 3.5× | ordering already optimal → the 3.5× is **pure constant factor** (pruning + kernels) |
| `rajat30` (large indefinite block) | **13.5×** | 43.9× | 43.9 ≈ **13.5 (fill) × 3.3 (constant)** → *both* a fill problem (scaling/pivoting) and the same constant factor |

The conclusion is actionable and ordered: because `add32`'s fill already equals
KLU's, the ordering is correct and the universal lever is the **constant factor**
(symmetric pruning, then CSC kernels). `rajat30` additionally needs **scaling**
to stop partial pivoting from inflating fill on the unscaled indefinite block —
and that fill fix is worth ~13×, so it precedes constant-factor work for that
matrix class. Measuring the two axes separately is what turns "it's 44× slower"
into a precise, prioritized work list.

## References

- Davis, T. A. & Palamadai Natarajan, E. "Algorithm 907: KLU, A Direct Sparse
  Solver for Circuit Simulation Problems." *ACM TOMS* 37(3), 2010.
- Gilbert, J. R. & Peierls, T. "Sparse Partial Pivoting in Time Proportional to
  Arithmetic Operations." *SIAM J. Sci. Stat. Comput.* 9(5), 1988.
- Eisenstat, S. C. & Liu, J. W. H. "Exploiting Structural Symmetry in Unsymmetric
  Sparse Symbolic Factorization." *SIAM J. Matrix Anal. Appl.* 13(1), 1992.
- Davis, T. A. *Direct Methods for Sparse Linear Systems.* SIAM, 2006.
