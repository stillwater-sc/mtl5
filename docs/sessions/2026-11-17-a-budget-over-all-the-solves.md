# Session: a budget over all the solves

**Date**: 2026-11-16 into 2026-11-17
**Duration**: One long working session, with context compactions along the way
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Finish the QA-sweep follow-ups that were in flight, make the Krylov solvers
judge convergence the same way, and triage and clear two more sweeps:

- the sweep #704/#718 tracking PRs (#868–#871);
- CGS's convergence test and a world-class IDR(s) (#873, #874);
- sweep #783 (#876–#878);
- sweep #872 (#886–#888).

**Merged:** #868, #869, #870, #871, #875, #879, #880, #881, #882, #883, #889,
#890, #891.

**Closed:** #704, #718, #721, #737, #778, #783, #864–#867, #873, #874,
#876–#878.

**Filed:** #873, #874, #876–#878, #884 (the column sweep on small and short,
wide matrices), #885 (the #778 follow-up), #886–#888.

This log covers the work from #868 on. The earlier PRs of 2026-11-14 to
2026-11-16 have their own CHANGELOG entries.

## The through-line

Many of the defects were cases where a result was measured against the wrong
reference, and nothing said so.

- **The Krylov solvers (#875)** stopped on the recursively updated residual.
  In CGS, BiCG and QMR it drifts from b - A x, so a solver could report
  convergence on a residual it didn't have. They now confirm on the true
  residual and restart from it, as `tfqmr` already did.
- **An iteration object reused for a second solve (#891)** kept counting
  toward `max_iter`, and kept measuring the relative tolerance against the
  first r0. #866 had made reuse work at all, and pinned that behaviour. The
  decision (option b) was an explicit `reset(r0)`: the solver can't reset the
  object itself, because only the caller knows what r0 is.
- **`eigen()`** takes its eigenvalues from a second `eigenvalue()` run, and
  pairs them by nearest distance with the Schur form's, which has the vectors.
  Dropping that pass changes the order the eigenvalues come back in, and
  `eigen()` documents and tests `eigenvalue()`'s order. It is left for a
  decision and is not in #891.

## Work completed

### Sweep #704/#718: #868–#871

Merged as prepared. The decisions taken before the PRs were these:

- `sparse_qr` keeps the exact-zero rank test, documented (#864).
- A missing or zero diagonal throws `std::runtime_error` when a
  preconditioner or smoother is constructed, in every build, and names the
  row (#867).

#704 and #718 were closed afterwards.

### True-residual Krylov convergence and IDR(s): #875, #879

- **#875:** `cg`, `bicg`, `cgs`, `bicgstab`, `bicgstab_ell` and `qmr` confirm
  convergence on b - A x and restart from it when the updated residual has
  drifted.
- **#879:** IDR(s) is rewritten as the biorthogonal algorithm (van Gijzen and
  Sonneveld, ACM TOMS Algorithm 913). It has residual replacement, redraws P
  on breakdown, and takes `idrs_options`.

### The version rule: #880

The project had been spending MINOR numbers on releases that contained only
fixes. The rule is now in CLAUDE.md:

- a `feat` or a breaking change bumps MINOR;
- a release with neither bumps PATCH;
- MAJOR stays at 5, because a MAJOR version is a management decision.

### Sweep #783: #881–#883

- **#881:** the thread pool.
- **#882:** the recursion infrastructure. The shark-tooth masks were removed,
  by decision.
- **#883:** the SIMD blocking overrides.

The column-sweep slowdown on small and on short, wide matrices is not a
defect, so it was filed as #884.

### Sweep #872: #889–#891

The sweep was triaged into three tracking issues:

- **#887 → #889:** `min` and `max` on inf, NaN and empty input,
  `fused_assign` aliasing, unsigned `signum`, `condition_number` without
  infinity, and a saturating unscaled `project_onto`.
- **#886 → #890:** matrix norms that follow the layout, and O(nnz) norms and
  predicates on `compressed2D`. It also makes sparse `mult` work with any B,
  handles aliasing in `symm`, `gemmt`, `syrk`, `syr2k` and `trmm`, and guards
  BLAS `int` dimensions. Two CI lanes found real bugs:
  - LAPACK: `condition_number` came out finite for a rank-1 matrix;
  - BLAS: `syrk(1, C, 0, C)` was wrong under `?syrk`.
- **#888 → #891:**
  - `reset(r0)` on the iteration objects;
  - `range::extent()` with a huge step;
  - an exact `shares_memory_with`, which now searches signed coefficients.
    The old test missed views 4 elements apart with strides {5, 1}, and
    `B = A` skipped its temporary;
  - the square seeded `randsvd`;
  - `ldlt_bk`'s scratch, now allocated once;
  - the multigrid thread-safety note;
  - the requested tests.

  CodeRabbit's review of #891 found a hole in `randsvd`. With an integral
  kappa, a four-argument call reads equally well as the rectangular
  `(m, n, kappa, mode)` and the square seeded `(n, kappa, mode, seed)`. The
  forms are now told apart by type, and an integral kappa doesn't compile.

## Open

- **Decision:** may `eigen()` drop its second pass and return the Schur run's
  eigenvalue order? If so, the test at `test_eigen_general.cpp:230` and the
  documented contract change with it.
- #886, #887, #888 and sweep #872 are still open. Their PRs have merged.
- #884 and #885 are filed, with no work started.
