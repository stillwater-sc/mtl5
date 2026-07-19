# Session: Eigenvalue/eigenvector solver epic (#202) — end to end

**Date**: 2026-07-18
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Take the eigenvalue/eigenvector capability from "dense values only" to a complete
suite spanning dense, iterative, and sparse — with LAPACK acceleration and a user
guide — by planning epic #202 and then delivering every child issue in one
session.

## Context

At the start, MTL5's eigen suite (`include/mtl/operation/`) was **dense-only**
and had notable gaps: the general routine `eigenvalue(A)` returned eigenvalues
but **no eigenvectors**; there was no sparse or iterative eigensolver; and LAPACK
acceleration existed only for the symmetric `syev` path. The session began by
auditing the suite, filing a tracking epic (#202) with five child issues, and
then implementing them in dependency order — with one bug discovered mid-stream
promoted to its own issue (#209).

## Work Completed

### General eigenvectors + `eigen` API (#203 → PR #208, merged)
`mtl::eigen(A)` returns eigenvalues **and right eigenvectors** as a
structured-bindable `{ eigenvalues, eigenvectors }` (complex), mirroring
`eigen_symmetric`. In the in-house path, eigenvalues come from the general QR
path and each eigenvector is recovered by **inverse iteration** on
`A - lambda_k*I` (partial-pivot complex LU with a pivot floor — the near-singular
regime inverse iteration exploits). When LAPACK is available and the type
qualifies, `eigen` instead dispatches to `geev` (added in #204), which returns the
eigenvectors directly.

- **CodeRabbit finding (Major), fixed and verified:** the first cut used an
  identical seed and factorization for every equal eigenvalue, so `eigen(I)`
  returned duplicate columns instead of a basis (the `A v = lambda v` residual
  was still 0, which is why the tests missed it). Fixed with **deflated inverse
  iteration** — a `k`-dependent seed plus modified Gram-Schmidt against prior
  eigenvectors sharing the eigenvalue, matched within a tiny cluster tolerance so
  complex-conjugate partners are never merged.

### Francis double-shift QR fix (#209 → PR #210, merged)
While validating #203, `eigenvalue()` was found to **silently return wrong
eigenvalues** on strongly non-normal matrices: its single real shift never
triggered deflation for complex eigenvalues, so it fell through to reading the
diagonal. Forsythe(5) returned `{3,3,3,3,3}` vs the true radius-0.87 circle — and
the old test only checked the trace, which the wrong answer satisfies.

- Replaced the core iteration with the **Francis implicit double-shift QR**
  (EISPACK `hqr`): real Schur form via 1×1/2×2 deflation, exceptional shifts, and
  a `std::runtime_error` on non-convergence instead of a wrong diagonal read.
- Bonus: this lifted the #203 eigenvector accuracy ceiling — Forsythe eigenvector
  residuals dropped from ~1 to ~1e-13.

### LAPACK `geev` dispatch (#204 → PR #211; CI job PR #212, merged)
`eigenvalue`/`eigen` dispatch to `geev` when `MTL5_HAS_LAPACK` is defined and the
matrix is a column-major `dense2D<float/double>` (mirroring `syev`); everything
else keeps the in-house path. Verified the path is genuinely taken (`dgeev_`
linked; predicates route `colmat → geev`, `rowmat → in-house`).

- **CI gap closed (#212):** the default CI matrix builds without LAPACK, so the
  `geev`/`syev` branches were only ever compiled locally. Added a `lapack` CI job
  (Linux GCC + Clang, `-DMTL5_WITH_LAPACK=ON`) that builds and runs them — its own
  run proved both jobs green.

### Matrix-free iterative eigensolvers (#205 → PR #213, merged)
`power_iteration`, `lanczos` (symmetric), and `arnoldi` (general) under
`mtl::itl`, all operating through the `LinearOperator` concept (`A * x`). Each
projects onto a Krylov subspace and **reuses the dense eigensolvers** on the small
projected problem (tridiagonal for Lanczos, Hessenberg for Arnoldi). Full
reorthogonalization; an `eigen_which` selector for the wanted end of the spectrum.

### Sparse eigensolver with shift-invert (#206 → PR #214, merged)
The capstone. `sparse_eigs` runs Arnoldi directly on a sparse operator (largest
magnitude, no factorization). `sparse_eigs_shift_invert` finds the k eigenvalues
nearest `sigma` by factoring `(A - sigma*I)` once with the **sparse LU direct
solver** and applying its inverse inside Arnoldi — `lambda = sigma + 1/theta`,
same eigenvector, "nearest sigma" becoming the fast-converging "largest |theta|".
The reusable `shift_invert_operator` perturbs tiny pivots so a shift on an
eigenvalue stays solvable.

### Solver guide (#207 → PR #215, merged)
`docs/algorithms/eigenvalues.md` grew incrementally across every PR and was
finished here: a "choosing a solver" decision guide, a runnable snippet for every
public eigen API (dense/iterative/sparse), the LAPACK dispatch conditions, a
pointer to the `examples/phase06_eigenvalue_svd/` worked examples, and the
custom-number-type story.

## Decisions & Rationale

- **Inverse iteration over Schur+`trevc` for #203.** Reuses the validated
  eigenvalue path instead of rewriting the Francis iteration to accumulate Schur
  vectors — far less error-prone for a reference implementation, and the acceptance
  metric (`A v = lambda v` residual) is met directly.
- **Projected problems reuse the dense solvers.** Lanczos/Arnoldi/shift-invert all
  solve their small tridiagonal/Hessenberg with `eigen_symmetric`/`eigen`, so the
  hard numeric work lives in one tested place and the layers compose.
- **Column-major-only LAPACK gate**, matching the existing `syev` convention, with
  the working copy built through the `(i,j)` accessor so it is correct regardless
  of source orientation.
- **Honest test calibration.** Several tests initially over-asserted the
  conservative `converged` flag or picked a `sigma` exactly on the midpoint between
  two eigenvalues (an ambiguous "nearest"); these were fixed to reflect real solver
  behavior rather than masked.

## Epic #202 — final status (closed)

| # | Delivered | PR |
|---|---|---|
| #203 | general eigenvectors + `mtl::eigen` API | #208 |
| #209 | Francis double-shift QR fix for `eigenvalue()` | #210 |
| #204 | conditional LAPACK `geev` dispatch | #211 |
| — | LAPACK CI job | #212 |
| #205 | iterative eigensolvers (power / Lanczos / Arnoldi) | #213 |
| #206 | sparse eigensolver with shift-invert | #214 |
| #207 | eigenvalue/eigenvector solver guide | #215 |

The layers build on each other: **dense** solvers (#203/#204/#209) solve the
projected problems inside the **iterative** Krylov solvers (#205), which are driven
by the sparse LU direct solver in the **sparse** shift-invert path (#206).

## Outcome

MTL5 now has a complete eigenvalue/eigenvector suite — dense (with LAPACK
acceleration and a correct double-shift QR), matrix-free iterative, and sparse
shift-invert — all documented and on `main`. Every PR passed the 8-platform CI
matrix and was verified locally on GCC and Clang.

Documented future work: implicit/thick restarting for the iterative solvers,
generalized problems `A x = lambda B x` (shift-invert with `B`), and finer-grained
LAPACK bindings (`hseqr`/`gehrd`/`trevc`).

## Issues / PRs

- Merged this session: #208 (#203), #210 (#209), #211 (#204), #212 (CI), #213 (#205), #214 (#206), #215 (#207)
- Epic: #202 (closed — fully delivered)
- Bug discovered and fixed mid-epic: #209 (double-shift QR stall)
