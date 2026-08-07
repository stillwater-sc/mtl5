# Session: a passing test proves nothing until you have watched it fail

**Date**: 2026-08-07 (spanning 2026-08-06)
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Resolve #362 — the Hermitian eigenproblem the complex-factorization work (#353,
#360, #361) had deliberately stopped short of. One question, asked once the
native path was in, pulled the rest of the day behind it: *do we have a LAPACK
vs native validation for it?* We did not. Getting one exposed that the LAPACK
dispatch was dead for the default matrix type, and closing that gap was the
second half of the day.

Four PRs merged: #414 (native complex Hermitian eigenproblem), #416 (zheev
cross-validation), #418 and #419 (row-major LAPACK dispatch, Categories A and B
of #417).

## The through-line

Last session's habit was *run the control*. This session's was the same habit
aimed at one specific lie: **a green test that exercises nothing**. Every step
had a version of it, and the only thing that told them apart was a check that
does not care whether the answer is correct — `nm`, asking whether the symbol is
even referenced.

| Looked covered | Actually |
|---|---|
| #361's compile-failure guard pinned the complex eigenproblem | It sat on `eigenvalue_symmetric_generic`, which never calls `householder` — matched its regex, covered nothing; the reduction site was open |
| The phase-accumulation fold is what the complex tests exercise | For `n ≥ 3` MTL5's real-β `householder` makes it a no-op; only the `2×2` (reduction-skipped) tests reach a genuinely complex subdiagonal |
| My first zheev cross-check compared native vs LAPACK | Row-major → dispatch never fired → generic vs generic. `nm`: no `zheev_` |
| The real `syev`/`geev` "dispatch" tests exercised LAPACK | Also row-major → also silently generic, and had been since they were written |
| Category B was green after the affected suites passed | The full sweep caught two more: a bit-exact `LL^ᵀ==LL^ᴴ` invariant and the `#297` determinism test |

The tell each time was that the *numeric* result was correct on both paths, so no
residual, no reconstruction, no tolerance could distinguish them. The control was
`nm -u`: compile a translation unit that uses **only** the path in question and
ask whether the LAPACK symbol appears. Before the fix: none. After: the symbol.
That is the whole proof; nothing about the answer's value is involved.

## Work Completed

### #414 — the reduction, and the guard that guarded nothing

The unitary reduction is `A → H·A·Hᴴ`, which is just `conj(τ)` on the right
factor — the adjoint bookkeeping #361 had already worked out for QR/LQ. The real
path stays bit-identical because `conj` is the identity and the QR-iteration
scalars are `magnitude_t<value_type>`, which *is* `value_type` for a real type.

Two things the issue itself flagged and both held:

- **The guard.** #361 put the complex-reject `static_assert` on
  `eigenvalue_symmetric_generic`, which does not call `householder`. The
  compile-failure test named that function, matched, and went green while the
  function that actually performs the reduction — `eigen_symmetric` — was never
  covered. Enumerating entry points against the reduction *site* first is the
  whole lesson.
- **The phase.** The issue, reasoning from LAPACK `zhetrd`, expected an arbitrary
  phase on every subdiagonal needing a diagonal `D` folded into the eigenvectors.
  MTL5's `householder` reflects to a **real β**, so the reduction produces real
  subdiagonals for free and `D` degenerates to ±1 — *except* for `n < 3`, where
  the reduction is skipped and `T(1,0)` is the original complex off-diagonal.
  The fold is real and load-bearing there and nowhere else, which is exactly what
  the `2×2` tests reach. Kept the fold (robust to a future reflector convention)
  and let the `2×2` cases be its negative control.

### #416 — the reference oracle, and the vacuous cross-check

`cheev`/`zheev` bindings, a `BlasHermitianMatrix` concept, and a `heev` dispatch.
The first cut of the cross-validation test used a default `dense2D` — row-major —
and "passed". It was comparing the generic path to itself. `nm` on the built
object: no `zheev_`. Switching the test to `mat::parameters<tag::col_major>` made
the dispatch fire (`U zheev_`, `ldd` → `liblapack`), and native agreed with zheev
to `1e-9`. The self-consistent #414 invariants catch a phase bug; only zheev
catches an algorithm that is self-consistently wrong.

### #417 — the dispatch that was dead for everyone

Switching the test to column-major to make it fire is what surfaced the real
finding: **`dense2D` is row-major by default, and every LAPACK dispatch guarded on
`!is_row_major_v<M>`**, so the default type never reached LAPACK — and the
pre-existing real `syev`/`geev` dispatch tests had been silently generic all
along. Two categories, filed as #417 and split across two PRs:

- **A (#418), redundant guard.** `syev`/`heev`/`geev`/`gesdd` already copy through
  the `(i,j)` accessor, so the guard only disabled them for the common case.
  Dropped it. `nm`: a row-major-only eigen/SVD TU now references all four LAPACK
  entry points and none before.
- **B (#419), load-bearing guard.** `lu`/`qr`/`cholesky` hand the raw `A.data()`
  buffer to Fortran, where row-major is the transpose of what LAPACK reads. Row-
  major now factors through a column-major scratch and copies the factor back via
  `(i,j)`, so the accessor-based solves are unchanged.

Category B is where the full-suite sweep earned its keep. Two failures the
affected-target runs had not shown:

1. **`cholesky_h_factor` vs `cholesky_factor`, bit-exact.** `test_cholesky.cpp`
   pins `L·Lᵀ == L·Lᴴ` for a real type with `==`, deliberately, "do not relax to
   a tolerance." Making `cholesky_factor` use `potrf` while `cholesky_h_factor`
   stayed generic diverged them in the low bits. Fixed by having
   `cholesky_h_factor` **delegate** to `cholesky_factor` for real BLAS types —
   same code path, invariant preserved by construction, speedup extended.
2. **`#297` threaded determinism.** It asserts the threaded factorization is bit-
   identical to a serial *generic* reference — a property of `parallel_for`, not
   of LAPACK. The default `double` path now dispatching to LAPACK broke it. On
   Theo's call, the LU/Cholesky determinism cases moved to a **non-BLAS element
   type** (`long double`), so `dense2D<det_t>` keeps taking the generic parallel
   kernel in every build; the `apply_householder_*` cases stayed `double`, since
   those kernels never dispatch. A `0×0` guard (`lda == 0` is a LAPACK error) fell
   out of the same test's degenerate-sizes case, surfaced through the delegation.

## The part that went wrong: the CI outage

Between #414's merge and the follow-ups, GitHub Actions had a multi-hour
`major_outage` that cancelled the CI. A watcher was armed correctly and reported,
each hour, `actions=operational ci=settled-not-green — re-run needed` once the
outage lifted at 00:07Z. **It was not acted on for ~10 hours.** The signal was
right; acting on it was the missing step. The watcher was also hardened mid-run:
its first counting scheme dropped blank (freshly-queued) rollup entries, which
could read as a false green — fixed to take the total from the array length and
treat anything not explicitly passed as pending. Recorded here because the
failure was mine, not the tool's, and the tool's near-miss was the same
false-green shape as the rest of the day.

## Issues and PRs

- **Closed**: #362 (via #414), #415 (via #416), #417 (via #419)
- **Merged**: #414, #416, #418, #419 — all green on the full matrix, LAPACK CI
  jobs included; #419 also cleared Tier-2 regression (dense solvers consume these
  factorizations)

## Lessons

- **A dispatch test cannot be written in the type it dispatches on.** If both the
  fast and slow paths return the right answer, the numeric assertion proves the
  answer, not the path. Prove the path with `nm`/`ldd`, or with a type that forces
  it (`col_major` to reach LAPACK, a non-BLAS type to reach the generic kernel).
- **Guard the reduction site, not a neighbour.** #361's mis-placed guard and this
  session's dead dispatch are the same error: a check attached to code that does
  not run the thing being checked.
- **The affected-target run is not the full sweep.** Both Category-B regressions
  passed the lu/qr/cholesky suites and only fell out of the whole-suite ctest —
  one of them (`#297`) in a file whose name does not mention factorization dispatch.
- **A correct signal is not a completed action.** The watcher said "re-run
  needed" for ten hours. Reading it was not the same as doing it.
