# Session: the lesson was already written down

**Date**: 2026-10-15
**Duration**: One working session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Resolve #535: extend the accumulator policy from the numeric phase of the sparse
direct factorizations to their `solve()` paths.

The issue was written before #542 changed the convention. Its 2026-10-10 status comment
recorded three updates:
- the parameter is now a leading `Accumulator = void`;
- the reductions for Cholesky and LDLᵀ live in `level_schedule.hpp`, not in `solve()`;
- threading that one header would cover the solve phase of every sparse direct solver,
  not just the three the issue named.

**Merged:** mtl5 #576.

**Closed:** #535.

## The through-line

The [2026-10-11 log](2026-10-11-true-under-the-flags-it-was-checked-with.md) ends with
this lesson:

> **State a numerical-identity claim together with its flags.** [...] a new test making
> the same claim needed the same treatment.

Four days later a new test made the same claim and didn't get the same treatment. The
FP-contract lane caught it, as it was built to. The lesson was recorded in a session
log, and session logs aren't read before writing a test. The repo's own record of it,
the comment block in `tests/unit/CMakeLists.txt` above the `-ffp-contract=off` list,
sits next to where tests are registered, not where they are written. A bit-identity
test is registered automatically by the `sparse/*.cpp` glob, so nothing leads the
author to that comment.

## Work completed

### #576: the accumulator policy on every sparse direct solve

**Scope.** A survey of the solve paths came first. It showed:
- every `*_numeric::solve()` (Cholesky, LDLᵀ, LU, supernodal LU and LDLᵀ) hands its
  reductions to five level-scheduled gather kernels;
- `qr_numeric::solve` has its reductions inline;
- the scatter kernels in `triangular_solve.hpp` aren't called by any solve, only by
  tests as bit-identity references.

The user chose "solve paths only": the five kernels, the five `solve()` methods, the
QR solve and the one-shot wrappers. `triangular_solve.hpp`, `native_klu` and the
`*_solve_refined` wrappers were left out.

**Kernels.** Each level-scheduled solve takes a leading `Accumulator = void`. `void`
keeps the original loop behind `if constexpr`, so the default is byte-identical rather
than identical by argument. The accumulator path folds b[i] into the accumulator:
`assign(acc, b[i])`, then `add_product(acc, -T(i,k), x[k])`, then round once and
divide. `upper_trisolve`'s accumulator path rounds the sum and subtracts it from b
afterwards. Folding b in is what makes an exact accumulator worth having here: the
quantity that cancels is b − Σ, not Σ. Negating the multiplier is exact, so
`Acc == Value` reproduces `acc -= a*b` bit for bit.

**QR.** The vₖᵀw dot was a straightforward port. The R back-substitution is a CSC
scatter, so a gather would have needed a transposed schedule. It keeps the scatter
instead, with one accumulator per row of R. y[col] is rounded out of `ws[col]` when
its column is reached, so the terms arrive in the default's order. An empty column
must still round its accumulator out, which the code does before the empty-column
`continue`.

**Wrappers.** The one-shot `*_solve` functions now take the leading `Accumulator` too
and pass it to both phases. Without that, `sparse_cholesky_solve` could only ever
produce a policy-free result, however the caller configured it.

**Tests.** A float row 1 − 10⁸ + 10⁸ is the fixture. 10⁸ is exact in `float`, but
its spacing is 8, so the raw path gives 0 and a `double` accumulator gives 1. It
cancels in all five kernels: the transpose and upper layouts each needed their own
3 × 3, because the order the terms are gathered in differs. The products are exact,
so FMA contraction can't change the raw answer, which keeps the fixture valid in
every lane. QR uses a hand-built `qr_numeric`, whose members are public. Each of its
two sums can then be driven alone: R = I with one reflector for the dot, and β = 0
with a cancelling R for the back-substitution.

### The FP-contract failure

The first push failed one assertion, in one lane: `d == r`, the level-scheduled solve
against `dense_lower_solve`, under `-march=x86-64-v3` with GCC's default
`-ffp-contract=fast`. Both loops in that comparison are code this PR didn't change.
GCC contracts the gather's `acc -= a*b` and the scatter's `x[i] -= a*b` into FMAs
differently, and the last bit moves.

The failure reproduced locally with the lane's flags, 1 of 1,428 assertions. The fix
puts the target on the existing pin list with a paragraph saying why, as #381, #549,
#550, #556 and #559 did. MSVC was the only job still pending, and pushing would have
cancelled it. MSVC is excluded from the pin, so its result was the one that mattered.
The fix was pushed after it passed.

## The parts that went wrong

- **The FP-contract pin, again.** See the through-line. The pin list held six
  targets before this one. Each asserts bit-identity between differently shaped
  loops, and this is the seventh. At least #552's and this one were pinned only
  after the lane failed.
- **Not checked locally under contraction before pushing.** gcc Debug and clang Debug
  both passed. Neither contracts: -O0 doesn't run the optimizer pass. The lane that
  does was a single `cmake -B` away, and that command is in `ci.yml`.

## Tooling notes

- **`gh run view --log-failed` returns nothing while the run is in progress**, even
  for a job that has finished. `gh api repos/<o>/<r>/actions/jobs/<id>/logs` returns
  that job's log immediately.
- **Reproducing a lane locally** takes the lane's configure line from `ci.yml`, a
  scratch build directory and one target. Here that meant one target built in about a minute,
  against 7 minutes for the CI lane.
- **Explicit template arguments on a generic lambda**: `run.template operator()<A>(x)`
  is how a test passes `void` / `float` / `double` into one checking lambda without a
  helper struct.

## Lessons

- **A lesson about where tests go belongs where tests are written.** A seventh entry on
  the same list says the `-ffp-contract=off` comment block isn't being read in time. Candidates:
  - a naming convention (`*_bitident.cpp`) that CMake pins automatically;
  - a note in CLAUDE.md's "Adding a New File" section;
  - stating any `==` against a reference loop in the test file's header comment
    together with the flag it needs.

  The first removes the step entirely.
- **"Bit-identical" in a header comment is a claim the test inherits.**
  `level_schedule.hpp` has long said "bit-identical to dense_lower_solve", but that
  holds only with contraction off. The existing `_mt` test was already pinned, so the
  comment's condition was in CMake, not in the comment.
- **Fold the cancelling term into the accumulator.** For a solve, the quantity at
  risk is b − Σ. An accumulator that holds only Σ and rounds before the subtraction
  gives up the case an exact accumulator exists for.

## Follow-ups

- Decide on one of the three mechanisms above so the eighth bit-identity test is
  pinned when it is written.
- `triangular_solve.hpp`'s scatter kernels and `klu_numeric::solve`'s block coupling
  are the remaining sparse solve sums outside the policy, if a caller ever needs them.
- `*_solve_refined` could pass `Accumulator` to the inner solves as well as the
  factorization. Whether that helps depends on how much of the refinement's accuracy
  the low-precision solve limits.
