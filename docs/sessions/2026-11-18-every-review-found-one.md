# Session: every review found one

**Date**: 2026-11-17 into 2026-11-18
**Duration**: One night, following the session of
[2026-11-17](2026-11-17-it-ran-the-same-loop.md)
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

- **#236:** the BLAS L2/L3 operators that still had no native-fast path.
- **QA sweep #893** (sparse direct solvers): triage it, then resolve its
  tracking issues in the order the user gave: #907, #909, then #908.

**Merged:** #906 (#236), #910 (#907), #911 (#909), #912 (#908).

**Filed:** #907, #908, #909, the tracking issues of #893.

**Closed:** #893, #907, #908, #909.

## The through-line

CodeRabbit found a real defect in every review round, and most of them were
in code written to fix something else. The fixes were right for the case in
front of them and wrong at an edge next to it:
- **#906:** routing `syrk` through gemmt rounded a double `beta` to float, so
  1e-50 became 0 and C was not read.
- **#910:** keeping an empty triangular column skipped, to match the level
  schedules, solved the system as if its diagonal were 1. And a schedule
  built once could still be handed a zero diagonal with the same pattern.
- **#911:** resetting IDR(s)'s redraw budget reset the seed with it, so a
  redraw repeated a space already tried. Walking a column-major matrix by
  columns left a 10⁶ x 1 matrix on one thread.
- **#912:** a macro-overridable `inline constexpr` was an ODR hazard, and
  `&UK[i]` asked for an `operator&` the number types need not have.

Each one had a test that failed without the fix.

## #236: what was left of it

The issue (July) listed nine operators with no fast path. #724, #744 and
#747 had since covered trsm, trmm, symm, gemmt, trmv and trsv, and ger was
at 91% of roofline. What was left:
- **`syrk` and `syr2k`** go through gemmt's blocked path: 52.0 and 53.8
  GFLOP/s at 512³, against OpenBLAS's 52.7 and 56.6.
- **`syr2k`'s first form was slower.** One gemmt over `[A B][B A]^T` was
  1.4x slower than two calls. Allocating and first-touching the two copies
  cost more than the product saved.
- **`symv`** takes strips of 8 rows. It is bit-identical, and 2x faster at
  n = 4096.

The issue's custom-number-type criterion was out of the user's chosen scope,
so #236 is still open.

## The sweep: three tracking issues

The triage mapped every finding:
- two had already been fixed, by #901 and #902;
- the `numeric_limits` sub-items were covered by the #838 decision;
- the BLAS `int` guards were not filed, because they need dimensions above
  2^31 to test.

**#907, correctness.** The shift-invert floor scaled as the magnitude
squared, which gave wrong eigenvalues reported converged at 10⁶. There was an
out-of-bounds read on an empty upper column. Diagonals were only asserted.
QR ignored the ordering for m > n. QR was also quadratic on a tridiagonal
matrix; it now touches only each column's pattern, is 27x faster at
n = 4000, and gives bit-identical factors on 400 random matrices.

**#909, follow-ups.** The IDR(s) redraw bug was confirmed before it was
fixed. An operator returned 0 on chosen calls, and an iteration reported
convergence on chosen calls, so IDR(s) restarted without a breakdown. Four
such breakdowns gave code 2 on main.
- **Decision (a):** expression operands stay copied dense. The issue's
  "test that a sparse expression does not allocate" could not pass under
  that decision. It became a test of the path that avoids the copy: forming
  the sparse matrix first, at 200000², where a dense copy would be 320 GB.
- **`saturating_cast`:** the fallback branch the sweep found "unreached" is
  reachable, by a type with only explicit integral conversions.

**#908, performance**, all bit-identical:
- `native_klu`'s solve is 3.7x faster on 1x1 blocks.
- The permutations no longer use the 5-slot inserter: a refactor with
  7-entry rows is 3.6x faster.
- **LDL^T loop order:** the sweep suggested putting the supernode's columns
  outermost in the block update. That was 0.90x on a 2D grid and 1.34x on a
  3D one, so it now applies from width 2, and the 3D grid runs 1.39x faster.
- **Measured and dropped:** deriving the LDL^T elimination tree from the
  Cholesky one made no measurable difference.

## Lessons

- **The `--limit-seen` watcher misses the limit lifting.** Twice it waited
  over an hour after CodeRabbit's included review had become available. The
  fix was to read the wait from CR's comment and post `@coderabbitai review`
  when it ran out.
- **A paid review was requested on a commit whose CI had failed.** The wait
  loop did not stop on the failure. It costs $0.75 and reviews a commit that
  is about to be replaced. The loops since gate the request on a green CI.
- **Bit-identity tests need contraction off on every platform.** The native
  KLU test passed on x86 and failed by an ulp on AArch64's SVE2 lane. That
  is the same caveat as the other bit-identity targets, and the test joined
  their list.
- **MSVC reads more of a `numeric_limits` specialization than GCC and
  Clang.** A test type with a partial specialization compiled on Linux and
  failed on Windows (`is_signed`).

## Open

- **#236:** the custom-number-type criterion (a blocked path for
  non-float/double types). Whether to close the issue is the user's call.
- **The BLAS dispatch of `syrk`/`syr2k` casts a double `beta` to float** for
  float matrices: it predates #906, and was noted on that PR.
- **An ASan job in CI** for the sparse tests: declined on #910 as out of
  scope, and not filed.
- **Release 5.24.0:** the range since v5.23.0 has 19 `feat`s and four
  breaking changes. `MTL5_FALLBACK_VERSION` and the peer-repo builds
  (mtl5-python, mp-spice) are still to do.
- **#195:** parked at 1.9-2.4x of SuperLU.
