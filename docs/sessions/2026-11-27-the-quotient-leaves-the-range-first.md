# Session: the quotient leaves the range first

**Date**: 2026-11-27 (evening; the afternoon's log is `2026-11-27-the-tracking-run-finds-what-the-gate-cannot.md`)
**Duration**: Two PRs
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Two follow-ups of the day's work, each with CodeRabbit's free reviews only:
- #958, BiCG and BiCGSTAB with a preconditioner scaled beyond 2^±512, which #955's audit found;
- #963, MINRES about 45% slower per iteration, which the first tracking run found.

**Merged:**

| PR | Issue | What |
|---|---|---|
| #967 | #958 | `bicg` and `bicgstab` rescale their out-of-range product by exact powers of two; `itl/krylov/pow2_rescale.hpp`, shared with `cg` |
| #968 | #963 | `minres` forms vᵀM⁻¹v plainly when it is in range; `minres+diag` in the instruction-count gate |

Both were merged by hand during a CodeRabbit wait, after the review fix for their one finding had passed GCC and Clang locally.

## The through-line

The morning's lesson was that a product of two directions leaves the range before anything else in a Krylov step: CG's pᵀAp, scaling as 1/s². Both of today's reviews found the next layer of the same lesson. Once the out-of-range quantity is rescaled, the quotient or the tolerance built from it can leave the range in its turn.

- **#967, the quotient.**
  - **What was fixed.** BiCG's p̃ᴴAp and BiCGSTAB's tᵀt were the products, as the issue said. The rescaled coefficient 2⁻ᶠρ/d was formed as ρ/d first.
  - **What CodeRabbit found.** For A = [2⁻²⁰⁰] with M⁻¹ = 2⁹⁰⁰, ρ = 2⁹⁰⁰ and d = 2⁻²⁰⁰ are both in range, but ρ/d = 2¹¹⁰⁰ is not, while the coefficient 2²⁰⁰ is. x went to inf.
  - **Also in CG.** #959's CG had the same quotient.
  - **The fix.** `scaled_quotient` divides the two mantissas and applies one power of two. Division rounds the same on values scaled by powers of two, so the bit-for-bit tests at 2^±900 still held.
- **#968, the tolerance.**
  - **What was fixed.** The fast path took vᵀz directly when it was in range.
  - **What CodeRabbit found.** For a negative vᵀz, the code-7 tolerance n·ε·‖v‖·‖z‖ used norms that can overflow where vᵀz does not. With 16 entries of v = 2⁻⁸ and M = diag(−2⁻¹⁰³¹), vᵀz = −2¹⁰¹⁹ but ‖z‖ = inf. The tolerance was infinite, and an indefinite M passed.
  - **The fix.** A negative value now goes to the scaled form, whose norms are finite.

## Decisions and findings

- **The rescaling helpers are shared.** `is_normal_magnitude`, `max_abs_exponent`, `by_pow2`, `scale_by_pow2` and `scaled_quotient` live in `itl/krylov/pow2_rescale.hpp`, used by `cg`, `bicg` and `bicgstab`. The header has a line in the benchmark coverage manifest; #952's CI failure was this omission.
- **BiCG's two exponents are exercised separately.** p and p̃ each get their own power of two. A temporary print showed 12 of the tests' 163 rescaled steps with e ≠ f, so the bit-for-bit check would catch the exponents being swapped.
- **A BiCGSTAB breakdown test from hand arithmetic was wrong.** The first try (A = diag(1, 0)) was meant to give t = 0, but α came out as 2s, not s, and the solve broke down on r*ᵀv (code 3) instead. The test that replaced it is #866's [[1, 1], [1, 0]], b = e₁, where tᵀs = 0 exactly. At 2^±600 that case goes through the rescaled branch, which must still give ω = 0, code 5.
- **#869 is the cause of #963, measured.**
  - **The bisection.** A standalone timing with headers from 35dc6a9~1, 35dc6a9 and main gave 20.5, 29.8 and 30.3 ms, all in 233 iterations. The fix gives 21.7 ms.
  - **The residue.** The remaining 6% is presumably the preconditioned Lanczos #869 introduced; not measured further.
- **MINRES at 2^900 breaks down (code 2)** with or without #968's change. The new test stops at 2^±600, and its comment says why. No issue is filed yet.
- **The gate now counts MINRES.** `krylov/minres+diag/f64` is the 15th kernel. Its first count on main is the baseline.

## Process notes

- **A foreign build twice.** kpu-sim's `ninja -j 16` was running when the #968 tests were to be built, at a load average of 10, and again a few minutes later. The builds waited on a background loop until no `cc1plus`, `ninja` or `gmake` remained, per the one-build rule.
- **CodeRabbit's allowance.** It ran out after each fix push: 50 and 52 minutes, with offers of $1.50 and $0.50 declined. A 12-minute wait on #968's first push was waited out, and the requested review arrived.
- **Watchers stopped with TaskStop,** not `pkill -f`, after the afternoon's self-match.
