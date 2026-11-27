# Session: the unnormalized vector

**Date**: 2026-11-27 (late night; the previous log is `2026-11-27-a-sidecar-has-to-say-enough.md`)
**Duration**: One PR
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

#971: MINRES with a preconditioner scaled beyond about 2^±700. Found while testing #968, filed with its cause measured. CodeRabbit's free reviews only.

| PR | Issue | What |
|---|---|---|
| #977 | #971 | The new Lanczos vector's solve is redone on a power-of-two scaled copy when it leaves the range |

## The fix

The issue had measured where the failure starts. The unnormalized v_new scales as s^(−1/2), so its solve z_new = M⁻¹v_new scales as s^(−3/2). That underflows past about 2^716 and overflows past about 2^−683, which matches the failures.

Of the issue's two options, (a) was taken: rescale v_new before the solve, by 2⁻ᵉ with 2ᵉ ≤ max|v_new|. The reasons:
- **It is exact.** β̂ = 2⁻ᵉβ_new, because the M-norm of a power-of-two multiple is that multiple of the M-norm. The recurrences take β_new = 2ᵉβ̂, and v and z are normalized by β̂, which gives the same vectors.
- **It costs nothing in range.** The test for a solve that left the range runs only after the plain M-norm (#968) has already failed. So the normal path has no extra pass, and stays the same bit for bit. Normalizing v in the 2-norm (option (b)) would have changed the recurrence everywhere.
- **The code-7 flag must be cleared.** The first M-norm, on the out-of-range z_new, is discarded, and so is any code-7 flag it set. A test covers both directions: an indefinite M reached at a later step still returns 7 at 2^±900, and an SPD M at that scale does not.

## Findings

- **MINRES now reaches 2^±1000.** That is further than the rescaled CG, BiCG and BiCGSTAB (#959, #967), whose tests stop at 2^±900. MINRES's normalized vectors scale only as s^(±1/2).
- **The tests fail on main** with the codes the issue measured: 6 at 2^−700 for the agreement test, and 2 or 6 at 2^±900 for the code-7 test.
- **Benchmark timings drift between runs.** The 128² benchmark ran at about 31 ms for both main and this branch, alternated, against 21.7 ms when #968 was measured. The two builds were timed back to back, so the comparison holds. The absolute value moved with the machine's state, and was not investigated.

## Process notes

- **CodeRabbit** ran out once, for 26 minutes. A $0.50 offer was declined. After the wait, the requested review found nothing, and #977 was merged by the full-cycle rule: CI on the head, CodeRabbit's `coveredCommitId` the head, and zero unresolved threads, checked in the same command as the merge.
- **The remote branch survived.** `gh pr merge --delete-branch` stopped at the local branch, which the worktree held. It was deleted by hand afterwards. The same happened earlier in the session, so the worktree has to be removed before the merge, not after.
