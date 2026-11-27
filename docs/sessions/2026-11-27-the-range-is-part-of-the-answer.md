# Session: the range is part of the answer

**Date**: 2026-11-26 (evening) to 2026-11-27 (the afternoon's log is `2026-11-26-the-estimate-is-not-the-residual.md`)
**Duration**: Seven PRs
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Resolve five issues and the follow-ups they raised, each driven to merge with CodeRabbit's free allowance only:

- #946, KLU refactor diagnostics;
- #949, a preconditioner that is not positive definite;
- #944, a benchmark regression gate;
- #953, the UKF examples;
- #955, CG with a scaled preconditioner.

**Merged:**

| PR | Issue | What |
|---|---|---|
| #951 | #946 | `rgrowth()` and `rcond()` for native KLU and sparse LU |
| #952 | #949 | Error code 7 in `minres` and `cg` for a preconditioner that is not positive definite |
| #954 | #944 (items 1 and 2) | The instruction-count gate (`bench-regression.yml`) and `mult` dispatch tests |
| #956 | #946 (optional items) | `condest()`, `solve_transpose`, and `native_klu_refactor`'s opt-in growth threshold |
| #957 | #953 | The UKF examples' conclusions computed from their output |
| #959 | #955 | CG's step rescaled when pᵀAp leaves the range |

**Filed:**
- #955, the CG scaling bug, found while writing #952's tests;
- #958, the same exposure in BiCG and BiCGSTAB, found by #955's audit.

#944's item 3, the wall-clock tracking, is open as its follow-up.

## Decisions

- **#944, recorded in the issue and in memory:**
  - a 2% threshold;
  - an intended slowdown is accepted by a committed `benchmarks/regression/accepted.json` (kernel, `max_ir`, reason), not by a PR label. The user chose this when asked;
  - wall-clock tracking becomes a local scheduled run that publishes into `benchmarks/data/`;
  - items 1 and 2 go in the first PR.
- **The gate counts instructions in a pinned environment.** Valgrind passes the host's cache description through, and the blocked GEMM derives kc and mc from it. So `ir_gate.py` pins:
  - one thread;
  - kc = 256 and mc = 96, through `MTL5_ENABLE_BLOCKING_OVERRIDE`, which the build must define;
  - the L3-free nc model.

  The gate therefore measures the GEMM at one fixed blocking, deliberately. How the blocking responds to a cache is a wall-clock question.
- **#949's imaginary part is not tested.** An ill-conditioned but Hermitian positive definite M gives vᴴM⁻¹v an imaginary part of order cond(M)·ε through its solve's rounding. No tolerance independent of M separates that from a non-Hermitian M.
- **BiCG and BiCGSTAB went to their own issue (#958).** #955's fix rescales one direction. BiCG has two direction sequences, and BiCGSTAB's exposure is in ω = tᵀs/tᵀt.

## The through-line

Most of the defects found this session were range defects, not precision defects. A quantity that the mathematics says is finite and well-scaled still had to be formed without leaving the floating-point range on the way.

- **#955.** For M scaled by s, ρ = rᵀM⁻¹r scales as 1/s, but pᵀAp scales as 1/s². At M = 4·2^600·I, pᵀAp underflowed to 0: a false breakdown, code 2. At 2^−600 it overflowed, α was 0, and CG ran to its limit.
  - The fix redoes the step on p̂ = 2^−e p, exactly, only when pᵀAp is not a normal number. The normal path is untouched, bit for bit.
  - The first form of the fix multiplied by one factor 2^−e. CodeRabbit found that this overflows when p has become subnormal: with M⁻¹ = 2^−1000·I the step went NaN after 14 iterations. The scaling is now applied per value.
- **#952.** The code-7 tolerance n·ε·‖v‖·‖M⁻¹v‖ multiplied the two norms first. In CodeRabbit's 2 × 2 example ρ = −2^1023 is finite, the product overflowed, and the tolerance became infinite.
- **#956.**
  - ‖A‖₁ overflowed for A = 1e308·[[1, 1], [1, −1]], whose κ₁ is 2.
  - Higham's last probe doubled ‖y‖₁ before dividing, which overflowed for 2^−1022·I.
  - `condest` now scales the probes by 2^(e−2) near ‖A‖₁, so the solves return values the size of κ.
  - A zero A then needed a NaN, and MSVC rejected the constant `Value{0} / Value{0}` (C2124).
- **#957** is the same through-line in prose. The UKF examples attributed float32's trouble to Cholesky's square roots on an ill-conditioned P. Two filters run side by side show something else.
  - The factorizations agree to rounding in float64, by 7.8e-8 at α = 1e-3 and 3.1e-13 at α = 1.
  - The float32 trouble is the weights. W₀ = −986894 at α = 1e-3, which cancels six of float32's seven digits. The step-0 bias is already 2.2e-2 at cond(P) = 1, and the two float32 filters drift 1.9 apart.
  - At α = 1 the gap is 6.6e-5.

## Findings that changed what was claimed

- **#953's issue said the indefinite test used diagonal matrices.** It already rotated them. The test kept them and added two cases:
  - a shifted Hilbert matrix;
  - ones with a tiny diagonal, where unpivoted LDLᵀ's solve has a backward error of 1.8e-4 and Bunch–Kaufman's 1.6e-16.
- **The float32 bearing-only filters no longer stop at step 5.** With one shared double-precision scenario and the bearing wrap, they run all 15 steps and drift apart instead. The diagnosis is the same: it follows α, not the factorization.
- **Cholesky's backward-error claim needed its condition.** It holds when the factorization completes. Completion is guaranteed for c·n^{3/2}·κ·ε < 1. Beyond that, a factorization *may* stop; diag(1, 2^−100) still factors exactly. CodeRabbit corrected the docs twice before the wording held.
- **`condest` matched the exact κ₁ on every matrix compared**, including a 1-D Laplacian of order 200, κ₁ = 2.0e4. It is a lower bound, as Hager's estimate is.

## Process notes

- **CodeRabbit's allowance.**
  - It ran out after nearly every push, about 45–55 minutes each time.
  - Paid offers from $0.25 to $3.50 were declined throughout, as the user asked.
  - Each wait went to preparing the next PR.
- **Worktrees.**
  - #953 and #955 were prepared in `git worktree`s off `main` while earlier PRs were in review, each committed but not pushed until its predecessor merged.
  - Removing a worktree is needed before `gh pr merge --delete-branch` can delete the local branch. The first try at #957 stopped there.
- **The gate's first runs.** #954's PR run had no baseline and reported only. Every PR after it (#956, #957, #959) passed the gate against main's baseline.
- **Infrastructure noise.** One LAPACK job failed on a 403 from `packages.microsoft.com` during `apt`. It was rerun, not fixed.
- **Valgrind locally.** It was unpacked from Ubuntu's `.deb` into the scratchpad (`apt-get download` and `dpkg -x`, with `VALGRIND_LIB`), not installed.
- **Merging.** Every merge was gated in the same command on zero unresolved threads, after checking that CI ran on the head and that CodeRabbit's `coveredCommitId` was the head.
