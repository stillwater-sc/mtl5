# Session: the estimate is not the residual

**Date**: 2026-11-26 (afternoon; the morning's log is `2026-11-26-the-trace-is-the-figure.md`)
**Duration**: Three PRs
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Resolve five issues that mtl5-python's ports of the examples had found: #936, #937, #938, #942 and #943. Review with CodeRabbit's free allowance. The user requested one review on #945 by hand.

**Merged:**
- #945 (#936);
- #947 (#937 and #938);
- #948 (#942 and #943).

**Filed:** #944 (a benchmark-regression gate) and #949 (MINRES and CG on a preconditioner that is not positive definite).

## Decisions

- **Two issues per PR where they share files and shape.** #937 and #938 are both example fixes for the docs page, and #942 and #943 are both Phase 13 examples. Each pair took one CI cycle.
- **5B keeps a 4 × 4 indefinite matrix.** #937 suggested `wilkinson(7)`. `wilkinson(4)` has the same inertia (exactly one negative eigenvalue) and fits the animation's panels.
- **#949 gets a new error code 7**, "preconditioner not positive definite". Codes 3 to 5 already mean solver-specific breakdowns. CG's matching negative-ρ check was added to #949's scope. Recorded in memory.
- **A benchmark gate on instruction counts, not wall-clock time** (#944). kpu-sim's `benchmark-regression.yml` gates on GFLOPS computed from simulated cycles, which are deterministic. MTL5's GFLOPS come from wall-clock time, and #454 had already shown that a wall-clock gate on hosted runners fails at random. #944 proposes three things, with four decisions left to the user:
  - cachegrind instruction counts per PR;
  - dispatch tests;
  - wall-clock tracking on the benchmark machines, as a report rather than a gate.

## The through-line

A solver's internal estimate is not the quantity the caller asked about.

- **GMRES (#936).** Left-preconditioned GMRES minimizes and tested ‖M⁻¹r‖, but compared it against rtol·‖b‖. With a diagonal of 400 it stopped at a true residual of 3.5e-6 for rtol 1e-8, and `relresid()` reported 8.7e-9.
  - The fix tests the true residual at the start and end of each cycle, as CG and TFQMR already did.
  - Within a cycle, the Arnoldi estimate is scaled by the cycle's opening ratio ‖r₀‖/‖M⁻¹r₀‖. Without that, a cycle following a false convergence opens below the tolerance and crawls at one step per restart.
- **CodeRabbit's four findings on #945 were all on this boundary:**
  - a NaN M⁻¹r divided into the basis;
  - a singular-breakdown verdict taken from the estimate, while the true residual had converged (CodeRabbit gave a 4 × 4 counterexample);
  - an inserter alive at `return`;
  - the scale ‖r₀‖/β overflowing for a preconditioner of 1e-310. Since |g| ≤ β, (|g|/β)·‖r₀‖ cannot overflow.

  Each fix has a test that fails without it.
- **Example 4A's trace stayed byte-identical through all four commits.** Without a preconditioner ‖r₀‖ == β to the bit, and the estimate is used unscaled.

## What the examples now show

- **5B** (`solve_three_ways`) uses `lehmer(4)`, `wilkinson(4)` and `laplacian_1d`, and adds a Hilbert sweep over n = 4 to 15:
  - the factorizations' backward error stays within 4ε;
  - `inv(H)·b`'s rises from 2.1e-13 to 6.4e-4;
  - Cholesky fails from n = 13.
  A second figure, `HilbertSweep`, replays the sweep. Seven tests pin its caption.
- **6A** (`vibrating_string`) gives the corner bound −1 ≤ s ≤ 5/4: det(B − λI) = μ⁴ − 3μ² + 1 + s.
- **`multigrid_poisson`** coarsens to 3 points with an LU coarse solve.
  - It takes 5 V-cycles and 4 PCG iterations at every n from 63 to 4095. The original 3 fixed levels fail from n = 1023.
  - ‖B − Bᵀ‖/‖B‖ is 3.4e-16 only with symmetric Gauss–Seidel and the exact coarse solve.
- **`krylov_comparison`** prints every return code, and shows:
  - MINRES breaking down on an indefinite M;
  - CGS breaking down without ILU(0), as SciPy's CGS also diverges there;
  - CG's residual rising to 1.24‖b‖ on the Helmholtz matrix, while MINRES's never rises.

## Corrections made before commit

- **The Hilbert commentary claimed too much in its first draft.**
  - "Grows with cond(H)": the inverse's backward error levels off near 1e-4 after n = 11.
  - "Fails once cond > 1/ε": Cholesky still succeeds at n = 12, where cond = 1.8e16.
  - "A lower bound": the SVD's cond is not monotone past 1e16.

  Each claim is now computed from the run.
- **The multigrid sweep looked as though it grew with n at 1e-10** (6, 6, 6, 10). The per-cycle factor is 0.018 at every n. At n = 4095 rounding stops the residual near 1e-10, so the sweep uses 1e-8.
- **The first `krylov_comparison` used a 1-D matrix,** on which ILU(0) is exact and every solver converged in one step. Part 1 is now 2-D.
- **IC(0) of the unshifted Laplacian made MINRES slower on Helmholtz** (277 iterations against 119), so it was left out.

## Process notes

- **CodeRabbit's limit.** The free allowance ran out twice on #945. Paid offers ($0.75, $0.50, $2.50) were declined, and the user requested one review by hand.
- **Building the next PR locally.** #947 was built while #945 was in review, and cherry-picked onto main after the merge. It had never been pushed.
- **Merging.** Every merge was gated on zero unresolved threads in the same command, and on a CodeRabbit review whose `coveredCommitId` is the head.
