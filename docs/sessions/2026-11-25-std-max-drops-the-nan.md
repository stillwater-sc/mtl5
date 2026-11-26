# Session: std::max drops the NaN

**Date**: 2026-11-25 into 2026-11-26
**Duration**: One day, through a GitHub API outage in the afternoon
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

- Choose the work for 5.26.0. #923 had already made it a minor release.
- **QA sweep #913** (iterative solvers): triage it, then resolve its tracking issues in the order the user gave: #924, #925, then #926, using CodeRabbit's free reviews only.

**Merged:** #927 (#924), #931 (#925), #932 (#926).

**Filed:** #924, #925, #926 (the tracking issues of #913), and #933 (animations for the pedagogical examples).

**Closed:** #924, #925, #926. The sweep #913 is still open, pending its closing map.

## The through-line

The same bug showed up three times, each time in a test written to catch NaN:
- `std::max(r, NaN)` returns r, so a residual check that folds its terms through `std::max` passes a NaN solve.
- My first fix in #931, `if (!(d <= r)) r = d`, let the next finite term overwrite a NaN already in r. CodeRabbit caught it on the thread.
- In #932 the same pattern sat in `max_eigen_residual` and in QR's `normal_residual`.

Each now returns the first non-finite term.

The library code had the same lesson, at a larger scale. `scaled_rhs` scales b by an exact power of two so that the float dot products neither underflow nor overflow, and its first bound took three review rounds to get right:
- **Underflow.** Scaling down a huge b sent a tiny entry of x0 to zero, and x0 did not come back.
- **Both ends.** The underflow clamp then undid the overflow clamp: b = (2^-100, 0) with x0 = (2^127, 2^-149) chose 2^23, and x(0) became inf.
- **The modulus.** The overflow bound used the modulus, which for complex<float> max + max i is itself inf, so no bound applied at all.

The final rule is one range: overflow sets the upper end, underflow the lower, both from the real and imaginary parts separately. A test pins each of the three failures.

## #924: the Krylov solvers

- **Complex systems.**
  - bicg and qmr built their shadow sequence with Aᵀ while the dots conjugate, so they never converged on a complex non-Hermitian system. `apply_adjoint` forms Aᴴx = conj(Aᵀ conj(x)), so a matrix-free operator still needs only `trans(A)`.
  - tfqmr did not compile for complex.
  - bicgstab_ell's MR dots had their arguments reversed: 47 iterations at n = 40 where BiCGSTAB takes 21.
- **`scaled_rhs`.**
  - Every solver now copies b, so `cg(A, x, x, …)` works.
  - The dot-product solvers scale an extreme b by a power of two. Inside the band (a quarter of `max_exponent`) nothing changes. Outside it, the result is the unscaled one bit for bit, checked at 2^-600.
- **Non-finite residuals** finish with the new error code 6, where a NaN in b ran all 100 000 iterations.
- **gmres** drops a zero-diagonal breakdown column, where it wrote NaN into x, and allocates once per solve.
- **CI** found two wrong assumptions in my own tests:
  - an iteration count off by 2 on macOS ARM64;
  - an exactness claim made for idr_s, which takes b's copy but not the power-of-two scaling, so it promises no bit-identity across scales.

## #925: preconditioners, multigrid, eigensolvers

- **Pivots.** ic_0 and ildl now reject a bad pivot at construction, under the #867 convention. One test pinned the old behaviour, IC(0) giving NaN on Hilbert(9) in float, and now expects the throw.
- **Cost.** ic_0 and ildl were cubic in row length, because they rescanned row j for each k of row i. A sorted merge visits the same pairs in the same order: from bandwidth 16 to 128 the build grows 39× where it grew 404×. The issue had asked for about 2× per doubling, but 4× is the floor for this pattern.
- **Other fixes:**
  - block_diagonal throws for a singular block;
  - the shape checks run in every build, not only under assert;
  - the ilut fill limits neither wrap nor overflow;
  - the W-cycle solves the coarsest level once per visit;
  - power_iteration at n = 0 writes its value.
- **Not done, by the user's decision** (option A, recorded in the PR): persistent workspace for multigrid and Jacobi. It would need `mutable` state in the const cycles and sweeps, against the #405 rule in `smoother/sor.hpp`, and nobody has measured the allocations.

## #926: the sweep range's own commits

- **Panel LU.**
  - The panel test barely reached the panels: dense_unsym(40) is one supernode, and random_unsym(150) made 0 updates at the default width. The new `panel_updates` counter shows that the larger fixtures take them.
  - The k = 0 pre-pass, which could apply nothing and switched panels off for 16, is gone.
  - The 16 n workspace is allocated on first use.
- **Level-scheduled solves.** The factor classes check their diagonals once, at install, and their solves skip the O(n) pass. The free functions still check on every call.
- **The int32 limit** is checked first, so a test reaches it with a 2×2 A.
- **The `keep_scalar` question** settled by reading. It applies only to a row-major A, and the `sweep_fixed_rows` branch runs only on a column-major one, so the dropped calls were no-ops. `static_assert`s now keep it that way.
- **The remaining items were tests:**
  - sparse QR on irregular patterns;
  - `reach()`'s cost;
  - frobenius_norm's threaded column paths, run in the file that forces 4 threads, since the default pool has 1;
  - syrk/syr2k at k = 0 and 1;
  - symv with a strided x;
  - eigen()'s per-index pairing.

## CodeRabbit and the outage

- **The allowance.** Free reviews come at one an hour, and promoting a PR spends one. Each fix round therefore cost about an hour of waiting. Paid on-demand reviews, offered at $0.50 to $3.75, were declined, as the user asked.
- **The outage.** In the afternoon GitHub's GraphQL API returned 500s and 502s: `gh pr ready` failed four times, and the CodeQL "Analyze (python)" upload failed. That run could not be re-run (a default-setup run), and the `ready_for_review` event never fired Tier 2 or CodeRabbit. An empty commit retriggered all three.
- **The pre-merge check.** CodeRabbit's "Linked Issues" check flagged the items the user chose to leave unchecked: the IDR(s) seed pin (#924) and the workspace items (#925). Both decisions are in memory, so the next sweep does not raise them again.

## Also

- **#933** tracks animations for the ten pedagogical examples in `docs/modernization/examples.md`. Example programs emit traces of MTL5's own numbers, and SVG components replay them, embedded beside each example (the user's choice). There are no new npm dependencies, and the components have tests in `docs-site/tests/`.
- **An observation, not filed:** on the unpreconditioned 100² convection-diffusion grid, bicg diverges and qmr, cgs and tfqmr stall, with main's headers as well. The new regression cases leave those combinations out.
