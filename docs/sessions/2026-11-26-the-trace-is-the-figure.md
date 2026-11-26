# Session: the trace is the figure

**Date**: 2026-11-26
**Duration**: One day, three PRs
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

- Close the QA sweep #913 with its map, once the user had merged #934 (the session log of 2026-11-25).
- Resolve #933, animations for the pedagogical examples. The user added a request: pedagogical examples for Phases 1 and 2, containers and views/permutations, so that the page is a complete progressive introduction to MTL5.
- Review with CodeRabbit's free allowance. The user paid for one review on #935.

**Merged:** #935 (part 1), #939 (part 2), #940 (part 3).

**Closed:** #913 (sweep, with its closing map), #933.

## Decisions

- **Three PRs** (the user's choice; recorded in memory):
  1. infrastructure, the new Phase 1–2 examples and 3A;
  2. 3B to 5B;
  3. 6A to 7B, which closes #933.

  The issue had proposed one PR per phase. Each PR costs a full CI cycle and an hour per review round.
- **Phase 2 gets its own directory**, `examples/phase02_views_permutations/`. The existing `phase02_basic_operations/` holds norms and arithmetic and was left alone.
- **Where the files live:**
  - the page is `docs/site/modernization/examples.mdx`, an MDX page copied verbatim so that it can embed components;
  - the traces are in `docs-site/src/data/traces/`, imported with `?url`, so the build inlines small ones and emits large ones as hashed assets under `/mtl5/`;
  - the components are in `docs-site/src/components/examples/`, with the shared player, plot helpers and models in `lib/`.

## The through-line

The figures draw the numbers MTL5 computed and nothing else. Each example writes a trace with `--trace <file>` (`examples/common/trace.hpp`), and the JavaScript only interpolates and draws. Three consequences shaped the work:

- **Iterates come from capped runs.** The Krylov solvers report a count and a residual, not their iterates. A run capped at k iterations follows the same recurrence as the full run, so its x is the full run's x_k (`examples/common/replay.hpp`).
- **Intermediate factorizations come from the factors.** For 5B, LU's working matrix after k steps, Cholesky's Schur complement and the partial Householder products are each rebuilt from what `lu_factor`, `cholesky_factor` and `qr_factor` return. Undoing the later row swaps, last first, recovers LU's row order at step k.
- **The captions are tested too.** Every number a caption quotes is computed from the trace at build time, and the claims behind them are pinned in `docs-site/tests/`: that 3A's preconditioned run converges at step n, that one rearrangement of Dürer's square is not magic, that Cholesky fails from degree 15. A library change that moves a number fails a test before it can leave a stale sentence. There are 164 tests.

## The examples

- **New, Phase 1:**
  - 1A, Leontief's input-output economy, solved as rounds of production: 30 rounds to 1e-10.
  - 1B, Kirchhoff's stamps on a Wheatstone bridge: `update_plus`, an amending inserter, the CRS arrays.
- **New, Phase 2:**
  - 2A, Dürer's 1514 square read through `trans` and `mapped`: 16 views sum to 34, and 6 of 7 rearrangements stay magic.
  - 2B, Cuthill–McKee on a shuffled 12 × 12 grid: bandwidth 128 → 12.
- **What the traces showed** that the programs had not said:
  - 3B: CG reaches its 500-step cap at every ε. At ε = 0.01 the converged answer overshoots to 1.2, because the cell Péclet number is 1.61. That is a discretization error, not the solver's.
  - 5A: the normal equations stall between 3e-5 and 1.2e-4 from degree 11 while QR reaches 1.2e-10. The eigenvalue estimate of cond(VᵀV) saturates near 1e22, where cond(V)² is 3e35.
  - 4B: the ω sweep's best value is exactly 2/(1 + sin πh). Gauss-Seidel cuts the oscillatory mode by 10⁵ in 10 sweeps while the smooth one barely moves.

## Bugs found along the way

- **`vibrating_string.cpp` contradicted itself.** Part 3 was titled for complex eigenvalues, and its takeaway said non-symmetric perturbations create complex pairs, but its one asymmetric entry left every eigenvalue real. It now says so and sweeps a lopsided Laplacian (the operator of 3B), which goes complex for d > 1. My first note claimed rounding would scatter the eigenvalues at d = 1. The run showed them exactly 2: the matrix is triangular there. The note was corrected before commit.
- **CodeRabbit, eight findings, all valid:**
  - `trace.hpp` kept references into a `std::vector` that the next insertion could reallocate: a use-after-free on the trace path, confirmed with ASan, and fixed with deque storage.
  - A trace that failed to write still exited 0.
  - The player skipped the rest of a paused tween.
  - The status line was not a live region.
  - The self-test's temporary directory could collide between build trees.
  - `regen_example_traces.sh` could run a stale binary from another configuration.
  - The shell's CSS `stroke-width` silently overrode every per-element width, because a CSS rule beats an SVG presentation attribute.

  Each fix has a test that fails without it.

## Process notes

- **Builds and the gate.** A build of another project was running when the session started its first. The gate script then matched its own command line, the pitfall already in memory, and was rewritten to match process names (`pgrep -x`). `pkill -f "astro preview"` killed its own shell for the same reason.
- **Stacking without force-pushing.** Parts 2 and 3 were built locally on top of the PR before them. After each squash merge they were cherry-picked onto a fresh branch from main; the commits had never been pushed, so nothing was rewritten.
- **Reviews.** Free reviews come at one an hour. Paid offers ($0.75 to $5.25) were declined, except for one paid review on #935, as the user asked.
