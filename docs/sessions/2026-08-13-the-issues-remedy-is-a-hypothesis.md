# Session: the issue's remedy is a hypothesis

**Date**: 2026-08-13 (spanning 2026-08-12)
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Clear two threading follow-ups left by the #297 rollout, both filed from local
code reviews and both marked "pure performance, no correctness impact": #310
(refactor paths rebuild solve schedules they could reuse) and #313 (the dense
matrix element-wise sweep parallelizes over rows only, so wide/short shapes run
serial). Two PRs merged: #422 and #423. One issue filed: #424, the half of #313
deliberately not done here.

## The through-line

Both issues were written by someone who had read the code carefully, and both
named a specific remedy. Both remedies were **hypotheses about where the cost
is**, and in each case the thing that decided the matter was a measurement or a
control — not the reasoning, which sounded right in both directions.

| The issue said | What checking it showed |
|---|---|
| #310: stop rebuilding the schedules and the refactor gets faster | Doing exactly that: **~2%**. Copying `prev`'s schedules costs about what rebuilding them costs, so the O(nnz) work moved rather than left. The 23% needed the copy gone too — the in-place path the issue also mentioned |
| #313: decompose `r = t / num_cols`, `c = t % num_cols` inside the body | Correct but pays a division per element. One division per *chunk* plus an increment does the same job; the flattening is the idea, the div/mod was incidental |
| #313: a `1 x 300000` expression will split once flattened | True — but a test that asserts the *result* cannot see it. A probe expression recording thread ids can, and fails on the old sweep (`1 > 1`) |
| CI: "three Linux x64 jobs failed" | All three died in `Set up sccache` (`socket hang up`) before a compiler ran. The GCC sibling job on the identical code passed — which is the tell |

The habit that carried over from the last session (*run the control*) applied
cleanly to #313's test. What was new here is that it also applies to
**performance work**: a change that provably removes work is not the same as a
change that provably runs faster, and only the benchmark separates them.

## Work Completed

### #310 (#422) — the fix as specified was 2%

The encapsulated LU factors install factors and their coupled level-solve
schedules together through `set_factor`, which always rebuilds. The schedules are
value-agnostic — they cache *positions* into the factor's value arrays — and a
same-pattern refactor leaves `col_ptr`/`row_ind` byte-identical, so the rebuild is
redundant. All true, and all it was worth was 2%.

The reason is the shape of the API. `sparse_lu_refactor(A, prev)` returns a **new**
factorization, so it copies `prev` — and the schedules are two `nnz`-length
`size_t` arrays each. Copying them costs about what rebuilding them costs. The
first implementation therefore traded a rebuild for a copy and reported, honestly,
almost nothing:

| path | ms / refactor |
|---|---|
| `sparse_lu_refactor` (before) | 26.3 |
| `sparse_lu_refactor` (schedules reused, still copied) | 25.7 |
| `sparse_lu_refactor_in_place` | **20.3** |

6400-DOF 5-point grid, `nnz(L)+nnz(U)` = 390K, GCC -O2, min of 5 runs. The
machine was noisy enough (28–35 ms spread on a single binary) that a single run
proved nothing in either direction; min-of-5 and a benchmark that isolates the
*install* step were both needed before the 2% was believable enough to act on.

So the deliverable became the in-place entry points — which #310 had itself
suggested (`refactor_in_place`) and which I had initially treated as the lesser
half. They replay values on the installed pattern and install them without
copying pattern, schedules, or permutations. The returning forms keep their
signatures as "copy `prev`, refactor in place", so no call site changed and
`native_klu_refactor` inherited the improvement through its per-block call.

Two invariants fell out of doing it properly rather than minimally:

- **`set_factor_values` re-checks the schedules' `built_nnz` guards**, not just
  the value counts. Those are the same guards the solves assert on, so the
  "no silently-stale schedule" acceptance criterion is enforced at the install
  rather than trusted.
- **Ordering the install first makes the whole refactor strongly exception safe.**
  Everything after it is noexcept, and values are computed into scratch arrays,
  so a zero pivot leaves the factorization untouched and still solvable with its
  previous values. That is asserted with a test that refactors a same-pattern
  matrix with a zero diagonal and then solves the original system.

### #313 (#423) — observing the split instead of inferring it

Row-per-work-unit means `parallel_for` sees `n == num_rows`, and it runs serially
below `grain * 2`. A `1 x N` expression is therefore permanently serial. Flatten
to `[0, rows*cols)` and it chunks like anything else.

The parts worth recording are the two that the issue's sketch did not specify:

- **One division per chunk, not per element.** Each chunk computes `(r, c)` once
  at its start and then increments with a rollover. A div/mod per element would
  have been a real cost on the cheap expressions this path exists to serve.
- **`work_per_elem = 1` reproduces the old chunk size exactly.** `grain = 65536`
  elements is what the row form's `65536 / num_cols` rows already produced, so
  tall matrices split identically and only the boundary alignment moves
  (row-aligned → element-aligned) — invisible to element-wise results.

For the test, asserting `D(r,c) == A(r,c) + B(r,c)` on a wide matrix proves the
answer, not the parallelism: it passes just as well when the sweep ran serially.
A `probe_expr` whose `operator()` records `this_thread::get_id()` into its own
slot makes the criterion directly observable, and reverting only `dense2D.hpp`
confirmed it fails on the old code with `1 > 1`. That check took two minutes and
is the difference between a regression guard and decoration.

### #424 — splitting rather than half-solving

#313 carried two independent limitations. The second — the grain being ~K× too
coarse for the opt-in lazy `mat_mat_times_expr`, whose `operator()(r,c)` is a
length-K inner product — needs a public per-element work-hint trait, which is an
API decision rather than a one-line change. Theo chose flattening-only for the
PR, so it moved to #424 carrying the full design (a `traits::elem_work`
customization point following the existing `is_expression`/`category`/`ashape`
idiom, and why the hint must be a runtime function: K is a runtime dimension).
#313's body was annotated in place rather than rewritten, and the PR's footer
moved from `Relates to` to `Resolves` once the issue's scope actually matched
what the PR delivered.

Verified while writing #424 that `mat_mat_times_expr` is the **only** expression
in `mat/expr/` or `vec/expr/` with non-unit per-element cost, so the scope claim
in the issue is checked rather than assumed.

## The parts that went wrong

- **I propagated a doc inaccuracy by copying it.** The new `parallel_ewise_2d`
  comment said "Serial (a single body call) at `MTL5_NUM_THREADS=1`", carried
  over verbatim from `parallel_ewise`, where it describes the single *chunk*
  call. For a per-element body it reads as "body runs once", which is false.
  CodeRabbit caught it. Fixed in both places (the second on Theo's ask), which is
  the right outcome — but the mechanism is worth naming: copying a neighbouring
  comment inherits its ambiguities along with its style.
- **The 2% version was nearly shipped.** Nothing about the code was wrong; it did
  what the issue asked. Had the benchmark been run *after* opening the PR rather
  than before, the honest report would have been "removes an O(nnz) rebuild" with
  a number attached that nobody would have questioned.
- **CI went green on a stale commit.** The re-triggered matrix passed on `7fd6637`
  while an uncommitted doc fix sat in the working tree. Merging on that green
  would have dropped the requested fix; the merge waited for `c68302b`. Green is
  a property of a **commit**, not of a PR.

## Tooling notes

- `gh pr edit` fails against this repo with a projects-classic GraphQL
  deprecation error. `gh api -X PATCH repos/{owner}/{repo}/pulls/{n} -F body=@file`
  works and is the workaround for any PR body/label edit.
- Three x64 jobs failed at `Set up sccache` with `socket hang up` while fetching
  the sccache tarball from GitHub releases. Diagnosis is the failing **step**,
  not the failing job list: the jobs died before compiling, and the GCC sibling
  on identical code passed. `gh api .../actions/jobs/{id} --jq '[.steps[] |
  select(.conclusion=="failure") | .name]'` answers this in one call when
  `gh run view --log-failed` returns nothing.

## Issues and PRs

- **Closed**: #310 (via #422), #313 (via #423)
- **Filed**: #424 — the per-element work hint split out of #313
- **Merged**: #422, #423 — both green on the full matrix including the TSan lane;
  #423 also cleared Tier-2 regression. Local validation for both: GCC 163/163,
  Clang 163/163, and for #423 an explicit ThreadSanitizer run of
  `op_test_ewise_sweep_mt` at `MTL5_NUM_THREADS=4`

## Lessons

- **An issue's proposed remedy is a hypothesis about where the cost is.** #310's
  reasoning was correct in every particular and still produced 2%, because the
  cost had moved to the copy the API forces. Benchmark the remedy, not just the
  premise.
- **Removing work and running faster are different claims.** The first is proved
  by reading the diff; the second needs a measurement, on a noisy machine, with a
  statistic (min of N) chosen before looking at the numbers.
- **A performance test must observe the mechanism, not the result.** Both paths
  return the correct matrix, so no assertion on values can tell a split sweep
  from a serial one. Record thread ids — and confirm the assertion fails on the
  old code, or it is proving nothing.
- **Copying a comment copies its errors.** The one review finding this session
  was inherited text, not new prose.
- **Green belongs to a commit.** Two pushes landed after a green board this
  session; both times the question worth asking was "green on which SHA?"
