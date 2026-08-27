# Session: the green check was checking something else

**Date**: 2026-08-25 – 2026-08-27
**Duration**: Multi-day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Resolve #446 (Universal types excluded from the threaded GEMV/GEMM paths by the
BLAS-scalar gate) and #461 (integer overflow is UB in the sparse and remaining L1
generic loops). Five PRs merged: #500, #501, #502, #504, #506. Two issues filed
along the way (#503, #505) and both then resolved.

## The through-line

Every signal that looked like verification this session was, at least once,
verifying something adjacent to the claim it appeared to support. None of them
were broken; each answered a real question that was not the question being asked.

| What the signal said | What it was actually checking |
|---|---|
| CodeRabbit check: **pass** | That CodeRabbit *ran*. On #501 and #502 it rate-limited and posted a limit notice; **no review happened**, and the check is green either way. On #500 a review *did* land — and I merged 26 minutes later without reading it |
| `static_assert(IlutOk<complex<double>>)` | That the type **name** is valid. Naming a specialization instantiates no member, and `ilut<complex<double>>::factorize` did not compile |
| `lu_factor`'s `FieldMatrix` gate protects `block_diagonal` | The *member's* constraint. Naming `block_diagonal<dense2D<int>>` never instantiates it — the same hole let four wrapper smoothers through untouched |
| Threaded SpMV gains **1.18×**, so threading barely helps | Memory bandwidth. With a multiply that costs what an emulated format costs, the same kernel gains **3.8×** |
| Serial timings flat across runs, so the refactor is free | Noise. Min-of-5 showed a **1.8× serial regression** the single-run numbers had hidden |
| Two monitors reported `WATCH_ENDED` with no findings | One had a `jq` filter that never matched. Silence is not a result — a lesson this log recorded in August and I repeated |

The unifying error is cheap to name and was expensive to keep making: **I checked
that a gate existed and not that the thing behind it worked.** Three defects came
from it — `block_diagonal`, the wrapper smoothers, and ILUT's complex claim. I
found the first and reasoned about it explicitly in a commit message, then failed
to look for the same shape anywhere else. Review found the other two.

## Work Completed

### #500 — separating two questions one predicate was answering

`is_blas_scalar_v` gated *"may this go to OpenBLAS or a SIMD micro-kernel?"* and
*"may these output rows be split across cores?"*. Only the first is about the
scalar type. Every emulated format ran its dense matvec and matmul on one core
regardless of `MTL5_NUM_THREADS`.

`ThreadableDenseMatrix` / `ThreadableDenseVector` carry the storage shape only.
They gate a row-partitioned wrapper around the **generic** kernel, deliberately
not a reroute into `gemm_blocked` — the issue itself warned that blocking earns
its keep through a vectorized micro-kernel and measured ~3.4× *slower* for a type
Highway cannot vectorize.

GEMM n=384 on a class-typed scalar, i7, min of 5:

| threads | before | after |
|---|---:|---:|
| 1 | 0.120s | 0.115s |
| 2 | 0.177s | **0.062s** |
| 4 | 0.113s | **0.031s** |
| 8 | 0.117s | 0.049s |

Two findings the measurement forced. The 65536-unit chunk budget is a wall-clock
quantity counted in operations, so charging an emulated format the same costs
**parallelism**, not overhead — `parallel_for` caps the team at `n / grain`
chunks. And routing the *serial* case through `parallel_for`'s own serial path
cost **1.8×** with assertions live: `mult`'s dimension equalities are also where
the optimizer learns the extents agree, and that does not survive an opaque
`[b, e)` pair passed through a lambda.

### #501 — the same fix for sparse, and the number that flattered it

`mat::operator*(compressed2D, dense_vector)` had threaded this traversal for any
value type since #221; the accumulator-aware `mult_sparse_crs` it duplicates had
not. **3.8× at four threads** on an expensive scalar.

`mult_sparse_crs` keeps its own copy of the loop rather than delegating to the
row-range kernel: delegating measured 6.5% slower, and that lands on `double`
SpMV at one thread — the inner loop of every Krylov iteration.

The transposed kernel **stays serial**, and now says why: it scatters into
`y(indices[k])` from many rows, so bands collide on output. Atomics do not exist
for a custom number type; privatizing costs `O(threads × y.size())` accumulators
and changes the summation grouping, forfeiting the bit-identity every other
threaded path here guarantees.

### #502 — a wrapping contract, and a defect the issue had not listed

Routed the sparse and L1 generic loops through `detail::generic_fma` plus two new
siblings. Also fixed two sites outside the issue's list — the eager dense
`operator*` loops, two lines from the sparse one it *did* name.

`std::abs` is **UB at the minimum of a signed type**, so
`one_norm(dense_vector<int32_t>)` over full-range data was undefined before any
accumulation happened — reached by exactly the test the issue asked for. The
honest consequence is recorded at `infinity_norm`: the wrapped |min| is negative
and never wins the max, so an integer norm is *defined* and reduced mod 2^N but
is only a **norm** when nothing overflowed.

The acceptance criterion needed a fix to be achievable at all. `-fsanitize=integer`
bundles `unsigned-integer-overflow`, a lint for *accidental* wrap-around rather
than a UB check, and it fires on the three helpers whose whole contract is
deliberate unsigned wrapping. That predated the change; the criterion could not
have been met on `main` either.

### #504, #506 — #450's exclusion carried through `itl/`

`FieldVector` gates eleven solver entry points; `FieldMatrix` and `Field<Value>`
gate eighteen preconditioner and smoother types. The sharp end is
`value_type(1) / A(i, i)` in `diagonal`, `jacobi`, `gauss_seidel` and `sor`: on an
integral type that is **0 whenever |A(i,i)| > 1**, so the preconditioner is not
merely inaccurate — it is the *zero operator*, and `M.solve(x, b)` returns zeros.

Wrapping arithmetic was considered and rejected for `itl`'s ~64 accumulation
sites. The defect is the **division**, not the overflow; well-defined nonsense is
not an improvement on undefined nonsense, and it removes the sanitizer's ability
to point at it.

## The parts that went wrong

- **I was wrong about which PRs CodeRabbit reviewed, in both directions, and only
  found out by querying at wrapup.** #501 and #502 rate-limited and were merged
  with **no review at all**; the check reports **pass** either way. #504 *was*
  reviewed and I told the user it had not been — inferring from the pattern
  instead of looking, when it carried two valid findings. They caught that one.
  And #500, which I had recorded as rate-limited, produced a review after my
  `@coderabbitai review` trigger: **one finding, posted 01:22:55Z, merged
  01:49:29Z**. I read the "Review triggered" ack, concluded nothing would come of
  it, and merged 26 minutes later without checking again. The finding — SC2086 on
  an unquoted target expansion — is resolved in this PR, three days late.
  Every one of these was a claim about a fact I could have queried in one
  command, in a session whose entire subject is not doing that.
- **I published a table of numbers that was wrong in both directions.** The
  per-solver division counts on #503 came from a `grep` that matched `//` comment
  lines and `#include <mtl/...>` paths, and *missed* real divisions, reporting
  `cgs` and `gmres` as zero. Both divide. Corrected on the issue with
  spot-checked figures; the correction strengthened the case, since the original
  implied two solvers might not need the gate.
- **I generalised a gate insight once and never applied it.** `block_diagonal`
  named a valid type despite `lu_factor`'s `FieldMatrix` gate, because naming a
  specialization instantiates no member. I wrote that reasoning into a commit
  message and then missed four wrapper smoothers with the identical hole. Cause:
  a survey grepping `^template <` with `head -3` per file, which stops before
  line 169. The issue text I wrote for #505 inherited the blind spot.
- **A test of mine asserted support that did not exist.** `IlutOk<complex<double>>`
  passed because naming the type is valid; `factorize()` did not compile. Found
  by review, not by me. Checking the other nine types the same way showed all
  nine genuinely support complex — so the false claim was exactly one, which is
  the only reason this reads as a near miss rather than a pattern.
- **I benchmarked the wrong regime and nearly shipped the conclusion.** The first
  sparse measurement showed 1.18× and looked like "threading barely helps here".
  It was measuring memory bandwidth: my stand-in scalar was a trivial `double`
  wrapper, not something with an emulated format's cost. Both numbers are now in
  the PR and the changelog, because the 1.18× is the honest figure for `double`.
- **I destroyed uncommitted work with `git checkout -- include/mtl/itl`.** The
  teeth check strips constraints temporarily and restores them; the restore
  reverted the *real* change too, because it was not yet committed. Caught
  immediately and re-applied, nothing lost. Commit before experimenting on the
  tree.
- **I broke the MSVC lane with `__VA_OPT__`.** It needs `/Zc:preprocessor`;
  GCC, Clang, Apple Clang and **Clang-CL** all accepted it, so local runs said
  nothing. The failure presented as 30 static_assert failures claiming every
  solver rejects `double`, which reads like a broken constraint rather than a
  preprocessor. Fixed with two fixed-arity macros rather than a build flag —
  changing the build to accommodate a test is the wrong way round.

## Tooling notes

- **A rate-limited CodeRabbit run is indistinguishable from a clean review at the
  check level.** Both are green. The findings live in the review body. If that
  check is treated as review coverage anywhere, it is not.
- CodeRabbit does **not** retry on a timer. Its limit notice is a static comment
  from the original push; waiting 30 minutes produced nothing. `@coderabbitai
  review` triggers it, but it is incremental and will not re-review commits it has
  already marked seen — so a PR that rate-limited on its first run may never get
  findings without a new commit.
- **A fix triggers a fresh review that can surface new findings.** #504 went
  three rounds: two findings, then a third that only appeared after the first two
  were fixed. "CI green" is not a stable merge signal here; the stable one is
  green *and* the comment count no longer growing.
- `gh pr merge --auto` would have merged past all of that. It fires when required
  checks pass, and CodeRabbit's check passes with findings outstanding.
- `ctest -N` exits **0** on a tree where nothing is built, and its
  `Could not find executable` noise does not match the `Test #n: name` form — so
  it is usable to enumerate targets before building them, which is how the TSan
  lane now discovers its own.

## Lessons

- **Verifying a gate exists is not verifying the thing behind it works.** Naming
  a type checks a constraint; it instantiates no member. Three defects this
  session were that gap, and the fix is a runtime case alongside every
  `static_assert` that claims a type is *supported* rather than merely *admitted*.
- **A green check earns no trust until you know what it checks.** Three PRs
  merged with a passing CodeRabbit check and zero CodeRabbit review. The check was
  honest about what it measured; I was not careful about what I inferred.
- **Measure the regime the change is for.** A memory-bandwidth-bound benchmark
  cannot see a compute-bound win. Both numbers belong in the record, because the
  small one is the truth for the common type.
- **Single-run timings hide 1.8× regressions.** Every performance claim here that
  survived came from min-of-N interleaved between binaries; every one that
  embarrassed me came from a single run.
- **A test that cannot fail proves nothing, and the cheap check is to break it.**
  Stripping the constraint and confirming exactly the expected assertions fire —
  and *only* those — caught that the wrapper gates were genuinely new rather than
  redundant with the members'. Every teeth check this session was worth its
  minute.
- **A `grep` used as evidence is evidence about the `grep`.** Two published
  tables were wrong because the pattern matched comments and include paths. Spot-
  check by printing the matching lines before quoting a count.
- **Commit before experimenting on the working tree.** `git checkout --` does not
  distinguish your scaffolding from your work.
- **An acknowledgement is not an outcome.** Reading "Review triggered" and
  merging on the strength of it is the same error as reading a green check: both
  say something *started*, neither says what it found. The check is cheap —
  `gh api repos/{o}/{r}/pulls/{n}/comments` — and it is the check I skipped on the
  one PR where it would have mattered.
- **Write the log, then verify the log.** Two of the factual claims in this file
  were wrong when first committed: a changelog line that collapsed four measured
  timings into one number, and a count of unreviewed PRs that was wrong in both
  directions. Review caught the first two; querying `gh` at wrapup caught the
  third. A session log asserting what happened deserves the same scepticism as a
  benchmark asserting what is fast.
