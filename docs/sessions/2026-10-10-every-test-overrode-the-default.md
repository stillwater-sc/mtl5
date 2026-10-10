# Session: every test overrode the default

**Date**: 2026-10-09 – 2026-10-10
**Duration**: Two-day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Started as a design question. CodeRabbit objected that adding a leading
`Accumulator` template parameter broke callers who spelled template arguments
explicitly: was `template <typename Accumulator = void, Matrix M, Vector V>` better
or worse than putting it last? The session grew from there into one calling
convention across the whole library, two MTL5 releases (**5.13.0** and **5.13.1**),
and one mtl5-python release (**5.13.0**, on PyPI).

Merged: mtl5 #539, #540, #541, #542, #543, #544, #545; mp-iterative #44; mp-spice
#18; mtl5-python #97. Closed #521 and #527. Status comments on #535 and #536.
Part C of #261 rewritten.

## The through-line

The accumulator policy's test suite was large, green, and blind in exactly one
place: the **default**. Every test that exercised an accumulator supplied its own
— `counting_acc`, `wide_acc`, `long double` — and every one of those either
specialized `accumulator_traits` or was a built-in type whose value-initialization
happens to be zero. So no test ever ran the primary template's `clear()` on a type
where `Acc{}` and zero differ. That was the bug that shipped in 5.13.0.

The same shape turned up four more times in two days. Each time a test passed
because it wasn't looking at the thing in question.

| What passed | What it was actually checking |
|---|---|
| The accumulator suite, through 5.13.0 | Custom accumulators that each override `clear()`. The primary template's `a = Acc{}` was never exercised on a type where `Acc{}` ≠ 0, and Universal's `lns{}` is **1.0** |
| `ctest`: 193/193 after #529 | That the eigensolver *file* passed. #529 had rewritten the Lanczos accumulator test in place into the Arnoldi one, and the count didn't move because the whole file is one ctest entry |
| `ctest`: 100% after a rebuild | Stale binaries. One target failed to compile (glibc 2.43's `uabs`) and `ctest` ran the previous build of it |
| The restored Lanczos test | That `float` and `double` accumulation both reach a 1e-4 floor. It can't tell them apart, so a `lanczos<double>` that silently accumulated in `float` would pass. CodeRabbit said so on #540; nobody read it before merge |
| A Release workflow that succeeded on 8 platforms | That MTL5 compiles and its own tests pass. The LNS regression was found by a *downstream* suite, mtl5-python's, within an hour of the tag |

## Work completed

### The convention: `Accumulator` leads, and defaults to `void`

The answer to the design question is that leading is correct, and CodeRabbit was
right about the cost. `Accumulator` is the only template parameter that can't be
deduced, and explicit arguments bind left to right, so a trailing one is reachable
only by restating every deduced type before it:

```cpp
supernodal_lu_numeric<double, decltype(A)::param_type, float>(A, sym);  // before #542
supernodal_lu_numeric<float>(A, sym);                                   // after
```

It is the standard library's own rule: `bit_cast<To>`, `make_unique<T>`,
`ranges::to<C>`. Applied in three steps:

- **#539, Krylov.** All ten solvers had `Accumulator` sixth, after five deduced
  types. A test in the repo was already calling `qmr` with all six spelled out.
- **#540, eigensolvers.** `power_iteration`, `lanczos`, `arnoldi`. A leading
  parameter's default can't name a later one, so `Accumulator = T` became `= void`,
  resolved to `T` inside. That kept default calls byte-identical: they still
  accumulate through `dot<T, T>`, not `void`'s BLAS dispatch. Moving it also took
  away the only way to pass a `double` tolerance with a `float` vector, so `tol` is
  now `std::type_identity_t<T>`.
- **#542, sparse direct and Householder.** Seven sparse factorizations and the
  three Householder functions, which were the last ones left. A pre-release audit
  of every accumulator-aware API is what found them.

After #542, every accumulator-aware *function* template in MTL5 follows the
convention. Class templates (`pc::ssor`, the smoothers) keep it trailing, since
their arguments are always explicit. `*_solve_refined<FactorValue, Accumulator>`
keeps it second, because neither parameter can be deduced there.

**The breaking change that doesn't fail to compile.** `axpy`, `scale`, `ger`,
`symv`, `trmm`, `trsm`, `symm`, `syrk` and `syr2k` gained their leading
`Accumulator` in front of `Scalar S` earlier in the range (#511, #515, #522–#526).
A call like `axpy<float>(0.1, x, y)` still compiles, and now means "accumulate in
float" instead of "convert alpha to float". Every other break in the release fails
to compile. This one leads the 5.13.0 Breaking section.

### #540: a lost test, found by asking where it went

The user remembered a `TEST_CASE` that no longer existed. `git log -S` placed it:
added in #528, and gone one PR later. #529 hadn't deleted it. It had *edited it
into* the Arnoldi test, changing the name, matrix and calls in place. The commit
message said "1 new accuracy test", and the new test's comment opens "As with
lanczos…", pointing at a test that no longer existed. The user had saved a copy,
which went back in.

### #541: glibc 2.43 declares `uabs`

C2y adds `uabs` to `<stdlib.h>`, and glibc 2.43 exposes it under g++'s default
`_GNU_SOURCE`. The test's own `uabs` in an anonymous namespace became ambiguous.
Renaming it fixed the clash (now `abs_as_u32`, exact rather than "reduced mod
2^32" as its comment had claimed). Moving the helpers into `mtl::test` alone would
not have, because the file has `using namespace mtl;`.

### 5.13.0, and the version question

The release checklist worked as designed. The CHANGELOG was written from
`git log v5.12.0..main` (24 PRs), reconciled entry by entry, and the auto-generated
release notes listed 25 PRs: the same 24 plus the release PR itself.

5.12.0 had shipped `!` changes as MINOR on a narrow precedent: every break removed
an instantiation that already gave wrong answers. That doesn't hold here. These
breaks hit *correct* call sites, one of them silently, and three downstream repos
needed changes. That went to the user before the tag, since the tag is the step
that can't be undone. The user chose 5.13.0.

### 5.13.1: `clear()` set a value-initialized `Acc`, not zero

Bumping mtl5-python from v5.11.0 to v5.13.0 failed 8 of 1412 tests, all of them
QR/LQ on `lns16` and `lns32`. The numbers ruled out rounding: Q's orthogonality
error was **1.5**, and both widths failed by nearly the same amount, which points
at the algorithm rather than at precision.

The cause was one line. `accumulator_traits::clear(a) { a = Acc{}; }` assumes
value-initialization is zero. A logarithmic number system encodes 2⁰ in its
all-zero bit pattern, so `lns{}` is 1.0. A five-line program confirmed it:
`lns{}` → 1, `lns(0)` → 0. 5.12.0 had started `householder`'s sum of squares at
`mag_t(0)`. #537 routed it through `clear()` **on the default path**, so every LNS
column norm picked up +1.

The fix is `math::zero<Acc>()`, the library's own spelling of zero.
`fma_accumulator` had the same flaw twice, in its `clear()` and in its member
default `T sum{}`. An audit of about 40 accumulator declarations and workspaces
found every other one cleared or assigned before it is read, so `clear()` was the
single point of failure. The regression test uses `shifted_zero`, a mock whose
value-initialization is 1. MTL5 stays Universal-free, so it can't use `lns`
itself. **All four cases fail with the old `clear()` and pass with the new one.**

### Downstream

| Repo | PR | Change |
|---|---|---|
| mp-iterative | #44 | Ten `*_quire` tests and `cg_precision`: six-argument lists → `<Quire>` |
| mp-spice | #18 | `native_klu_factor` ×2, `sparse_lu_numeric` ×1 → `<Acc>` |
| mtl5-python | #97 | Pin v5.11.0 → **v5.13.1**, floor raised to 5.13.1, five sparse-direct call sites |

mtl5-python 5.13.0 was released with `gh release create --target` at the
verified merge commit. That published 13 files to PyPI. A fresh
`pip install mtl5==5.13.0` imports, and its 30 LNS factorization tests pass
against the published wheel.

The floor matters more than usual here. A build against MTL5 5.13.0 *compiles
cleanly* and returns wrong LNS results, so the `find_package` version is the only
thing keeping a system-installed 5.13.0 out. The CMake comment says so. And
because mtl5-python was pinned to 5.13.1, no PyPI user ever got the broken QR.

## The parts that went wrong

- **5.13.0 shipped a regression that MTL5's own suite couldn't see.** It
  passed all 46 PR checks and the Release workflow's 8-platform run.
  mtl5-python's suite caught it within the hour: the fix PR opened 62 minutes
  after the tag, and 5.13.1 shipped 97 minutes after 5.13.0. The blind spot came from how the tests were built:
  every test accumulator overrode exactly the function that was broken. The
  lesson from 2026-08-27 was that checking a gate exists isn't checking the thing
  behind it. This is the same lesson one level down: a test with a custom
  accumulator checks the *plumbing*, not the default the plumbing falls back to.

- **I stated something about downstream repos that I hadn't checked.** The
  pre-release audit said nothing in the peer repos called the sparse
  factorizations with explicit template arguments. mp-spice did, in three places,
  and mtl5-python in five. I found them a few minutes later, while preparing the
  PR, and corrected it in the next message. #540's equivalent claim had been
  checked by `grep` and was true. That's the difference: one claim was evidence,
  the other was an assumption written up as a finding.

- **mp-iterative's `main` was broken for about 19½ hours.** #539 merged on
  2026-10-09 at 22:09 UTC, and the companion #44 at 17:46 the next day. The #44
  description said to merge it immediately after #539. Nothing enforced that,
  and I didn't check the order before #539 went in. mp-spice's window was three
  minutes (#542 at 18:18, #18 at 18:21), because there the order was handled as
  one task.

- **A CodeRabbit finding went unread before merge.** On #540: *"Make this test
  detect a Lanczos accumulator regression."* It's correct. The restored test only
  checks that both configurations reach 1e-4, which the test's own comment admits
  can't distinguish them. The Arnoldi test it was rewritten from has the same
  shape. This is the second session running where a review finding surfaced at wrapup
  rather than before merge (2026-08-27 was the first). #541, #543 and #545 were rate-limited and got
  **no review at all**, and the check is green either way.

- **`ctest` reported 100% on a tree where one target hadn't compiled.** The glibc
  `uabs` failure stopped one target from building, and `ctest` ran its *previous*
  binary. I caught it because a test binary's timestamp predated the edit, not
  because anything failed.

- **I mistook a slow test for a hung one.** A Debug-build eigensolver run takes
  about a minute. Two invocations with output piped through `tail` printed
  nothing, and I started looking for a hang before checking how long a run
  takes. `$TMPDIR` was also unset in the tool shell, so `> $TMPDIR/x.txt` wrote
  to `/x.txt` and failed with permission denied.

## Tooling notes

- **`ctest` runs whatever binary is there.** A failed rebuild of one target
  leaves the old executable in place, and `ctest` reports on it. Check the build
  for errors before reading the test summary, or build with `-- -k` and look for
  `error:` explicitly.
- **A test that is one ctest entry hides its test-case count.**
  `test_eigensolvers.cpp` is one entry, so replacing a `TEST_CASE` with another
  leaves "193/193" unchanged. Catch2's own summary (`9 test cases`) is the
  number that would have moved.
- **`gh release create --target <sha>`** is what makes a hand-cut release safe.
  Without `--target`, `gh` tags the tip of `main` at that moment, not the commit
  CI went green on.
- **A `chore(deps)` commit is a no-op for semantic-release, and that's the
  point.** The automatic `wheels.yml` run on the merge did nothing (build and
  publish jobs skipped), which is what lets the hand-set minor version survive.
- **`FETCHCONTENT_SOURCE_DIR_<NAME>`** points a downstream build at a local
  checkout without touching its `CMakeLists.txt`. It's how mp-iterative,
  mp-spice and mtl5-python were all tested against unmerged MTL5 branches.

## Lessons

- **Test the default with a type that breaks the default's assumptions.** A
  default that is correct for `double`, `float` and `int` is not therefore
  correct. `shifted_zero` costs ten lines. Every test accumulator in the repo
  overrode `clear()`, so none of them could have found this.
- **A downstream suite is a test suite.** mtl5-python's 1412 tests ran types
  through MTL5 that MTL5's own suite never does. Bumping a downstream pin before
  calling a release finished, not after, would have caught 5.13.0's bug before
  the tag.
- **"Nothing downstream calls this" is a claim, so query it.** One `grep` across
  sibling repos, run before writing the sentence.
- **Merge order across repos is part of the change.** A companion PR that says
  "merge immediately after" has no mechanism behind it. Merging the pair as one
  task gave a three-minute window; leaving it to a description gave nineteen
  hours.
- **A test that passes for both policies doesn't test the policy.** An accuracy
  test whose fixture can't separate `float` from `double` accumulation proves the
  code runs, not that it honours the accumulator. It needs a fixture where the
  two differ, and an assertion that they do.
- **Read the review before merge, every time.** Two sessions running, a
  CodeRabbit finding has turned up at wrapup instead. It's one `gh api` call.
