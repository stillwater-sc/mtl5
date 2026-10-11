# Session: true under the flags it was checked with

**Date**: 2026-10-10 – 2026-10-11
**Duration**: Evening into the next day, about 15 hours wall-clock with an overnight gap
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

This picks up where the [2026-10-10 log](2026-10-10-every-test-overrode-the-default.md)
stopped. That session left one follow-up: the eigensolver accumulator tests couldn't
tell a `float` accumulator from a `double` one. From there:

- the follow-up was fixed;
- a scalar-operator bug a Kalman filter over posits had hit (#538) was resolved;
- **5.13.2** was cut, and mtl5-python moved its pin to it;
- PR #552, opened from the cortex port work in a parallel session, got a performance analysis page. Writing it turned up two CI failures, both fixed before merge.

**Merged:**
- mtl5 #547, #548, #551 (5.13.2) and #552;
- mtl5-python #98.

**Closed:** #538, #549, #550.

## The through-line

Each claim checked today was true in the configuration it had been checked in, and
something outside that configuration was where it stopped being true.

| Claim | Where it was true | Where it wasn't |
|---|---|---|
| "The eigensolvers honour the accumulator" (the tests before #547) | Any accumulator that reaches the 1e-4 tolerance floor | A solver that silently ignored the accumulator also passed. The fixture couldn't separate the two policies |
| "The scalar operators take any scalar" (before #548) | `double`, `float`, `int`: everything `std::is_arithmetic` | Every type MTL5 exists for: posits, `cfloat`, LNS, fixed-point and `std::complex`. `K = P H^T / S` didn't compile |
| "i-k-j is bit-identical to i-j-k" (#552 as opened) | Baseline x86-64, where there is no FMA to contract into | `-march=x86-64-v3` and AArch64: the new loop fuses every multiply-add, the old one only some |
| "The CRS branch returns early" (#552 as opened) | Builds without BLAS | With `MTL5_HAS_BLAS`, the code after an `if constexpr` return is still instantiated, and it named `M::orientation` on types that have none |
| "`mult()` on `dense2D<double>` runs the generic kernel" (#552's test) | Builds without BLAS or `MTL5_NATIVE_FAST_GEMM` | Under BLAS, `?gemm` runs instead and sums in its own order. A compile error had been hiding this one |
| "The hand-written control loop costs 75 µs" (first A/B run) | Most processes | About one process in twelve ran it ~20 % slower for its whole lifetime, in either binary |

The fixes went the same way each time. Either the claim was narrowed to the
configuration that holds it (`-ffp-contract=off`, the BLAS guard), or the test was
moved into the configuration that breaks it (the cancelling fixture, the class-typed
scalar). Two of the six were caught by CI lanes built for exactly that purpose: the
FP-contract lane and the LAPACK lanes. They were red when #552 was handed over for a
performance page, and nobody had looked yet.

## Work completed

### #547: eigensolver tests that can tell the policies apart

The 2026-10-10 log recorded the gap: the Lanczos test #540 restored, and the Arnoldi
test #529 rewrote it into, only asked whether `float` and `double` accumulation both
reach 1e-4. They do. So a solver that ignored its accumulator would have passed.

Each solver now also has a test on a 1024×1024 diagonal: `2^20`, then 1022 copies of
`2^-5`, then `-2^20`. Its Rayleigh quotient cancels. A `float` accumulator absorbs the
small terms into the large ones and gets **exactly 0**. A `double` accumulator gets
the exact value, `(1022 · 2^-5) / 1024`. `max_iter = 1` (`subspace = 1` for the Krylov
solvers) means the check is about accumulation, not convergence. Arnoldi comes out
0.2 % high because of its second Gram–Schmidt pass, so it gets a 1 % tolerance, and
the test says why.

Teeth check: with each solver patched to ignore the policy, only the new
`double`-accumulator assertions fail. The original tests were kept and renamed to say
what they check, which is that both policies converge.

The first commit was rate-limited by CodeRabbit. A full review would have cost
money on top of the plan (up to $0.25 for one file), and that's the user's decision,
not mine to make. The user said merge, and #547 went in with the second commit
reviewed ("no actionable comments") and the first possibly not.

### #548: scalar operators for every `Scalar`, not just `is_arithmetic` (#538)

`scalar * matrix`, `matrix * scalar`, `matrix / scalar` and the three vector forms
were gated on `std::is_arithmetic_v<S>`. That excludes every number type the library
is built for, even though they all satisfy `mtl::Scalar` and `mtl::Field`. It showed
up in a Kalman filter over posits: `K = P H^T / S` didn't compile.

The obvious fix, `Scalar<S>`, doesn't work. For a matrix operand, checking
`Scalar<dense2D<T>>` asks whether `dense2D<T> * dense2D<T>` is valid, which recurses
into these same operators. GCC reports "satisfaction of constraint depends on itself"
13 times. The gate is `(!Collection<S> && Scalar<S>)`: `&&` in a constraint
short-circuits, so `Scalar` is never asked about a container. The test uses a
`real_wrapper` class type as a stand-in for a posit, since MTL5 stays Universal-free.
It covers `std::complex`, and a case where matrix × matrix, matrix × vector and
`2 * A` still resolve as before. Against main's headers, all 12 scalar-operator lines
in the test fail to compile.

CodeRabbit was rate-limited and **didn't review #548 at all**.

### 5.13.2

#551. The range `v5.13.1..main` held #546, #547 and #548. The CHANGELOG was written
from that range, and each PR has a home in the section. The tag went on the merge
commit `083c50f`, and the GitHub Release published at 04:36 UTC.

mtl5-python #98 moved the FetchContent pin to v5.13.2. It was tested before the tag
existed, by pointing `FETCHCONTENT_SOURCE_DIR_MTL5` at the release branch: 1408
passed, 4 skipped, the same as on 5.13.1. The `find_package` floor stays at 5.13.1
because nothing in the bindings needs 5.13.2. The package version stays 5.13.0: the
minor tracks MTL5 and `chore(deps)` gives semantic-release nothing to bump.

### #552: the performance page, and what writing it found

#552 makes two speed changes to `mtl::mult`. #549 runs the generic dense kernel as an
i-k-j loop when B and C are row-major. #550 adds a CRS × dense matrix product. The
request was a page that describes the problem, uses non-zero structure images to show
the opportunity, explains the restructuring, and measures it. It's
[docs/performance/ikj-gemm-and-crs-dense.md](../performance/ikj-gemm-and-crs-dense.md).

**Images.** `mtl::io::spy` draws one pixel per element, so an 81×81 filter operand
comes out 81 pixels wide. The page's figures come from a small program,
`benchmarks/fig_mult_ikj.cpp`, using the same PNG writer with a 6-pixel cell:

- F = Φ ⊕ I: 4.4 % non-zero;
- the measurement Jacobian H;
- the Joseph factor I − KH: 27–100 % non-zero, depending on how many clones observe the track;
- two figures of B's memory as the i-j-k and i-k-j inner loops read it.

The cortex source was not read: a permission check denied it. The operands are
modelled from the dimensions in the PR, and the page says so.

**Measurement.** One source, `benchmarks/bench_mult_ikj.cpp`, built against main's
headers and against the PR's. Six interleaved rounds per binary, pinned to a P-core,
single-threaded. Two hand-written loops, compiled into both binaries, served as
controls.

| result | before | after |
|---|---|---|
| P·P, 81×81, baseline | 137 µs | 76 µs (1.81×) |
| P·P, 81×81, AVX2 | 141 µs | 53 µs (2.65×) |
| square sweep, n = 8–256 | | 1.07–3.3× (up to 7× with AVX2) |
| F·P | 635 µs | 3.6 µs |

F·P after the change is 3.8× faster than the column-by-column matvec, the only sparse
route before. The CRS product's time is proportional to nnz, and on a fully dense
operand it costs the same as the dense product. The cost still left is building the
CRS operand: 13 µs for F, more than the product it enables.

**The findings:**

1. **Contraction.** The `-march=native` A/B's `#check` lines disagreed between builds.
   A four-way experiment settled why. Without FMA, or with `-ffp-contract=off`, both
   kernels equal an unfused reference. With `-march=native`, the new kernel equals a
   fully fused reference, and the old one equals *neither*: it fuses some terms and
   not others. #552's CI was already red on the `FP-contract (x86-64-v3)` and
   `ARM64 (Clang 18)` lanes for this reason. Fix: `op_test_mult_ikj` joins the
   threaded bit-identity tests on `-ffp-contract=off` in `tests/unit/CMakeLists.txt`,
   and the CHANGELOG states the condition.
2. **The BLAS gate.** The LAPACK lanes failed to compile. `mult()`'s BLAS condition
   was one conjunction, `BlasDenseMatrix<...> && is_row_major_v<MA> == ...`. A
   conjunction in a constant expression instantiates every variable template it
   names, even after a false operand. `is_row_major_v` needs `M::orientation`, which
   `compressed2D` and `transposed_view` lack. The `transposed_view` case already
   existed before #552; the PR's test was the first to reach it under BLAS. Fix: two
   nested `if constexpr`.
3. **Behind the compile error, a test assumption.** With the build fixed, one check
   still failed under BLAS: it assumed `mult()` on `dense2D<double>` reaches the
   generic kernel. Under BLAS it reaches OpenBLAS. That check now runs only when
   neither `MTL5_HAS_BLAS` nor `MTL5_NATIVE_FAST_GEMM` is defined.

Verified locally:
- full suite: default GCC, GCC `-march=x86-64-v3`, BLAS + LAPACK;
- `op_test_mult_ikj` alone: Clang `-march=x86-64-v3`, `MTL5_NATIVE_FAST_GEMM=ON`.

CI then passed 46 checks. CodeRabbit reviewed the final head: "No actionable comments".

## The parts that went wrong

- **#552 was red when the work on it started, and I didn't check.** The request was a
  page, and I went straight to benchmarks. I found the failing lanes only after
  measuring, when the `-march=native` checks disagreed. Four failing jobs were one
  `gh pr checks 552` away from the start. Both failures were real defects in the PR,
  and an analysis page that repeated the "bit-identical" claim would have documented
  a statement CI was already refuting.

- **My first A/B was noisy, and I spent time on the wrong hypothesis.** The hand loop
  read 75 µs in some processes and 91 µs in others, so the control moved between
  builds. I tested 4K aliasing first: 64 offsets of C relative to B, all 75 µs. Then
  buffer placement within a process: also flat. The effect turned out to be per
  process (one run in twelve of an identical binary). That's the case the repo's run
  contract already handles with interleaving and min-of-N. More rounds would have
  settled it sooner than a theory about it did.

- **Two PRs merged without a complete review.** #548 had no CodeRabbit review at all,
  and #547's first commit may not have been reviewed. Both ran into the included
  review limit. This is the second session running where review coverage, not review
  findings, was the gap (2026-10-10 had #541, #543 and #545 unreviewed). #552 and #551 did get reviewed.

- **The branch moved under me.** Between pushing the page and starting the CI fixes,
  main was merged into `perf/ikj-generic-gemm` from GitHub. A `git fetch` before
  editing caught it, and it fast-forwarded. A push without that fetch would have been
  rejected, not lost, but it's the right reflex on a branch that more than one
  session touches.

- **Smaller ones.**
  - A `grep -c` used to check a CHANGELOG edit reported 0 because `^` in mid-pattern
    is an anchor. That aborted an `&&` chain, so the commit didn't happen until the
    next step.
  - Importing `mtl5` from the mtl5-python repo root picked up the source tree's stale
    `_core` instead of the installed one; running pytest from outside the repo fixed
    it.

## Tooling notes

- **`spy` is one pixel per element.** It's right for large matrices, where the point
  is downsampling. For a figure of a small matrix, draw cells with
  `mtl::io::write_png_rgb` directly.
- **`gcc -fopt-info-vec-optimized`** settles "does this loop vectorize?" faster than a
  benchmark. The old i-j-k k-loop is reported as vectorized, but it's an in-order
  reduction. The i-k-j loops are reported vectorized and versioned on a runtime alias
  check.
- **`-Q --help=optimizers`** prints the effective defaults. It showed
  `-ffp-contract=fast` for GCC 15 even under `-std=c++20`.
- **`if constexpr (...) { ...; return; }` doesn't discard what follows.** Only an
  `else` branch is discarded. Code after an early return is instantiated for every
  type that reaches the function.
- **`gh pr merge --delete-branch` and a worktree.** When another worktree has the
  branch checked out, merging without `--delete-branch` and then deleting the remote
  ref with `gh api -X DELETE .../git/refs/heads/<branch>` leaves that worktree intact.

## Lessons

- **State a numerical-identity claim together with its flags.** "Bit-identical" is a
  claim about the compiler as much as about the loop order. The repo already knew
  this (#381's tests were on `-ffp-contract=off` with a paragraph explaining why),
  and a new test making the same claim needed the same treatment. Where a lane
  exists to check a claim, the claim should name the lane's configuration.
- **Check CI before building on a PR.** A request to document a change is also a
  request to document what the change does, and a red lane is information about
  that.
- **A compile error can hide a runtime failure behind it.** After fixing a build, run
  the test in that configuration before calling the lane fixed. The BLAS guard was
  the second of two problems on the same lane.
- **Controls make a benchmark readable.** Two loops whose code was identical in both
  binaries showed at once that the first run's difference wasn't the change. Without
  them, 91 µs would have been reported as the "after".
- **Let a test reach for the type that breaks the assumption.** That's the same
  lesson as 2026-10-10's, from the other side. The cancelling diagonal breaks
  "`float` is accurate enough", and the class-typed scalar breaks "a scalar is
  arithmetic".
