# Session: complex Hermitian factorizations, silent acceptance, and a measurement that said no

**Date**: 2026-08-03
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Continue clearing the correctness backlog `mtl5-python` filed while binding the
dense surface, and take the next performance issue.

What connected the day was not the subject matter. It was a single failure
shape, met five times in five different disguises:

> **Something reports success while doing the wrong thing — and the test that
> should have caught it passes, because it is structurally aimed elsewhere.**

Four of those were in the library. One was in the guard written to prevent it.

## Work Completed

### GEMM parallel efficiency: 79% → 91% (#348)

The blocked GEMM's shared B panel was packed by each jc-team's **leader** while
the rest of the team sat on the barrier — a serial region proportional to
`kc*nc` per `pc` step. At N=2048 with one team of 8, thread 0 packed ~67 MB
while seven cores idled.

Packing is pure data movement into computable disjoint offsets, so the team now
splits the NR-column panels among its members. The packed bytes are identical
for any split, so the result is unchanged. Parallel efficiency on 8 P-cores went
**79% → 91%**, and the scaling data and results page were refreshed (#350).

### The Platform Performance Engineering lab (`ppe/`, #354)

A progression of GEMM implementations (naive → loop-order → blocking+packing →
register tiling → compile-time microtile) measured across `int8/16/32/64` and
`fp16/32/64`, with each step's hypotheses written down *before* the experiment
that tests them.

The most useful result is the step that goes **backwards**. `v5_micro` — the
standard "next" optimization, a microtile with compile-time `MR`/`NR` and no
edge tests — is **5× slower** than a plain loop reorder for `fp32`. The compiler
says why: full unrolling replaces the vectorizable inner loop with 32 scalar
operations, GCC then reports the nest as unvectorizable, and 32 scalar
accumulators spill (`+512` bytes of stack). The penalty is proportional to how
much the type gains from SIMD, so `v5` is simultaneously the *best* kernel for
`int32` and catastrophic for the floats.

`MR × NR` is not a free parameter: it is bounded by the register file, and which
bound binds depends on a compiler decision that source-level "optimization" can
silently flip.

### `ldlt` returned a wrong answer for Hermitian input (#352 → #356)

`ldlt_factor` computes `A = L·D·Lᵀ`. Fed a Hermitian matrix it ran to completion
and returned a plausible wrong answer under `info == 0`. Added `ldlt_h_factor` /
`ldlt_h_solve` for `A = L·D·Lᴴ`, and a guard refusing the mismatched input.

**Review found two more defects in that fix**, both real:

1. The Hermitian guard used **exact** equality, so it only caught matrices that
   are bit-exactly Hermitian. A **one-ULP** perturbation slipped past it and
   produced an answer wrong in the first significant digit — the exact bug the
   guard existed to prevent. Worth recording *why* it hides: a naive triple-loop
   `Bᴴ·B` **does** come out bit-exact Hermitian, so the toy case passes; a
   blocked GEMM or FEM assembly does not. The exact test works on precisely the
   population a test author writes and fails on the one a user supplies.
2. `ldlt_h_factor` had the **mirror hole** in the same commit: it took `real()`
   of the diagonal unconditionally, so complex-symmetric input silently lost its
   imaginary diagonal, also under `info == 0`.

Both guards are now scale-relative (`n · eps · scale`, using LAPACK's `CABS1` to
avoid an O(n²) sweep of `hypot`), degrading to an exact test for types without a
`numeric_limits` specialization rather than to an arbitrary constant.

### Sparse containers accepted an orientation they ignored (#355 → #358)

`compressed2D` took a `Parameters` bundle whose orientation could be
`col_major`, and **never read it**. A `col_major` instance was byte-for-byte a
CSR matrix, so a caller who populated it from genuine CSC arrays got the
**transpose** while the constructor reported success.

Resolution was the issue's *smallest* option — a `static_assert` — not its
"complete" one. The reason is concrete: **34 headers across six subsystems read
the raw arrays as CSR.** Real CSC support means generalizing every one of them,
each a place to reintroduce the same silent wrong answer. That is a feature, not
a bug fix. The guard costs nothing: nothing in the tree instantiates a
`col_major` `compressed2D`, and nothing could have, because no such matrix was
ever correct.

`ell_matrix` had the same inert-orientation hole and was folded in on request,
with its severity stated honestly: its only fill path is the `compressed2D`
constructor, which the first commit already closed, so it was
constructible-but-unfillable rather than wrong. `coordinate2D` was left alone —
it stores explicit `(row, col, value)` triplets, so orientation genuinely cannot
change a result.

Two docs overclaims surfaced and were corrected: a "zero-cost CSC view over
`compressed2D` with col-major parameters" that was never implemented, and a
"Full support (CSR, CSC, COO, ELLPACK)" comparison row. Both pointed readers at
the construct now rejected.

### A new kind of test: compile-failure with an expected diagnostic

`#358` added `tests/unit/compile_fail/`. Sources that must **not** compile, built
on demand by ctest (`EXCLUDE_FROM_ALL`), matched against an `// EXPECT-ERROR:`
regex declared in each source.

It deliberately does **not** use CMake's `WILL_FAIL`. `WILL_FAIL` passes whenever
the build returns non-zero, so a typo in the test source would make it pass green
while proving nothing. The `cholesky` case proved the point within hours: with
its guard removed the file *still* fails to compile, via the old `operator<=`
error — `WILL_FAIL` would have been green, while the regex caught the regression
to the worse diagnostic.

### #110 — measured, and closed without a code change

The issue asked to decouple GEMM parallel granularity from the `mc` cache block,
offering three implementation options. Before building any of them, an `mc`
sweep (varying only `l2_bytes`, so `mr/nr/kc/nc` stay fixed):

| mc | nib | balance at T=8 | T=1 | T=8 | efficiency |
|---:|---:|---|---:|---:|---:|
| 32 | 64 | balanced | 57.32 | 412.33 | 89.9% |
| **64** | 32 | balanced | **56.81** | **414.45** | **91.2%** |
| 128 | 16 | balanced | 54.84 | 385.88 | 88.0% |
| 192 | 11 | **imbalanced** | 53.71 | 266.70 | 62.1% |
| 256 | 8 | balanced | 47.89 | 330.92 | 86.4% |
| 320 | 7 | **1 thread idle** | 48.80 | 275.43 | 70.6% |

Two findings, both against the plan:

- **No `mc` above the shipping 64 is better at `T=1`.** The premise — that `mc`
  wants to grow for cache reasons and parallelism starves it — is false here.
  Throughput falls from 57.32 at `mc=32` to 47.89 at `mc=256`, rebounds slightly
  to 48.80 at `mc=320`, and is never within 13% of `mc=64` again. So growing
  `mc` is worse with no threading involved at all, and the region all three
  options would unlock is the region that measures badly.
  (An earlier draft of this log called the `T=1` column *monotonic*; it is not —
  the last point rises. The conclusion is unchanged, since every `mc > 64` is
  worse than `mc=64` either way, but the stronger word was not the measured one.)
- **The coupling is real but harmless.** The two imbalanced points collapse
  exactly as the issue predicts, while every balanced point sits at 86–91%. The
  mechanism was right; the inference that it cost anything was not.

The acceptance bar (≥85%) had already been met by #348 the same day. Closed.
**Had the issue's plan been implemented as written, it would have been days of
work for a measured zero.**

### Complex `cholesky` and `qr` (#353 → #360, #361)

Neither compiled for `std::complex`, failing first on relational operators
applied to a complex where a *magnitude* was meant. The issue's own framing
drove both fixes:

> patching only the comparisons would produce something that compiles and
> computes the wrong factorization — the same failure mode as #352

So neither PR repairs a comparison.

**#360** restricts `cholesky_factor` to real element types (there is no
complex-symmetric variant to offer — "positive definite" is a statement about an
*ordering*, and a complex symmetric matrix has no real diagonal to order) and
adds `cholesky_h_factor` / `cholesky_h_solve` for `A = L·Lᴴ`, accumulating the
pivot in the **magnitude** type so the positivity test is well-formed.

**#361** rewrites the reflector as `H = I − τ·v·vᴴ` with `H·x = β·e₁`, β real
(LAPACK `zlarfg`). `H` is unitary but **not Hermitian** for complex τ, so
`H⁻¹ = Hᴴ` — one fact that lands differently in every consumer: `conj(tau)` in
`qr_extract_Q`, plain `tau` in `lq_extract_Q`, conjugated row plus `Hᴴ` on the LQ
factor side.

Three things this turned up:

- **LQ is not the transpose of QR.** `householder()` annihilates a *column*; LQ
  annihilates a *row*. Getting half of it right produced a **perfectly unitary
  Q** (`‖QᴴQ−I‖ = 5.6e-16`) while `L·Q` differed from `A` by **O(1)**. A
  unitarity check alone passes it, which is why the tests assert reconstruction.
- **A `sigma == 0` shortcut broke the real-β contract** (found in review): a
  vanishing tail with a complex leading entry returned `tau = 0`, leaving a
  complex `R` diagonal. Silent — Q stayed unitary, `Q·R` still reproduced `A`.
- **The fix nearly opened wrong-answer paths in two other files.** Making
  `householder()` complex-capable stripped several incidental errors that were
  holding `hessenberg_factor` and `eigen_symmetric` shut. Both apply `H·A·H`, a
  similarity only because real reflectors are Hermitian. Both now reject complex
  explicitly.

## The mistake worth recording

In #361 the complex guard went into **`eigenvalue_symmetric_generic`, which does
not call `householder` at all.** The function performing the `H·A·H` reduction is
`eigen_symmetric`, and it was left open.

Worse: the compile-failure test targeted the guarded-but-irrelevant function,
matched its regex, and **passed while covering nothing of the hazard it was
written to close.**

That is precisely the failure mode the compile-failure harness had been
introduced for the same week — cited twice earlier in the day as the answer to
this exact problem. The lesson is not "write compile-failure tests". It is:

> A negative control on the *guard* proves nothing unless you have also checked
> the guard is on the *function that matters*. Enumerate entry points against the
> hazard site before writing the test, not after.

Caught in review. The relocated guard now fails only its own test when removed,
which the previous version could not do no matter what was broken.

## Process notes

- **A comment silently failed to post.** `gh pr comment` was passed a `-q` flag
  it does not accept; the error vanished into a pipe. Only caught by counting
  comments afterward. Verify the post, not the exit path.
- **A near-duplicate issue was filed** (#357) without searching first; #353
  already covered it, and more broadly. Consolidated and closed.
- **A prediction was published and had to be withdrawn**: #357 asserted that
  `cholesky` mishandled Hermitian input the way `ldlt` did. It does not — it
  fails to compile, which is the loud and safe outcome. Corrected in the PR that
  carried the claim.

## Results

| | |
|---|---|
| PRs merged | #348, #350, #354, #356, #358, #360, #361 |
| Issues closed | #352, #355, #110, #353 (+ #357 consolidated) |
| Issues filed | #362 (Hermitian eigenproblem) |
| GEMM parallel efficiency | 79% → **91%** on 8 P-cores |
| New CI capability | compile-failure tests with expected-diagnostic matching |

Every real path touched today is **bit-identical** to before — verified by
hex-float digest against builds of the pre-change code, not by tolerance. That
is why the real branch of `householder()` is kept verbatim inside an
`if constexpr`, and why the `_h` agreement tests use `==`.

## Follow-ups

- **#362** — Hermitian eigenproblem: unitary reduction `A → H·A·Hᴴ` with a real
  subdiagonal and phase accumulation. The guards make its absence loud.
- **#347** — `nc` derivation sizes the packed-B panel to 100% of a *shared* L3.
- **#351** — single-core GEMM sustains ~72–78% of FMA peak.
- `mtl5-python` can drop its `TypeError` for complex `cholesky`/`qr`/`lq`.
