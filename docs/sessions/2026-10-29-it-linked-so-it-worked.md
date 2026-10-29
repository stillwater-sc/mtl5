# Session: it linked, so it worked

**Date**: 2026-10-25 to 2026-10-29
**Duration**: A multi-day working session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Give MTL5 the matrix functions that a Lie-group or filtering layer needs,
generic in the element type: start #650 with `expm`, `schur` and `sylvester`.
Then resolve #656, a request from branes-ai/cortex for a triangular-output
product and a congruence A P Aᵀ for its Kalman covariance updates. Then cut
5.21.0.

**Merged:** mtl5 #649, #651, #653, #655, #657, #658, #659, #660, #661, #662.

**Closed:** #656. #650 stays open for the complex Schur form and the
condition estimators.

**Filed:** stillwater-sc/universal#1846, `posit<64,3>` `sqrt` accurate only to
double.

## The through-line

Two of this session's defects passed a check that only established that
something existed.

The CMake probe for `?gemmt` linked a call to `dgemmt_` and declared the
routine available. On Ubuntu 24.04, OpenBLAS 0.3.26 exports `dgemmt_`, so the
probe passed, and every call MTL5 made with k > n came back with "parameter
number 10 had an illegal value" and C untouched. The cause was in OpenBLAS's
`interface/gemmt.c`: the Fortran entry checks `ldb` against A's row count and
`lda` against B's. 0.3.27 fixed it. The probe now runs an n = 1, k = 2 call
and accepts the routine only if it returns 11. CI then showed the right
outcome on each lane: "links but fails its probe" on the two OpenBLAS 0.3.26
lanes and "found" on BLIS.

The Householder β bug was the same shape in numerical form. The formula
2v₀²/(σ + v₀²) was correct in exact arithmetic and passed every float and
double test. It failed only in types with little range below 1, where v₀² for
a small tail underflows. It surfaced only because the `sylvester` accuracy
table was measured for eleven element types instead of two: half and
posit<16,1> stood out at 57 and 215 ε. The rewrite, β = −v₀/‖x‖, squares
nothing small, and those rows fell to 1.6 and 4.7 ε.

## Work completed

### Matrix functions (#649, #651, #655, #657)

- **The assessment (#649)** concluded that neither quaternions nor Lie-group
  classes belong in MTL5, but the matrix functions they are built on do. That
  became the plan in #650.
- **`expm` (#651):** scaling and squaring with Padé approximants, using Al-Mohy
  and Higham's 2009 refinements. θ_m is tabulated for nine unit roundoffs by
  `tools/expm_theta.py`, so a posit or a double-double gets a degree suited to
  its own precision rather than double's. On Linux aarch64, the reference
  values printed with 21 digits failed the binary128 check: the test was
  measuring the reference's own rounding. They are now printed to 40 digits.
- **`schur` and `rsf2csf` (#655):** the Francis loop moved out of
  `eigenvalue()` into `detail/francis_qr.hpp`, with an eigenvalues-only and a
  Schur-form variant. `eigenvalue()` stays bit-identical on 864 cases. Without
  a power-of-two prescaling, the in-house path did not converge for entries
  near 1e±200.
- **`sylvester` (#657):** Bartels–Stewart on the Schur forms, with `trsyl`'s
  perturbed-pivot and overflow conventions, plus a `_scaled` variant for the
  condition estimators to come.

### `mm_read_dense` duplicates (#653)

A user reported, through the mtl5 Python package, that a Matrix Market file
with a repeated coordinate read differently from SciPy. `mm_read_dense`
assigned where the sparse reader accumulates. The test for mtl5-python is
drafted and goes in when that repo's MTL5 pin moves to 5.21.0.

### Householder β (#658, #659)

This is the fix from the through-line. The sylvester accuracy table was
remeasured on main and republished (#659). Every row moved in the last digit,
because the Schur factors changed.

### `gemmt` and `congruence` (#656, #660)

- **`gemmt`** forms one triangle of αAB + βC and mirrors it. It has four paths:
  BLAS, an i-k-j path for row-major operands, a path for a transposed CRS
  `B`, and a dot-product fallback. The generic paths are bit-identical to
  `mult`.
- **`congruence`** is T = A P with `mult`, then `gemmt(T, Aᵀ)`. A dense A is
  copied out as Aᵀ first, because the plain loop gained only 1.07–1.2× over the
  two-product form; after the copy the generic gain is 0.73–0.78× the time. A
  CRS A follows nnz: 60 µs against 1819 µs dense at d = 200.
- **The origin story (#661)** traces the operation back to the Joseph
  covariance update. Several claims in the source text were corrected before it
  went in: the standard update formula, "guaranteed positive semi-definite
  regardless of rounding", and an unverified claim about BLIS packing. Review
  then tightened two more: gain sensitivity is second order only near the
  optimal gain, and a block-diagonal R is not diag(r).

## Review fixes

- **FMA contraction (#660).** The bit-identity tests failed on AArch64 GCC, on
  the x86-64-v3 lane, and on the BLIS lane, whose Ubuntu 26.04 amd64v3 image
  contracts by default. `op_test_gemmt` joined the `-ffp-contract=off` list, as
  `op_test_mult_ikj` had.
- **Guard too narrow (#660).** The double CRS-vs-dense comparison was skipped
  only with `?gemmt`. With BLAS but a broken `?gemmt`, the dense A·P is still a
  `dgemm`. CodeRabbit then added the `MTL5_NATIVE_FAST_GEMM` case: its
  micro-kernel calls `fma` explicitly, which no contraction flag reaches.
- **Coefficient type (#660).** A float C with double α took BLAS, which rounds
  α to float, or the generic path, which combined in double, depending on the
  operands. CodeRabbit proposed skipping BLAS for a mismatched coefficient
  type. That would have sent `gemmt(2.0, Af, Bf, 0.0, Cf)` to the slow path, so
  instead both paths convert α and β to C's element type, as the BLAS
  interface does.
- **Symmetric C (#660).** With β ≠ 0, only the formed triangle of the old C is
  read. That precondition is now documented for both `gemmt` and
  `congruence`.

## The parts that went wrong

- **#658 fixed half and broke fp8 and fixed point**, and only the release's
  downstream build showed it. Building mtl5-python against the candidate
  failed six tests. The old squared β had underflowed to zero in 8-bit floats
  and fixed point, which made those columns the identity: degenerate, but
  finite. The new β did not underflow, so it produced reflectors those types
  cannot hold. fp8 overflowed v into NaN. fixpnt<16,8> got β = 2⁻⁸, one ulp,
  and β vᵀv = 0.5 where it should be 2. #662 checks that quantity and
  returns the identity when the reflector is further from orthogonal than
  leaving the column would be. The first version of the check summed vᵀv in
  the type, which saturates fixpnt at 128 and read c ≈ 2 for a reflector whose
  true c was 2.6. A property test over 800 random vectors caught that, and the
  sum was rewritten so that every partial stays near 2. The same check took
  posit<64,3> from 29 to 2.5 ε in the sylvester table.

- **The probe that only linked** is the through-line above. The local
  OpenBLAS was 0.3.32 and BLIS passed, so nothing local could have shown it.
  The failure was visible only on CI's distribution package.
- **A test comment about which builds sum in vendor order was wrong twice**:
  first it named only `?gemmt`, then it missed the native blocked GEMM. Both
  cases had already been reasoned about elsewhere in the same file.
- **A stale background waiter** from #651 was still polling hours later. It
  was harmless, but a waiter should die with its PR.

## Tooling notes

- A Catch2 test can be built by hand against a configured build directory.
  It needs `-I<build>/_deps/catch2-src/src`, `-I<build>/_deps/catch2-build/generated-includes`,
  and the two static libraries. That made it quick to check the "BLAS without
  `?gemmt`" and `MTL5_NATIVE_FAST_GEMM` configurations without a new configure.
- CodeRabbit's outside-diff findings have no threads. They are answered in
  the commit message that fixes them.

## Lessons

- A fix that removes an accidental underflow also removes the safety net that
  the underflow provided. When a result was "degenerate but finite" by
  accident, the fix has to make it so on purpose.
- A capability probe has to run the capability, in the case that would fail,
  not merely link it.
- An accuracy table over many element types is a test of the algorithm's
  formulas, not only of its precision. The types with little range below 1
  are the ones that show a squared small quantity.
- Where one function has a BLAS path and a generic path, every input the
  interface accepts must reach both with the same arithmetic. Coefficient
  types are inputs too.

## Follow-ups

- #650: the complex Schur form and `complex_for<T>`, the transposed `trsyl`
  forms, and the condition estimators.
- `eigenvalue()` still uses a fixed 1e-10 default tolerance and overflows near
  1e200.
- mtl5-python: move the MTL5 pin to v5.21.0 and add the `mm_read_dense`
  duplicates test. When the pin moves, `TestFixpnt16LQSaturates` fails by
  design: fixpnt16 LQ is now degenerate like its QR (residual 0.16, was 1.6).
  Its premise was also wrong, since fixpnt16 QR was never "usable": in 5.20.0
  every τ is already 0. fixpnt16 belongs in `DEGENERATE_QR`.
- Rerun `bench_congruence` under the `performance` governor.
- `normest1`, and Padé degrees above 13 for `expm`.
