# Session: the sign was the cause

**Date**: 2026-10-29 to 2026-10-31
**Duration**: A multi-day working session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Measure the Givens rotations and Householder reflectors MTL5 is built on, in
every number system mp-numerics can instantiate, and publish what was found.
Then act on it: scale the rotations by an exact power of two (#671), and give
`householder()` LAPACK `dlarfg`'s sign (#672). Then cut 5.22.0.

**Merged:** mtl5 #667, #668, #669, #670, #673, #674, #675; mp-numerics #16,
#17, #18, #19; mtl5-python #119; universal_dtypes #73.

**Closed:** #671, #672.

**Filed:** mtl5 #671 and #672; stillwater-sc/universal #1881 (Dekker's split
overflows near DBL_MAX, so dd and qd products and quotients there are NaN),
#1882 (dd and dd_cascade `numeric_limits`: an unnormalized `max()`, `min()`
inconsistent with `min_exponent`, an `epsilon` that is not a power of two) and
#1883 (dd `sqrt` 7.9 units of 2⁻¹⁰⁶ against dd_cascade's 0.75).

## The through-line

Both changes came down to one choice that only showed in a narrow type.

MTL5's reflector chose Parlett's sign for x₀ > 0: v₀ = −σ/(x₀ + ‖x‖), which
reflects onto +‖x‖ and keeps R's diagonal non-negative. In exact arithmetic
it is as good as LAPACK's. In double it cost at most about 9 ε. But for a
small tail it makes v₀ small and v(1) ≈ −2x₀/x(1) large, and a posit or a
fixed-point type has few bits there. The study's tail sweep fixed x₀ = ±1, so
MTL5's scaling, a division by 1, rounded nothing, and the sign was the only
difference: posit<64,2> measured 8.0×10⁵ ε with Parlett's sign and 4.1 ε with
`dlarfg`'s.

The Givens rotation had the same shape. r = sqrt(a² + b²) is exact in
principle and fine in double for half the exponent range. LAPACK's two
remedies, `lapy2`'s max-scaling and `dlartg`'s safe scaling, take their
thresholds from where a type overflows, and in a tapered type or dd that is
not where the type is precise. Scaling by an exact power of two was the only
form that was clean and within a few ε in all 65 types.

## Work completed

### The study (mp-numerics #16, #17, #18; mtl5 #667–#670)

- **mp-numerics #16 and #17** measured seven rotation and reflector forms over
  a scale sweep, a tail sweep and repeated application, in IEEE floats,
  posits, fixed point, cfloats, LNS, takums, dd and qd.
- **The mtl5 docs** tell the story: a history from Jacobi to `dlartg` and
  `dlarfg` (#667), animated figures with their generator (#668), an
  interactive explorer whose arithmetic is tested against the generator's
  vectors (#669), and the results page (#670).
- **The explorer's reflector rounded in a different order** from
  `qr_factor`'s `apply_householder_left`. The old order failed 21 of 154
  reference vectors once the generator produced reflector-application cases
  (#670).
- **Universal's dd near DBL_MAX:** the study's dd column led to the three
  Universal issues above. `split()` sends DBL_MAX's high half to infinity, so
  `dd(DBL_MAX) * 0.5` is NaN. mp-numerics scales dd and qd limb by limb with
  `ldexp` to stay clear of it, and its README says so (#18).

### Givens rotations, scaled (#671, #673)

- **`math::pow2_scaling<T>`** is a customization point: `std::ilogb` and
  `std::ldexp` for the built-in types, an exact halving-and-doubling fallback
  for anything else. `givens_rotation` and `scaled_hypot` are built on it, and
  GMRES, MINRES and `eigenvalue_symmetric` use them at all five sites.
- **The Wilkinson shift's numerator** b·b/(…) overflowed once its hypot no
  longer did; the new test at 2^600 found it. It is now b·(b/(…)).

### `householder()`, LAPACK's sign (#672, #675)

- **The decision:** keep the default and add a tag, or change the default.
  The default real path was the only one using Parlett's sign. The complex
  path, the textbook tag and the `geqrf` dispatch already used LAPACK's, so a
  LAPACK build and a non-LAPACK build gave R opposite signs. The default
  changed, as a breaking change in 5.22.0, with `qr_normalize_signs` and
  `lq_normalize_signs` to restore the old convention.
- **The negligible-tail shortcut was removed.** It existed because Parlett's v
  was huge for a tail below an ulp. With `dlarfg`'s sign, v is tiny there, and
  keeping the shortcut would have made R(k,k)'s sign depend on the tail's size.
- **The fixture that #662 fell back on** now keeps a reflector. The first column of
  mtl5-python's QR fixture, in a fixpnt<16,8>-like type, had β = 2⁻⁸ with
  Parlett's sign. It now gets a reflector accurate to the type's 2⁻⁸
  resolution: tail 0.0029, orthogonality 0.0031.

### macOS 14 (#674, mtl5-python #119, universal_dtypes #73)

GitHub retires the macOS 14 runner image on 2026-11-02, and its scheduled
brownouts cancelled the macOS 14 job on each CI run of #673. mtl5 moved to macos-15 and
macos-26. mtl5-python and universal_dtypes build their wheels on macos-15 with
`MACOSX_DEPLOYMENT_TARGET` pinned at 11.0, the tag of every macOS wheel they
have published. mtl5-python's post-merge `validate-wheels` run built
`mtl5-5.20.2-cp312-cp312-macosx_11_0_arm64.whl` on the new image. Universal's
workflows already used `macos-latest` and `macos-15-intel`, so it needed
nothing.

## Downstream, before the release

- **mtl5-python** built against the #675 branch: every narrow format
  improved on its 8×8 fixture. fixpnt16 QR went from 0.135 to 0.0149 and LQ
  from 0.160 to 0.0129. posit8 QR went from 0.209 to 0.044. fp8 and fixpnt8
  stopped returning Q = I. Six tests pin that degeneracy and fail by design;
  mtl5-python pins MTL5 v5.20.0, so they change with the pin.
- **mp-numerics** builds against MTL5 main. Two `test_rotations` checks pinned
  the old reflector, and #19 updated them in the merge window after #675.
  CodeRabbit noticed that an installed 5.21 would be found ahead of main;
  `find_package(MTL5 5.22 CONFIG QUIET)` now rejects it.

## The parts that went wrong

- **A bit-identity test without the contraction pin.** #673's test claimed
  the scaled rotation gives the unscaled one's bits inside the old band. That
  holds for separately rounded operations. The LAPACK (BLIS) lane, on Ubuntu
  26.04's FMA-capable GCC default, failed 1223 checks, and `-march=x86-64-v3`
  reproduced it locally. The first fix relaxed the test to 2 ε. CodeRabbit
  pointed at the repo's own convention: the `-ffp-contract=off` target list
  in `tests/unit/CMakeLists.txt`, which #660 had already used for the same
  reason. The test went on the list, and back to `==`.
- **A probe that measured the wrong module.** The first mtl5-python
  measurement printed identical numbers for main and the branch. The script
  ran by path, so `sys.path[0]` was the script's directory, and both runs
  imported the venv's installed mtl5. An explicit `PYTHONPATH` fixed it, and
  `mtl5.__file__` is now printed first.
- **A hung runner.** The LAPACK (g++-13) job of #675 sat in `apt-get install`
  for 74 minutes. It was cancelled and rerun.

## Lessons

- A test that claims bit identity across two code paths needs the
  contraction pin, whatever the paths are. It is a property of the claim, not
  of the code.
- A sign convention that differs between paths of the same function is a bug
  even when each path is correct. The LAPACK and non-LAPACK builds of
  `qr_factor` disagreed on R's signs, and no test compared them.
- An accuracy study over many types finds formula choices, not only precision
  limits. Parlett's sign and the unscaled hypot were both invisible in double.
- When a measurement shows no difference between two builds, check which
  build was measured before believing it.

## Follow-ups

- mtl5-python: move the MTL5 pin to v5.22.0. Update the six fp8 and fixpnt8
  degeneracy tests and `TestFixpnt16LQSaturates`; fixpnt16 is now usable for
  QR and LQ. Add the `mm_read_dense` duplicates test from 5.21.0.
- mp-numerics #14 (decompositions) and #15 (Kalman filters). Regenerate the
  givens_householder results against 5.22.0, where `givens_mtl5` and
  `householder_mtl5` become the remedies.
- universal_dtypes installs `cibuildwheel` unpinned; mtl5-python pins it.
- #650: the complex Schur form, the transposed `trsyl` forms, and the
  condition estimators.
