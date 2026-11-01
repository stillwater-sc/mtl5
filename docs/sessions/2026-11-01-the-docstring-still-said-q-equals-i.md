# Session: the docstring still said Q = I

**Date**: 2026-11-01
**Duration**: A single working session, after the 5.22.0 tag
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Tag 5.22.0. Then move every peer repository that pins MTL5 to it, and cut
mtl5-python 5.22.0 to PyPI.

**Released:** MTL5 v5.22.0 (#676); mtl5-python v5.22.0 on PyPI.

**Merged:** mtl5 #676; mtl5-python #120; mp-dsp #219; mp-dsp-python #136.

## The through-line

Every repository built against 5.22.0, and every bump PR's CI passed. The
local runs failed only for environmental reasons, listed per repo below. What
needed fixing was the text describing the old results.

#675 gave `householder()` LAPACK `dlarfg`'s sign. In mtl5-python that ended
the narrow formats' fallback to Q = I. The code needed no change, but three
kinds of text still described the fallback:

- **Tests.** Six tests in `tests/test_universal_factorizations.py` asserted
  that fp8 and fixpnt8 return Q = I and R = triu(A), and
  `TestFixpnt16LQSaturates` asserted that fixpnt16 saturates in LQ. They were written as documentation of a limit, and against
  5.22.0 they failed as designed. They are now tests that the narrow formats
  degrade without degenerating. Q is not I, the factors are finite, and the
  residual stays under a bound per format. All eight new tests fail against
  the old reflector.
- **Docstrings.** `qr`'s and `lq`'s docstrings in `mtl5/__init__.py`, and the
  QR registration comment in `python/src/mtl5_dense_factor.cpp`, still said
  fp8 and fixpnt8 return Q = I. CodeRabbit found them. Nothing executes a
  docstring, so the suite can't catch one that is out of date.
- **CHANGELOG.** In mp-dsp the first draft said the unscented Kalman filter
  does not reach MTL5's `ldlt`. It does, at `ukf.hpp:186`. CodeRabbit caught
  that too.

## Work completed

### MTL5 5.22.0

Tagged on 2d30faa after the peer repos built against the candidate: mp-blas
18/18, mp-ir 4/4, mp-iterative 17/17, mp-ode 1/1, mp-spice 1/1, and
mp-numerics. `release.yml` passed.

### mtl5-python #120 and the 5.22.0 release

- **Pin and version:** `GIT_TAG v5.22.0`; the package version is 5.22.0, since
  its minor tracks MTL5's.
- **Floor:** the `find_package` floor rises from 5.19.0 to 5.22.0, because the
  tests now pin results that 5.21.0 and 5.22.0 change.
- **New test:** `tests/test_io.py` gains `TestDuplicateEntries`, for 5.21.0's
  `mm_read_dense` summing duplicate coordinates (#653).
- **Local run:** 1633 passed. The three failures are environmental and fail on
  main too: a stale `_core` and two scikit-learn pairwise tests.
- **Release:** the minor is cut by hand, so `v5.22.0` was tagged on the merge
  commit d3cdb14 and the GitHub Release published.
  - `wheels.yml` built the four platforms and the sdist, and `publish-pypi`
    uploaded 13 files.
  - The macOS wheels are tagged `macosx_11_0_arm64`, the deployment target
    #119 pinned.

On the 8×8 fixture after the bump:

| format   | QR residual before | QR residual after |
|----------|--------------------|-------------------|
| fixpnt16 | 0.135              | 0.015             |
| posit8   | 0.21               | 0.044             |
| fp8      | Q = I              | 0.10              |
| fixpnt8  | Q = I              | 0.11              |

fixpnt16 is now usable for QR and LQ. The 8-bit formats have real orthogonal
factors but are still too inaccurate to use.

### mp-dsp #219

- **Pin:** the headers-only fallback fetched MTL5 v5.12.0, ten minor releases
  behind. It now fetches v5.22.0.
- **What mp-dsp uses from MTL5:** the dense containers, `inv`, `trans` and
  `ldlt`.
- **Result:** 63/63 pass with 0 warnings, and no test changed.

### mp-dsp-python #136

- **Pins:** `MPDSP_REQUIRED_MTL5_VERSION` and `MPDSP_MTL5_PIN` move from 5.12.0
  to 5.22.0, and the README's peer table follows.
- **Local run:** 1666 passed. The rest failed or errored for environmental
  reasons: seaborn is missing, and the version reads 0+unknown outside a tagged
  checkout.
- **One CodeRabbit comment declined:**
  - **The point:** the settings are `CACHE` entries without `FORCE`, so an
    existing build directory keeps the old pin.
  - **Why declined here:** all six floor and pin settings in the repo behave
    that way, including the ones #132 bumped.
  - **Impact:** fresh configures, CI and wheel builds are unaffected.
  - **Next:** a migration belongs in its own change.
- **Worktree:** the work ran in a separate worktree, so the clone's
  `feat/issue-133-precision-profiles` checkout was never touched.

### Not bumped

mp-blas, mp-ir, mp-iterative, mp-ode, mp-spice and mp-numerics track MTL5
main, so there was no pin to move. universal_dtypes does not use MTL5.

## The parts that went wrong

- **A real merge sent to check a flag's syntax.** To see how
  `--match-head-commit` parses, `gh pr merge 120` was run with a shortened SHA.
  GitHub rejected the SHA, so nothing merged. The command would have merged
  #120 if the SHA had been accepted. The real merge followed, with full SHAs
  read from `gh pr view`.
- **The editable install shadowed the build under test.** The venv has an
  editable mp-dsp-python install pointing at the clone, and it outranked
  `PYTHONPATH`. The test run used a clean venv instead. This is the same trap
  as the last session's probe, which also measured the wrong module.
- **`/tmp` quota.** The tmpfs hit "Disk quota exceeded" partway through. The
  scratch builds were deleted when done.

## Lessons

- **When a result changes, search the prose too.** Tests, docstrings, comments
  and CHANGELOGs that describe the result go stale with it, and only the tests
  fail.
- **Write a limit test so it can tell better from worse.** A test that pins a
  degenerate result fails when the result improves. "Degrades without
  degenerating", with a bound per format, keeps failing when the result gets
  worse but survives improvements.
- **A fallback pin is never exercised, so nothing reports it falling behind.**
  mp-dsp's fallback sat at v5.12.0 through ten releases, because every build
  that mattered found MTL5 some other way.
- **Never probe a state-changing command's syntax by running it.** Read
  `--help`, or use a target that can't succeed.

## Follow-ups

- mp-numerics #14 (decompositions) and #15 (Kalman filters). Regenerate the
  givens_householder results against 5.22.0. Closing #13 is the maintainer's
  call.
- mp-dsp-python: a stale-cache migration for the `MPDSP_*` CACHE settings
  (declined in #136).
- universal_dtypes installs `cibuildwheel` unpinned.
- mtl5-python's `[Unreleased]` CHANGELOG section mixes entries that shipped in
  5.20.1, 5.20.2 and 5.22.0 with semantic-release's per-version sections
  below it.
- #650; the eigenvalue tolerance; `normest1` and the `expm` Padé degrees.
