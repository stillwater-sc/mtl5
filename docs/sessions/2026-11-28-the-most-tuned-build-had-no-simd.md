# Session: the most tuned build had no SIMD

**Date**: 2026-11-28 (afternoon; the previous log is `2026-11-28-one-commit-for-every-card.md`)
**Duration**: One PR
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

#720: with GCC, `MTL5_NATIVE_ARCH=ON` on an SVE machine built a one-lane `simd::batch`. The issue's recommendation was taken: prefer NEON for a scalable target until #427, and warn when a user opts out.

| PR | Issue | What |
|---|---|---|
| #983 | #720 | Native builds on scalable-SVE AArch64 disable Highway's SVE targets |

## The fix

- **Where:** CMake, in the `MTL5_NATIVE_ARCH` block, rather than in `batch.hpp`. The native flags are a CMake option and build-interface only, so this is the one place that knows what they are. One test-compile asks whether they define `__ARM_FEATURE_SVE` without a fixed `__ARM_FEATURE_SVE_BITS`.
- **What:** `HWY_DISABLED_TARGETS=HWY_ALL_SVE`. Highway's static target is the lowest set bit of what remains:
  - with crypto, bf16 and dotprod that is `NEON_BF16`, which Highway allows from GCC 13.2;
  - without AES it is `NEON_WITHOUT_AES`.
- **The opt-out:** `MTL5_NATIVE_SCALABLE_SVE=ON`. It keeps the scalable target and warns, which is the issue's option 1.

## Measured without SVE hardware

No local machine has SVE. Instead, `aarch64-linux-gnu-g++` 15 configured five flag sets, and `simd_test_batch` ran under `qemu-aarch64 -cpu max`:

| Flags | Result |
|---|---|
| N2 with crypto | `NEON_BF16`, 4 lanes |
| N2 without crypto | `NEON_WITHOUT_AES`, 4 lanes |
| The opt-out | `SVE2`, 1 lane |
| Fixed-length SVE | not detected |
| A78C, no SVE | not detected |

CI's N2 runner (SVE vector length 16 bytes) then confirmed `NEON_BF16` with 4 lanes: its `-march=native` includes AES.

## CI lanes

- **`native`** now asserts `HWY_NEON_BF16` and runs the full suite: 321 tests, all passing. It used to run only the SIMD tests, because the rest of its build was scalar. It is the only lane on `NEON_BF16`.
- **`native-scalable`** is new. It uses `MTL5_NATIVE_SCALABLE_SVE=ON`, so the one-lane pin in `test_batch` and the opt-out's warning stay covered. When #427 lands, this is the lane that flips to `size > 1`.

## Process notes

- **CodeRabbit:** the included review, after an 8-minute wait. A $1.25 offer was declined. It found nothing.
- **A wrong target name:** the first qemu run built `test_batch`. The test executables are `simd_test_batch` and so on: the directory is a prefix.
