# Session: a count is not a seed

**Date**: 2026-11-27 (late; the evening's log is `2026-11-27-the-quotient-leaves-the-range-first.md`)
**Duration**: One PR, one issue filed
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

#961, an indefinite parametric `randsym`, which mtl5-python (stillwater-sc/mtl5-python#165) was waiting for. CodeRabbit's free reviews only.

**Merged:**

| PR | Issue | What |
|---|---|---|
| #970 | #961 | `randsym_indefinite(n, kappa, mode, num_negative[, seed \| gen])` |

**Filed:** #971, MINRES with a preconditioner scaled beyond about 2^±700.

## Decisions

Both were asked before the code was written, and both are recorded in memory.

- **A new name, `randsym_indefinite`.** The issue proposed a `randsym` overload with the count as a fourth argument. Its non-deterministic form, `randsym(n, kappa, mode, num_negative)`, would have the same parameter types as the existing seeded `randsym(n, kappa, mode, std::uint64_t seed)`, since `std::size_t` and `std::uint64_t` are one type on Linux and macOS. A count would have been read as a seed, or the declarations would have collided. The user chose the new name.
- **A random subset of positions is negated, as LAPACK's `DLATMS` does with `RSIGN = 'T'`.** This was the issue's open question.
  - **The draw comes after the spectrum.** Mode 5 draws its spectrum first, so the subset follows it in the stream.
  - **Portable.** It is a partial Fisher–Yates over `detail::random_unit`, the raw engine stream, not `<random>`'s distributions, as #844 requires of every seeded generator.
  - **Nothing drawn for zero.** With `num_negative = 0` no subset is drawn, so the result is `randsym`'s matrix for the same seed, and a test checks that bit for bit.

## Findings

- **cond = κ needs n ≥ 2** (CodeRabbit on #970). `make_sigma` returns {1} for n = 1 whatever κ is, so a 1 × 1 `randsym_indefinite` is [1] or [−1]. The header and the docs had claimed cond₂(A) = κ without the condition. They now state it, and a test pins the 1 × 1 and empty cases.
- **The pinned positions are a portability test.** Seed 961 negates positions {0, 4, 5} of the mode-3 spectrum (λᵢ = 2^−i), read back from the negative eigenvalues' magnitudes. They were computed by a scratch program replaying the draw. CI's eight platforms check that every standard library gives the same.
- **MINRES's range is about 2^±700, by a 3/2 power** (#971, found while testing #968).
  - **The measurements.** A sweep on main gave code 0 up to 2^±600 and at 2^700, code 2 from 2^750, and code 6 from 2^−700.
  - **The cause.** The unnormalized Lanczos vector v_new scales as s^(−1/2), so its preconditioner solve scales as s^(−3/2). That underflows past about 2^716 and overflows past about 2^−683, which matches the onsets.
  - **Not the M-norm.** #968's M-norm is not involved. The issue proposes rescaling v_new by a power of two before the solve.

## Process notes

- **CodeRabbit's allowance** ran out after the review fix's push, with a 53-minute wait. A $0.75 offer was declined, and the user merged #970 by hand during the wait.
- **Ordering of what's next** (asked for #885, #928, #929 and #930):
  1. #928 and #929 in one PR. Both add fields to the result struct in `benchmarks/platform/run.hpp` and to the JSON writer in `mtl5_bench.cpp`.
  2. Then #930, the sidecar keys.
  3. Then #885, the lab campaign. Its sessions then carry versions, spreads and complete sidecars.
