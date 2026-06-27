# Session: Epic #157 wrap-up — mixed-precision GEMM SIMD widening and the kernels guide

**Date**: 2026-06-27
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Close out the mixed-precision tensor-ops epic (#157) by giving GEMM the SIMD
**widening** fast path its mixed-accumulator dot product already had, and by
writing the introductory documentation that ties the whole mixed-precision story
together for newcomers.

## Context

Epic #157 makes MTL5's dense BLAS express the **three independent precisions** of a
mixed-precision op — element (storage) / accumulator (compute) / result
(serialize) — through a shared `accumulator_traits` policy with a fused
accumulate→store epilogue. The foundation and the per-operation policy shipped in
the prior session (#158 traits, #159 dot, #160 gemv, #161 gemm, #162 norms, #163
dispatch, #164 convert, #165 SIMD widening dot). One gap remained: GEMM's
mixed-accumulator path still ran the **scalar generic kernel** — the blocked
native-fast GEMM was same-type only. This session closed that gap (#176) and added
the conceptual on-ramp (#200).

## Work Completed

### SIMD widening GEMM (#176 → PR #199, merged)
Made `mult<double>(A_float, B_float, C_double)` accumulate in fp64 through the
blocked GEMM at SIMD speed instead of the scalar generic kernel.

- **Two-type kernel.** Generalized `gemm_microkernel` and `gemm_blocked` to
  `<TC accumulator, TAB operand>` with `TAB = TC` default. The micro-kernel loads
  narrow operands via `if constexpr (sizeof(TAB) < sizeof(TC))
  batch<TC>::load_widen<TAB> else load_unaligned` and accumulates in `TC`
  registers. **For `TAB == TC` it compiles to the original same-type kernel**
  (identical load + identity cast) — zero regression to the heavily-used
  float×float / double×double paths, by construction.
- **Dispatch.** `mult` routes the float→double case (dense, contiguous,
  `MTL5_NATIVE_FAST_GEMM`) to `gemm_blocked<double,float>`; every other custom
  accumulator keeps the generic scalar kernel.
- **Results.** Matches a double GEMM on the exact widened operands to 1e-9
  (bit-exact on the edge-tile shapes tested); **10–16× faster** than the scalar
  generic kernel (Highway, N=256/512); ASan/UBSan clean; gcc + clang across
  scalar-batch, Highway, and `NATIVE_FAST_GEMM`+Highway configs.
- **CI coverage (CodeRabbit review):** the default CI preset leaves
  `MTL5_NATIVE_FAST_GEMM` off, so the widening test only exercised the generic
  kernel in CI. Added `test_gemm_widen_native.cpp`, a TU that forces the macro so
  the blocked widening kernel runs in **every** CI build (scalar-batch on default
  gcc/clang; true SIMD on a Highway build).

### Mixed-precision kernels guide (#200, merged)
`docs/algorithms/mixed-precision-kernels.md` — the introductory page on
mixed-precision algorithm design and optimization, structured why / what / how:

- **Why** — store narrow, accumulate wide: error concentrates in accumulation,
  not storage; the bandwidth/cache/SIMD/energy economics of narrow types.
- **What** — the Element → Accumulate → Result model, `accumulator_traits`, the
  `mult/dot/two_norm<Accumulator>` API, the `void`-default and
  custom-accumulator-forces-native guarantees, and the iterative-refinement tie-in.
- **How** — the SIMD widening GEMM as the worked example: blocking / packing /
  register tiling, why the naive mixed path is slow, the widen-on-load trick, the
  two-type kernel, the dispatch, and the measured speedups.

Cross-links the two deep design notes (BLAS kernel architecture; custom number
types through the SIMD BLAS) and Measuring Solver Accuracy rather than duplicating
them; wired into `FILE_MAP` + the algorithms overview (sidebar auto-generates).

## Decisions & Rationale

- **Two-type kernel over a forked widening variant.** A single `<TC,TAB>` template
  with `TAB = TC` default keeps the same-type path byte-identical (the `if
  constexpr` picks the original load), avoiding code duplication while isolating
  the hot path from regression — verified by the unchanged same-type GEMM suites.
- **Widen on load, not a separate up-convert pass.** Reusing the `batch::load_widen`
  primitive from #165 keeps operands stored/streamed narrow (bandwidth, cache,
  lanes) while accumulating wide — no extra pass over the data.
- **Force native-fast in a dedicated test TU.** Rather than a new CI job, a TU that
  defines `MTL5_NATIVE_FAST_GEMM` gives blocked-path coverage in every existing CI
  build, with the scalar-batch fallback still exercising the two-type kernel logic.

## Epic #157 — final status (closed)

| Sub-issue | Delivered |
|---|---|
| #158 | shared `mtl::math::accumulator_traits` + `value<Result>` round-out |
| #159 | `dot` / `dot_real` accumulator + result policy |
| #160 | `gemv` accumulator + result policy |
| #161 | `gemm` / `mult` accumulator policy, result type inferred from `C` (headline) |
| #162 | `two_norm` / `frobenius_norm` accumulator policy |
| #163 | dispatch: a non-default accumulator forces the native kernel |
| #164 | standalone `mtl::convert` (non-fused re-quantization) |
| #165 | SIMD widening `dot` (~2.6×) |
| #176 (follow-up) | SIMD widening `gemm` (10–16×) — this session |
| #200 (follow-up) | mixed-precision kernels introduction — this session |

The epic's design — *Element → Accumulate → Result*, fused epilogue, byte-identical
default, native-kernel dispatch for custom accumulators, portable across
`Universal` number types — is fully realized and now has both SIMD acceleration
(dot + gemm) and a teaching entry point.

## Outcome

Mixed-precision dense BLAS in MTL5 is complete and documented: the three-precision
policy across dot/gemv/gemm/norms, SIMD widening for the two compute-heavy
reductions, and an introductory guide. The capability that distinguishes MTL5 from
the fixed-precision external BLAS — accumulate-wide-store-narrow at SIMD speed,
for arbitrary number types — is on `main`.

## Issues / PRs

- Merged this session: #199 (SIMD widening GEMM, #176), #200 (mixed-precision kernels doc)
- Epic: #157 (closed — fully delivered)
- Builds on the prior session's #158–#165 (mixed-precision tensor BLAS foundation)
