# Session: Benchmark methodology, MKL comparison, and the native-BLAS performance plan

**Date**: 2026-05-31
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Harden and correct the benchmark harness, stand up a reproducible native-vs-OpenBLAS-vs-MKL comparison, fix two latent correctness bugs surfaced along the way, publish a Doxygen API reference, and — the headline — research how high-performance BLAS implementations work and lay out a trackable plan to bring MTL5's native dense kernels close to hand-tuned performance.

## Context

The day began as housekeeping (remove stray multibyte characters, harden CI) and grew into a deeper look at MTL5's dense performance story. Benchmarking exposed that the "native" numbers were not measuring what we thought, which in turn revealed that the benchmark methodology itself was wrong for the question we were asking. Fixing that cleanly set up the real goal: a concrete engineering plan to make the native BLAS fast.

## Work Completed

### Correctness fixes
- **`antisymmetric_tensor::set` out-of-bounds write** (#61 → PR #63). For `i == j` the diagonal `assert` is compiled out under `NDEBUG`, and control fell through to `asym2_index(i, i)` which returns the `num_stored` sentinel — a write one past the end. Proven with an AddressSanitizer repro (stack-buffer-overflow), fixed as an explicit no-op, regression test added.
- **Benchmark `native` eigenvalue backend was secretly LAPACK** (PR #78). `eigenvalue_sym_op<Native>` called `mtl::eigenvalue_symmetric`, which auto-dispatches to LAPACK — so `native` and `lapack` were identical (~1.0x). Extracted `eigenvalue_symmetric_generic()` and pointed the native path at it; native eig is now ~5–10x slower than LAPACK, as a real C++ reference should be.

### CI / tooling hardening
- Pinned all GitHub Actions to commit SHAs + `persist-credentials: false`; added `.github/dependabot.yml` (#62 → PR #64). Dependabot immediately bumped the actions (#66–#71).
- Gated sccache to trusted runs (push / same-repo PR) to close a zizmor `cache-poisoning` finding (#65 → PR #74).
- `.gitignore` for Claude Code per-user/runtime files (PR #76).

### Documentation
- **Doxygen C++ API reference** wired into the Astro/Starlight docs site (PR #73): `docs-site/Doxyfile` (header-only, `INPUT = ../include/mtl`), `npm run api`, a sidebar link served at `/mtl5/api/`, and a CI step. Mirrors the sister `universal` repo's pipeline.
- Benchmark README: fixed the `MTL5_WITH_BLAS/LAPACK` option names and documented the Intel MKL build (`BLA_VENDOR=Intel10_64lp`) (PR #75).

### Benchmark suite evolution
- **Size-N sweep + L1/L2/L3 suite groups** (PR #77): `--sweep START:STOP:STEP|:xFACTOR`, default sizes that bracket powers of two with odd / 1.5x neighbours to surface padding overhead.
- **Installed Intel oneAPI MKL 2026.0** and produced a three-way native/OpenBLAS/MKL comparison across an odd-size sweep. Findings: OpenBLAS is highly competitive with MKL; MKL leads on QR and small-size LU/Cholesky; OpenBLAS overtakes MKL on `eig_sym` past N≈400; GEMM is ~tied at ~70+ GFLOP/s; native is ~4–5 GFLOP/s (~15x off).
- **GFLOP/s-vs-N plotting** (`plot_results.py`) + committed example CSVs and rendered plots with platform provenance (PRs #78, #79, #80).
- **Methodology correction — one executable per backend** (PR #81). Diagnosed why the plots showed two different `native` curves: the policy-tag design measured `native` + `blas` in the *same* binary, so identical generic code was timed in different process environments (and, on the hybrid i7-12700K, on different P/E cores). Rewrote `bench_all` to call only the public `mtl::` API and let build flags pick the backend (one build = one backend = one CSV); deleted the `Native/Blas/Lapack` policy-tag harness (−800 LOC net). Added BLAS `?dot` dispatch to `mtl::dot`/`dot_real` (it had none — only the benchmark's private wrapper did), and guarded the `int` length cast in both `dot` and `two_norm` against overflow.

### Performance research and plan (the main deliverable)
Researched how production BLAS gets its speed and how to layer the techniques into MTL5:
- **GotoBLAS / BLIS / OpenBLAS**: the 5-loop cache-blocking nest (`kc→L1, mc×kc→L2, kc×nc→L3`), operand **packing** into contiguous, aligned, SIMD-friendly buffers (a TLB argument as much as a cache one), and a register-blocked `mr×nr` **micro-kernel** doing `kc` rank-1 FMA updates. BLIS's insight: write the loops + packing once; isolate the arch-specific work in one micro-kernel. Low et al.: blocking params can be *computed* from a hardware-traits struct rather than searched.
- **SIMD abstraction** (xsimd `batch`, Eigen `PacketMath`, `std::simd`): write the kernel once against a vector-batch type with a `constexpr` lane count and one `fma` primitive; the lane width is the unroll factor — decoupling unrolling from the algorithm.
- **MTL5 today**: naive IJK `mult_generic`, no SIMD/blocking/packing, unaligned heap storage, no `-march` flags — but a clean dispatch seam in `mtl::mult` where a native fast path slots between external BLAS and the generic fallback.

Filed **Epic #82** with 11 trackable sub-issues (#83–#93), assigned to Ravenwater, on the **MTL5** project board, milestone **v0.6**, each sub-issue sized **M** / estimate **5**:

| Phase | Sub-issues |
|-------|------------|
| 0 — Foundations | #83 SIMD abstraction layer · #84 over-aligned storage/allocator · #85 build flags + `hw_traits` + constexpr blocking params |
| 1 — L1/L2 | #86 vectorized L1 (dot/nrm2/axpy/scal) · #87 optimized GEMV |
| 2 — GEMM core | #88 register micro-kernel · #89 packing · #90 macro-kernel + 5-loop nest + dispatch · #91 correctness/edge tests |
| 3 — Scale | #92 multithreading · #93 benchmark gate (within 10–20% of OpenBLAS) |

Critical path: #83 → #88 → #89 → #90; Phase 1 parallelizable for early wins.

## Pull Requests Merged

| PR | Title |
|----|-------|
| #63 | fix(tensor): prevent out-of-bounds write in `antisymmetric_tensor::set` on diagonal |
| #64 | chore(ci): pin GitHub Actions to commit SHAs and disable credential persistence |
| #66–#71 | chore(deps): Dependabot bumps of the pinned actions |
| #73 | feat(docs): add Doxygen C++ API reference to the docs-site |
| #74 | chore(ci): gate sccache to trusted runs to prevent GHA cache poisoning |
| #75 | docs(benchmarks): fix BLAS/LAPACK option names and add MKL instructions |
| #76 | chore(build): gitignore Claude Code per-user and runtime files |
| #77 | feat(benchmarks): add size-N sweep and L1/L2/L3 suite groups |
| #78 | fix(benchmarks): Native eig uses generic solver; add GFLOP/s-vs-N plotting |
| #79 | chore(benchmarks): add LAPACK odd-size sweep to example data |
| #80 | chore(benchmarks): add rendered GFLOP/s-vs-N plots for the example sweeps |
| #81 | refactor(benchmarks): one executable per backend via the public API |

## Key Decisions

- **Benchmark a backend the way an application uses it** — one binary compiled against one backend, not in-process policy switching. This removed a measurement artifact and is the model the performance Epic will be evaluated against (a `native-fast` build/label).
- **`dot` should be accelerated in the library, not just the benchmark** — the public API now dispatches to BLAS, keeping the L1 surface coherent with `two_norm`.
- **Compute blocking parameters, don't search them** — adopt the analytical BLIS model so the SIMD/cache parameters are `constexpr` from a `hw_traits` struct.
- **SIMD substrate**: model on `std::simd` semantics (native in GCC ≥ 11, `vir-simd` shim for Clang), with xsimd as the option if single-binary runtime ISA dispatch is wanted.

## Environment Notes

- Intel oneAPI **MKL 2026.0** installed (`/opt/intel/oneapi`), selectable via `-DBLA_VENDOR=Intel10_64lp` after sourcing `setvars.sh`.
- Benchmark host: i7-12700K (hybrid 8 P-core + 4 E-core) — pin to a P-core (`BENCH_CPU=4`) for stable single-thread numbers; unpinned runs let short L1 kernels land on E-cores and skew results.

## Next Steps

- Start the Epic at the root of the critical path: **#83 (SIMD abstraction layer)**, with **#86/#87** (L1/GEMV) in parallel for early, low-risk wins.
- Add a `native-fast` label to the benchmark harness as the acceptance instrument (#93).
