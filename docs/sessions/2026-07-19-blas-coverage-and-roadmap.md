# Session: capability assessment, expansion roadmap, and complete core BLAS coverage

**Date**: 2026-07-19
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Step back from feature work to assess where MTL5 stands and where it should go
(functionality, performance, distributed memory, hardware accelerators), turn
that into a tracked roadmap, and then execute the first concrete piece of it:
completing MTL5's core BLAS Level-2 / Level-3 operator surface with a
multi-backend (OpenBLAS + BLIS) benchmark comparison.

## Context

Coming out of the eigenvalue epic (#202), the library is functionally broad but
had never had a written, source-grounded assessment of its maturity or a
prioritized expansion plan. On the BLAS side it implemented only `gemv` (L2) and
`gemm` (L3) of the matrix-level routines, and the benchmark harness compared
against OpenBLAS/MKL but not BLIS.

## Work Completed

### Capability assessment & expansion analysis (#219 → PR #219, merged)
A source-grounded assessment (`docs/design/capability-assessment-and-expansion.md`)
built from parallel surveys of `include/mtl/`, `tests/`, `benchmarks/`, and CMake.
Four axes with a maturity scorecard:

- **Functionality** — mature dense/sparse/iterative/eigen coverage; advanced gaps
  (generalized eig, matrix functions, implicit-restart Krylov, AMG).
- **Performance** — strong single-thread (Highway SIMD + BLIS-style blocked GEMM
  at ~80–84% of OpenBLAS); **on-node threading is the biggest gap** (only one GEMM
  kernel threads, default 1).
- **Distributed memory** — none, but the Krylov contract (`LinearOperator` +
  `dot`, and the `operation/resource.hpp` "future distributed-vector" seam) is
  nearly drop-in ready.
- **Hardware accelerators** — no GPU code; the `interface/` backend pattern is a
  clean hook but inherits a float/double-only limit that collides with the
  custom-number-type mission.

Key cross-cutting finding: both "absent" axes share one enabling prerequisite — a
**memory-space abstraction** in the storage layer. The report closes with a
6-step roadmap.

### Roadmap issues (#220–#227)
Filed the roadmap as a tracked epic **#220** with children **#221** (on-node
threading), **#222** (runtime cache/ISA detection + SVE/RVV SIMD), **#223**
(memory-space abstraction — the shared enabler), **#224** (distributed iterative
solving), **#225** (GPU BLAS backend), **#226** (functional depth), plus **#227**
(benchmark all core BLAS routines vs OpenBLAS and BLIS). Assigned, milestoned
(v0.7), and sized on the project board.

### BLIS backend + expanded BLAS benchmark coverage (#227 → PR #228, merged)
Added a `blis` benchmark backend (CMake `BLA_VENDOR=FLAME`, auto-skipped if
absent) to `run_sweeps.sh`/`run_scaling.sh`; expanded the harness to benchmark
all core BLAS routines MTL5 implements (added `axpy`/`scal`); made
`analyze_gate.py --reference` baseline against OpenBLAS, BLIS, or MKL. Validated
locally against real OpenBLAS **and** BLIS (`libblis`, `BLA_VENDOR=FLAME`).

### Complete core BLAS L2/L3 operators (#229, three batches — PRs #230/#231/#232, merged)
Per the decision to pursue **broad BLAS completeness**, filed tracking issue
**#229** and delivered it in three reviewable batches:

- **Batch 1 — Level 2** (#230): `ger`, `symv`, `trmv`, `trsv`.
- **Batch 2 — Level 3 triangular** (#231): `trmm`, `trsm`.
- **Batch 3 — Level 3 symmetric** (#232): `symm`, `syrk`, `syr2k`.

Each operator is a generic templated function (any Matrix/Vector type and
orientation, and custom number types) with optional external-BLAS dispatch for
column-major dense float/double, mirroring the existing `gemv`/`gemm` gating; new
`s/d` bindings were added to `interface/blas.hpp`. Every op is tested on both the
generic and real-BLAS paths (undefined-symbol checks confirm the BLAS path is
taken) and wired into the benchmark harness for the OpenBLAS/BLIS comparison.

A CodeRabbit review on batch 2 caught that empty (`m==0`) column-major matrices
passed `lda==ldb==0`, violating BLAS's `lda,ldb >= max(1,m)` requirement; fixed by
clamping the leading dimensions to `max(1,m)` (and applied up front in batch 3).

### Board maintenance
On merge, moved the completed tracking issues (**#227**, **#229**) to **Done** on
the MTL5 project board.

## Decisions & Rationale

- **Broad BLAS completeness vs. threading-first.** The dependency analysis showed
  the native factorizations are unblocked/scalar (no L3 building blocks), so
  completing L3 is a genuine prerequisite for fast native factorizations — but the
  user chose full BLAS completeness now; threading (#221) remains the top overall
  roadmap item and is independent.
- **Generic + BLAS-dispatch per operator**, gated on column-major float/double —
  the established `gemv`/`gemm` pattern — so custom number types keep working and
  external BLAS accelerates the standard case.
- **`syrk`/`syr2k` produce the full symmetric result** (both triangles): generic
  computes the lower triangle and mirrors; the BLAS path calls the one-triangle
  kernel and mirrors, so the two paths agree and the result is unsurprising.
- **Scope kept tight per batch**: left-side/no-transpose triangular variants and
  accumulator-policy-awareness for the reduction ops were explicitly deferred.

## Outcome

MTL5 now has a written capability assessment and a tracked expansion roadmap, and
covers the **core BLAS surface end to end**: L1 dot/nrm2/axpy/scal, L2
gemv/ger/symv/trmv/trsv, L3 gemm/trmm/trsm/symm/syrk/syr2k — each with a generic
path, external-BLAS dispatch, tests on both paths (GCC + Clang), and benchmark
coverage against OpenBLAS and BLIS. The natural next step is the roadmap's
highest-value item, on-node threading (#221).

## Issues / PRs

- Merged this session: #219 (assessment), #228 (BLIS + BLAS benchmark, closes #227),
  #230/#231/#232 (BLAS L2/L3, closes #229)
- Roadmap filed: epic #220 with #221–#227; tracking #229
- Board: #227 and #229 moved to Done
