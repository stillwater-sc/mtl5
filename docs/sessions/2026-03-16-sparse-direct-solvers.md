# Session: Sparse Direct Solvers for MTL5

**Date**: 2026-03-16
**Duration**: Full day session
**Participants**: Theodore Omtzigt, Claude Code

## Objective

Design and implement native sparse direct solvers for MTL5, inspired by Timothy Davis's *Direct Methods for Sparse Linear Systems* and the CSparse/SuiteSparse ecosystem. Establish a conventional commit workflow with CodeRabbit AI review.

## Context

MTL5's iterative solver library (ITL) was inspired by Anne Greenbaum's *Iterative Methods for Solving Linear Systems*. The sparse direct solver side — rooted in Davis's work and the SuiteSparse ecosystem (UMFPACK, CHOLMOD, KLU, SPQR, SuperLU) — was absent from MTL5 except for an existing UMFPACK binding. This session rectified that.

## Work Completed

### Research and Design
- Researched CSparse architecture, SuiteSparse ecosystem, SuperLU, KLU algorithms
- Studied Gilbert-Peierls sparse triangular solve, elimination trees, AMD/COLAMD orderings
- Produced architectural proposal covering 7 implementation phases
- Committed design document: `docs/sparse-direct-solvers-design.md`

### Phase 1: Infrastructure (PR merged directly to main)
| Component | File | Lines |
|-----------|------|-------|
| Permutation utilities | `sparse/util/permutation.hpp` | ~185 |
| CSC format + conversion | `sparse/util/csc.hpp` | ~160 |
| Sparse accumulator | `sparse/util/scatter.hpp` | ~95 |
| Elimination tree | `sparse/analysis/elimination_tree.hpp` | ~130 |
| Postorder traversal | `sparse/analysis/postorder.hpp` | ~95 |
| Triangular solve | `sparse/factorization/triangular_solve.hpp` | ~220 |
| RCM ordering | `sparse/ordering/rcm.hpp` | ~100 |
| Ordering concepts | `sparse/ordering/ordering_concepts.hpp` | ~30 |
| 7 test files | `tests/unit/sparse/test_*.cpp` | ~500 |

### Phase 2: Sparse Cholesky (PR #3, CodeRabbit reviewed)
- Up-looking LL^T with symbolic/numeric separation
- Pluggable orderings, automatic permutation in solve
- 12 test cases; 6 CodeRabbit findings resolved (include hygiene, runtime validation, buffer overflow guard, ordering validation)

### Phase 3: Sparse LU (PR #4, CodeRabbit reviewed)
- Left-looking PA=LU with threshold partial pivoting
- `requires OrderedField<Value>` concept constraint
- Sparse workspace tracking (not dense clearing)
- 12 test cases; 4 CodeRabbit findings resolved (concept constraint, permutation validation, sparse clearing, threshold test divergence)

### Phase 4: Sparse QR (PR #6, merged)
- Householder QR for square and overdetermined systems
- Golub & Van Loan convention: H = I - beta*v*v^T
- Least-squares solve: min ||Ax - b||
- 11 test cases

### Phase 5: AMD/COLAMD Orderings (PR #7, merged)
- AMD: greedy minimum degree with fill-in edge tracking
- COLAMD: AMD on A^T*A column intersection graph
- Both satisfy `FillReducingOrdering` concept
- 14 test cases

### Phase 6: External Solver Interfaces (PR #8, merged)
- SuperLU, KLU, CHOLMOD, SPQR — RAII wrappers following UMFPACK pattern
- CMake find logic, `#ifdef` guards, CRS-to-CCS conversion
- Tested with `libsuitesparse-dev` and `libsuperlu-dev` installed locally
- Initial version had hand-written C declarations; fixed to use actual library headers

### Development Workflow (PR #2, merged)
- `.coderabbit.yaml` with C++20-specific review instructions
- Branch protection on `main` (PRs required, CI must pass)
- Conventional commit conventions documented in `CLAUDE.md`

### Position Paper (PR #9, merged)
- `docs/position-mixed-precision-acceleration.md`
- Vision: MTL5 + Universal for accelerated mixed-precision linear algebra
- Three-layer architecture: application → algebra (MTL5) → arithmetic (Universal) → hardware
- Engineering roadmap from current state to KPU integration

### Phase 8: Exhaustive Validation (PR #10, merged)
- 35 new test cases, 243 assertions
- Cross-solver consistency: Cholesky/LU/QR agreement up to n=50
- UMFPACK vs native solver comparison
- Scale testing up to 100x100
- Edge cases: block diagonal, permutation matrices, identity, rectangular

## Metrics

| Metric | Value |
|--------|-------|
| PRs created | 9 (PR #2-10) |
| PRs merged | 9 |
| New source files | 16 headers + 14 test files |
| New lines of code | ~6,500 |
| Total tests | 90 (up from ~76) |
| Test assertions | ~450+ |
| CodeRabbit findings resolved | 10 |
| External libraries integrated | 5 (UMFPACK, SuperLU, KLU, CHOLMOD, SPQR) |

## Remaining Work (from design document)

| Phase | Status | Description |
|-------|--------|-------------|
| 1-6 | Done | Infrastructure, Cholesky, LU, QR, AMD/COLAMD, external interfaces |
| 7 | Not started | Unified dispatch in `operation/` (auto-select native vs SuiteSparse) |
| 8 | Done | Exhaustive validation |
| — | Not started | Dulmage-Mendelsohn / BTF decomposition |
| — | Not started | Mixed-precision infrastructure (`mixed_precision_refine()`) |
| — | Not started | Universal integration tests (posit/cfloat/lns through solvers) |

## Decisions Made

1. **Left-looking algorithms** (not multifrontal) for native solvers — simpler, educational, type-generic
2. **Symbolic/numeric phase separation** throughout — enables pattern reuse and mixed-precision
3. **SuiteSparse as complement, not replacement** — production CPU backend for float/double; native solvers for custom types
4. **Conventional commits + CodeRabbit** — every PR gets AI review, findings must be resolved before merge
5. **`requires OrderedField<Value>`** on LU — pivoting needs ordered comparisons, incompatible with complex/integer
6. **Use actual library headers** for external interfaces — hand-written `extern "C"` declarations break with real installations
