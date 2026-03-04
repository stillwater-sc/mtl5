# Assessment

## Three documents created in docs/:

  1. mtl5-status-assessment.md — Current state: 88 implemented, 35 stubs, ~200 unported. All 16 tests passing.
  2. mtl5-stub-completion-plan.md — Phases 5–9 to fill all 35 stubs:
    - Phase 5 (Critical): Triangular solvers → LU → QR → Cholesky → ILU(0)/IC(0)
    - Phase 6: Eigenvalues, SVD, TFQMR/QMR/IDR(s), Kronecker
    - Phase 7: COO, ELLPACK, views, Matrix Market I/O
    - Phase 8 (Optional): Expression templates
    - Phase 9: Recursion, sparse vector, BLAS/LAPACK bindings
  3. mtl5-full-porting-plan.md — Phases 10–14 for unported MTL4 features:
    - Phase 10: Triangular views, permutation, block diagonal
    - Phase 11: Vector utilities, reordering
    - Phase 12: Setup helpers (Laplacian, Poisson), extended I/O
    - Phase 13: Advanced ITL (CGS, ILUT, multigrid)
    - Phase 14: BLAS/LAPACK external bindings
    - ~140 MTL4 files eliminated — replaced by C++20 concepts, if constexpr, etc.

Key insight: Of MTL4's 367 files, ~140 are eliminated by C++20, leaving ~95 files of real work (~10,000 lines including tests) across Phases 5–14.

