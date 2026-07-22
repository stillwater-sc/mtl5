# Session: matrix/vector/tensor property & predicate module

**Date**: 2026-07-21
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Deliver the property/predicate module (#244) end to end: a cohesive set of
runtime "is this matrix/vector/tensor X?" queries — the usability layer that
packages the library's existing numerical primitives (Cholesky, LU, SVD,
eigen, norms) as boolean/scalar answers. Ship it as a sequence of small,
reviewable PRs, merging each once CI is green.

## Context

An earlier property-function audit found MTL5 had **no** runtime predicates:
no `is_spd`, `is_singular`, `condition_number`, `determinant`, `is_finite`, etc.
The building blocks all existed, but a user asking "is this matrix SPD /
singular / well-conditioned?" had to call a factorization and interpret its
return code. Issue #244 captured this as a four-phase module; this session
implemented all four phases plus the optional follow-up.

The day opened by closing out the prior threading epic — merging the on-node
threading documentation PR (#243, the reference page + the multi-core scaling
case study) — before starting #244.

## Work Completed

### Batch 1 — structural + vector predicates (#245, merged)
`operation/matrix_properties.hpp` and `operation/vector_properties.hpp`: the
cheapest, highest-use tier (O(n) dense / O(nnz) sparse, no factorization).
- Matrix: `is_square`, `is_empty`, `is_symmetric`, `is_hermitian`,
  `is_upper/lower/is_triangular`, `is_diagonal`, `is_banded`,
  `is_diagonally_dominant`.
- Vector: `is_zero`, `is_finite`/`has_nan`/`has_inf`, `is_normalized`/`is_unit`,
  `is_orthogonal_to`.

Established the module's **tolerance policy**: structural checks take an absolute
`tol` defaulting to 0 (exact-as-constructed); norm-based checks default to a
relative `128·eps`. CodeRabbit flagged that `abs(dev) > tol` accepts NaN (unordered
compares false); fixed to the **NaN-safe `!(abs(dev) <= tol)`** form throughout,
with NaN regression tests.

### Batch 2 — factorization-backed predicates (#246, merged)
`operation/factorization_properties.hpp`, wrapping the destructive dense
factorizations on a **copy** so the caller's matrix is unchanged (O(n³)):
- `is_spd`/`is_positive_definite` — symmetric (within `sym_tol`) + Cholesky exists.
- `is_singular`/`is_nonsingular`/`is_invertible` — LU zero pivot, or smallest
  `|U(k,k)| <= tol`.
- `determinant` — `sign(perm) · ∏ U(k,k)`; exactly 0 when singular.

Edge conventions: non-square → not-SPD / singular / det 0; empty 0×0 →
nonsingular, det 1. Tests use exact Pascal (SPD, det = 1) and randspd
(det = eigenvalue product).

### Batch 3 — spectral / condition / rank (#247, merged)
`operation/spectral_properties.hpp`, wrapping the dense SVD and eigensolver:
`spectral_radius`, `condition_number` (σ_max/σ_min; +∞ when rank-deficient),
`rcond`, `numerical_rank` (default cutoff `max(m,n)·eps·σ_max`), `nullity`.

Verified against **both** the in-house and a `-DMTL5_WITH_LAPACK=ON` build,
which mattered: the in-house SVD yields a tiny-but-nonzero σ_min for a singular
matrix, so the rank-deficient condition-number is tested as "very large or
infinite" and rank-deficient counts pass an explicit `tol`. CodeRabbit review:
added a direct `<algorithm>` include, switched local dimensions/indices to
`std::size_t`, and added a square-matrix precondition to `spectral_radius`
before its empty-matrix fast path.

### Batch 4 — rank-2 tensor predicates (#248, merged)
`tensor/properties.hpp`: `is_symmetric` and `is_antisymmetric` (the latter with
an explicit zero-diagonal check), closing the gap where the
`symmetric_tensor`/`antisymmetric_tensor` storage *types* existed but no
predicate could test an arbitrary rank-2 tensor. Same NaN-safe, exact-default
policy as the matrix predicates.

### Follow-up — orthogonality + inertia (#249, merged)
The optional extras named on #244, beyond the four planned batches:
- `is_orthogonal`/`is_unitary` (`AᴴA == I`) and `is_normal` (`AAᴴ == AᴴA`),
  product-based (O(n³)) with a **scale-aware** default tolerance
  (~`128·n·eps·max|A|²`, since the residuals grow with matrix magnitude).
- `inertia` → `(positive, negative, zero)` eigenvalue-sign counts of a symmetric
  matrix, and `is_indefinite`.

## Decisions & Rationale

- **Inertia via the symmetric eigensolver, not Bunch–Kaufman D-blocks.** The
  roadmap suggested reading pivot signs from the BK `LDLᵀ`. By **Sylvester's law
  of inertia** the `(n₊, n₋, n₀)` triple is a congruence invariant, so both give
  the same answer — but the eigenvalue route is robust for singular/semidefinite
  inputs (a zero eigenvalue is classified by tolerance), whereas the BK factor
  early-returns on a zero pivot and would need fragile 2×2-block sign extraction.
- **Two tolerance families, documented per predicate.** Exact-by-default and
  NaN-safe for structural checks (you usually built the structure exactly);
  relative/scale-aware for norm-, factorization-, and product-backed checks
  (a computed quantity is rarely bit-exact). A negative-`tol` sentinel selects the
  auto default for `numerical_rank`/`inertia`/orthogonality.
- **Factorization predicates operate on a copy.** The `cholesky_factor`/
  `lu_factor` routines are destructive; the predicates copy so a query never
  mutates its argument.
- **Verify on both SVD/eigen backends.** Every spectral test was run against the
  in-house and LAPACK builds, because their accuracy floors differ and the
  singular-case thresholds depend on it.

## Outcome

The full #244 module — 21 predicate/scalar queries across matrices, vectors, and
rank-2 tensors — is merged to `main` and issue #244 is closed. No new
dependencies; everything reuses `cholesky`/`lu`/`svd`/`eigenvalue`/`norms`. All
five PRs were green on the 8-platform matrix (+ LAPACK + regression) and passed
CodeRabbit review.

Candidate follow-ups remaining on the roadmap: #236 (native L2/L3 kernels), #222
(SIMD portability), #223 (memory-space abstraction), and the #221 threading
tail (level-scheduled triangular solve, transposed SpMV).

## Issues / PRs

- #244 — property/predicate module epic (**closed**, delivered)
- #245 — batch 1: structural + vector predicates (merged)
- #246 — batch 2: factorization-backed predicates (merged)
- #247 — batch 3: spectral / condition / rank (merged)
- #248 — batch 4: rank-2 tensor predicates (merged)
- #249 — follow-up: orthogonality + inertia (merged)
- #243 — on-node threading documentation, closing epic #221 (merged at start of day)
