# Native KLU: BTF + Block-wise Gilbert–Peierls LU

## Status

**Planning.** This document specifies a native, header-only implementation of the
KLU algorithm for MTL5. Today MTL5 ships only an *external* KLU binding
(`include/mtl/interface/klu.hpp`, guarded by `MTL5_HAS_KLU`) that wraps
SuiteSparse. There is **no native KLU**: the Block Triangular Form (BTF /
Dulmage–Mendelsohn) layer is unimplemented (Phase 11 in
[`sparse-direct-solvers-design.md`](../sparse-direct-solvers-design.md), marked
"not started"). The per-block engine it needs — a Gilbert–Peierls left-looking
sparse LU with threshold partial pivoting — already exists in
`include/mtl/sparse/factorization/sparse_lu.hpp`.

This plan closes that gap: add BTF, then a thin driver that factors each
diagonal block with the existing GP-LU. The payoff is a header-only KLU that
works on **any** value type (posits, LNS, custom floats) — the external KLU is
hardcoded to `double`.

## Motivation

KLU (Davis & Palamadai Natarajan, *ACM TOMS* Algorithm 907, 2010) is the
direct solver of choice for circuit-simulation matrices (Modified Nodal
Analysis). Those matrices are typically **reducible**: after a row/column
permutation they become block upper triangular with many small diagonal
blocks. KLU exploits this:

1. **Permute to BTF.** Only the diagonal blocks need factorization; the
   off-diagonal coupling is handled by back-substitution. This slashes both
   fill and flops versus a monolithic LU.
2. **Factor each diagonal block** with Gilbert–Peierls left-looking LU +
   partial pivoting.
3. **Block back-substitution** across the block structure to solve.

This is the path called out in [`spice-modernization.md`](spice-modernization.md)
for native SPICE matrix handling without a SuiteSparse dependency.

## What already exists (building blocks)

| Component | Location | Reuse |
|-----------|----------|-------|
| GP left-looking LU (symbolic/numeric split, threshold pivoting) | `sparse/factorization/sparse_lu.hpp` (`lu_symbolic`, `lu_numeric`, `sparse_lu_*`) | Factor each diagonal block |
| Sparse triangular solve + reach | `sparse/factorization/triangular_solve.hpp` | Already used by GP-LU |
| Fill-reducing orderings (AMD, COLAMD, RCM) | `sparse/ordering/` | Per-block column ordering |
| Permutation utilities (invert, compose, apply, symmetric/row/col permute, permute_vector) | `sparse/util/permutation.hpp` | Compose BTF perm with block orderings; permute RHS/solution |
| CSC view, scatter/gather | `sparse/util/csc.hpp`, `sparse/util/scatter.hpp` | Block extraction, sparse accumulation |
| CSR `compressed2D` accessors (`ref_major/minor/data`, `nnz`) | `mat/compressed2D.hpp` | Graph traversal |

**What is missing:** the BTF decomposition itself, the block driver, and tests.

## Algorithm: BTF via Dulmage–Mendelsohn

BTF is two stages on the bipartite/directed graph of the sparsity pattern:

### Stage 1 — Maximum matching (row↔column)
Find a maximum-cardinality matching between rows and columns using
**cheap matching + augmenting paths (Hopcroft–Karp-style DFS)**. For a
structurally nonsingular square matrix this yields a perfect matching that
gives a **zero-free diagonal** under a symmetric reinterpretation (column → its
matched row). Singular/rectangular inputs produce the fine Dulmage–Mendelsohn
decomposition (horizontal/square/vertical sub-blocks); for the square solver
path we require structural nonsingularity and report otherwise.

### Stage 2 — Strongly connected components
With the matching applied so the diagonal is zero-free, build the directed
graph where edge `j → i` exists iff `A(i, matched(j)) ≠ 0`, and find its
**strongly connected components** via **Tarjan's algorithm** (iterative, to
avoid deep recursion on large circuits). The SCCs, in reverse topological
order, are the diagonal blocks; the condensation is a DAG, giving block
**upper** triangular form.

Output: a row permutation `p`, a column permutation `q` (the matching folds
into the columns, so a single symmetric permutation does not suffice in
general), and the block boundary array `blocks` (start index of each diagonal
block), such that `A(p, q)` is block upper triangular with a zero-free
diagonal. This is the `cs_dmperm` interface from Davis.

References: Davis, *Direct Methods for Sparse Linear Systems*, Ch. 7;
Duff & Reid (MC13/MC21); SuiteSparse `BTF`.

## Proposed API (header-only, generic)

New file `include/mtl/sparse/ordering/dulmage_mendelsohn.hpp`:

```cpp
namespace mtl::sparse::ordering {

struct btf_result {
    std::vector<std::size_t> row_perm;   // p[new] = old row
    std::vector<std::size_t> col_perm;   // q[new] = old column
    std::vector<std::size_t> blocks;     // size nblocks+1; block b = [blocks[b], blocks[b+1])
    bool structurally_singular = false;
    std::size_t nblocks() const { return blocks.empty() ? 0 : blocks.size() - 1; }
};

// Maximum matching (row, column) -> matched column for each row, -1 if none.
std::vector<std::ptrdiff_t> maximum_matching(const mat::compressed2D<...>& A);

// Full BTF: matching + Tarjan SCC. Returns row+column permutations and block
// boundaries such that A(row_perm, col_perm) is block upper triangular.
template <typename Value, typename Parameters>
btf_result block_triangular_form(const mat::compressed2D<Value, Parameters>& A);

} // namespace mtl::sparse::ordering
```

New file `include/mtl/sparse/factorization/native_klu.hpp`:

```cpp
namespace mtl::sparse::factorization {

template <typename Value>
struct klu_symbolic {
    ordering::btf_result btf;                 // BTF permutation + block boundaries
    std::vector<lu_symbolic> block_symbolic;  // per-block column ordering (AMD/COLAMD)
};

template <typename Value>
struct klu_numeric {
    klu_symbolic<Value> symbolic;
    std::vector<lu_numeric<Value>> block_numeric;  // GP-LU per diagonal block
    // store permuted off-diagonal coupling for block back-substitution

    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const;      // block back-substitution
};

// Symbolic: BTF, then per-block fill-reducing ordering.
template <typename Value, typename Parameters, typename Ordering = ordering::colamd_ordering>
klu_symbolic<Value> klu_symbolic_analyze(const mat::compressed2D<Value, Parameters>& A,
                                         const Ordering& ord = {});

// Numeric: factor each diagonal block with existing sparse_lu_numeric.
template <typename Value, typename Parameters>
klu_numeric<Value> klu_factor(const mat::compressed2D<Value, Parameters>& A,
                              const klu_symbolic<Value>& sym,
                              Value threshold = Value{1});

// One-shot convenience (mirrors sparse_lu_solve).
template <typename Value, typename Parameters, typename VecX, typename VecB>
void native_klu_solve(const mat::compressed2D<Value, Parameters>& A,
                      VecX& x, const VecB& b);

} // namespace mtl::sparse::factorization
```

Design notes:
- **Reuse, don't reinvent.** Each diagonal block is extracted as a
  `compressed2D` sub-matrix and handed to the *existing* `sparse_lu_symbolic` /
  `sparse_lu_numeric`. KLU adds only the BTF wrapper + block solve loop.
- **Solve = block back-substitution.** With `P·A·Pᵀ` block upper triangular,
  solve blocks bottom-to-top: for block `b`, subtract the already-solved
  contributions from off-diagonal coupling blocks, then `lu_numeric::solve` the
  diagonal block. RHS/solution are permuted with `util::permute_vector` /
  `ipermute_vector`.
- **Generic value type** via `requires OrderedField<Value>`, matching
  `sparse_lu`. No `double` hardcoding.
- **Singular handling**: report `structurally_singular` rather than throwing
  deep in matching; the square solver path requires a perfect matching.

## Implementation phases

| Step | Deliverable | Tests |
|------|-------------|-------|
| 1 | `maximum_matching` (cheap + augmenting DFS) | matching cardinality, zero-free diagonal, singular detection |
| 2 | Tarjan SCC (iterative) on matched graph | known block structures, single-block (irreducible) case |
| 3 | `block_triangular_form` assembling perm + blocks | `P·A·Pᵀ` is block upper triangular; round-trip permutation validity |
| 4 | `klu_symbolic_analyze` / `klu_factor` (block extract + per-block GP-LU) | factor reconstructs each block |
| 5 | `klu_numeric::solve` (block back-substitution) | residuals on reducible matrices |
| 6 | `native_klu_solve` one-shot + dispatch hookup | cross-validate vs `sparse_lu_solve` and vs external `klu_solver` when `MTL5_HAS_KLU` |

## Testing strategy

`tests/unit/sparse/test_dulmage_mendelsohn.cpp`:
- Maximum matching on known patterns (full diagonal, permuted, structurally
  singular → cardinality `< n`).
- BTF on a hand-built reducible matrix → expected block count/sizes.
- Irreducible matrix → exactly one block (BTF is a no-op structurally).

`tests/unit/sparse/test_native_klu.cpp`:
- Solve correctness (`‖Ax − b‖∞` small) on reducible block-triangular matrices.
- Cross-validate native KLU vs native `sparse_lu_solve` on the same systems.
- Cross-validate native KLU vs the external `klu_solver` under `#ifdef MTL5_HAS_KLU`
  (reuse the patterns already in `tests/unit/interface/test_klu.cpp`).
- Generic value type smoke test (e.g. a non-`double` field) to prove the
  templated path compiles and solves.
- Reuse `make_unsym_tridiag` / block-triangular builders from the interface test.

## Files to add

```
include/mtl/sparse/ordering/dulmage_mendelsohn.hpp   # matching + SCC + BTF
include/mtl/sparse/factorization/native_klu.hpp      # block driver + solve
tests/unit/sparse/test_dulmage_mendelsohn.cpp
tests/unit/sparse/test_native_klu.cpp
docs/sparse/native-klu.md                            # user-facing doc (+ FILE_MAP entry)
```

Wire-up: tests are auto-discovered by the `sparse/*.cpp` GLOB in
`tests/unit/CMakeLists.txt` — no CMake edits needed. Add the headers to
`mtl.hpp` once stable. Add the user doc to `FILE_MAP` in
`docs-site/sync-content.mjs`.

## Open questions / risks

- **Matching algorithm choice.** Hopcroft–Karp gives best asymptotics; a
  simpler cheap-matching + DFS-augment (MC21-style) is easier to verify and
  usually sufficient for circuit matrices. Start simple, optimize if profiling
  warrants.
- **Recursion depth.** Tarjan must be iterative — circuit graphs can be large
  and deep; recursive SCC risks stack overflow.
- **Off-diagonal coupling storage.** Decide whether to keep the permuted matrix
  whole and index sub-blocks, or to physically extract coupling blocks. Whole +
  index is lower-memory; extraction is simpler to get correct first.
- **Scaling/refinement.** KLU optionally scales rows and does iterative
  refinement. Out of scope for v1; note as a follow-up.

## References

- Davis, T. A. & Palamadai Natarajan, E. "Algorithm 907: KLU, A Direct Sparse
  Solver for Circuit Simulation Problems." *ACM TOMS* 37(3), 2010.
- Davis, T. A. *Direct Methods for Sparse Linear Systems*, SIAM, 2006 (Ch. 7,
  Dulmage–Mendelsohn & BTF).
- Duff, I. S. & Reid, J. K. MC13/MC21 (matching and block triangularization).
