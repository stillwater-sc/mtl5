# Native Supernodal LDLᵀ

MTL5 provides a native, header-only **supernodal LDLᵀ** factorization for
symmetric matrices: `A = PᵀL D Lᵀ P`, with `L` unit lower triangular and `D`
diagonal. It complements the scalar `sparse_ldlt` (up-looking, column-by-column)
by grouping columns that share a nonzero structure into **dense panels
(supernodes)** and performing the bulk of the arithmetic as dense block updates.

This matters for MTL5's mixed-precision goal: the dense block update is exactly
where you want to **store the factor in a low precision and accumulate in a
higher precision**. The kernel exposes that precision boundary directly through
`mtl::math::accumulator_traits`.

## Why LDLᵀ (square-root free)

- No `sqrt`, which is friendlier to low precision and to custom number types
  (posit/LNS) where `sqrt` is costly or less accurate.
- The diagonal `D` is a natural place to study precision.
- Symmetric ⇒ no pivoting, so supernodes follow directly from the elimination
  tree — the lowest-risk place to land supernodal + mixed-precision machinery.

## API

```cpp
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
using namespace mtl::sparse::factorization;

// One-shot solve (A symmetric, b/x dense_vector):
supernodal_ldlt_solve(A, x, b, mtl::sparse::ordering::amd{});

// Reusable symbolic + numeric:
auto sym = supernodal_ldlt_symbolic(A, mtl::sparse::ordering::amd{});
auto fac = supernodal_ldlt_numeric(A, sym);   // factor (CSC L + diagonal D)
fac.solve(x, b);
```

The symbolic phase fuses the fill-reducing ordering with a postorder of the
elimination tree (so a supernode's columns are contiguous) into a single
permutation `sperm`, then detects fundamental supernodes. The numeric phase is
left-looking over supernodes; it emits a **standard CSC factor** so the solve and
the generic iterative-refinement loop are reused unchanged.

## Mixed precision

`supernodal_ldlt_numeric` takes a third template argument, the accumulator type:

```cpp
// Factor stored in float, every accumulation carried in double:
auto fac = supernodal_ldlt_numeric<float, decltype(A)::param_type, double>(A, sym);
```

The panel is touched **only** through `accumulator_traits`' `add_product` /
`value` / `clear`, so an add-only super-accumulator (e.g. a Universal `quire`)
works as well as a wider IEEE type. Each factor entry is rounded to the storage
precision exactly once, when it is finalized (single-rounding semantics). The
default accumulator equals the storage type and is byte-identical to a plain
factorization.

### Iterative refinement

`supernodal_ldlt_solve_refined` factors in a low precision and recovers accuracy
with a high-precision residual via the generic `iterative_refine` loop:

```cpp
mtl::sparse::refine_options opt; opt.rel_tol = 1e-12;
// factor in float (accumulate in double), refine the residual in double:
supernodal_ldlt_solve_refined<float, double>(A, x, b,
                                             mtl::sparse::ordering::amd{}, opt);
```

## Validation

Correctness is checked against the scalar `sparse_ldlt` oracle and (when built
with SuiteSparse) the supernodal CHOLMOD binding. Tests cover symbolic supernode
detection (diagonal ⇒ all singletons, dense ⇒ one panel), numeric agreement on
tridiagonal / 2-D Laplacian / dense SPD systems, the mixed-precision accumulator
boundary, iterative refinement, and edge cases (1×1, diagonal, zero pivot).

## Scope and limitations

- Only **fundamental** supernodes are formed; relaxed amalgamation (the `relax`
  knob) is reserved for a later milestone.
- The triangular **solve** reuses the CSC kernels; a blocked supernodal solve is
  a future optimization.
- Unsymmetric supernodal LU (SuperLU-style, pivoting × supernodes) is a separate,
  later effort — see the [sparse direct solver design](../design/sparse-direct-solvers.md).

## References

- Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006.
- Ng & Peyton, "Block sparse Cholesky algorithms on advanced uniprocessor
  computers," *SIAM J. Sci. Comput.*, 14(5), 1993.
- Liu, Ng, Peyton, "On finding supernodes for sparse matrix computations,"
  *SIAM J. Matrix Anal. Appl.*, 14(1), 1993.
