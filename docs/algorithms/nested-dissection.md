# Nested Dissection Ordering

`mtl::sparse::ordering::nested_dissection` is a fill-reducing ordering for the
sparse factorizations, next to `amd`, `colamd` and `rcm`. Like `amd`, it works on
the pattern of A + Aᵀ, and it returns `perm` with `perm[new] = old`.

```cpp
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/ordering/nested_dissection.hpp>

auto sym = mtl::sparse::factorization::sparse_cholesky_symbolic(
    A, mtl::sparse::ordering::nested_dissection{});
// or tuned: nested_dissection{/*cutoff=*/64, /*balance=*/0.7}
```

## The idea

Elimination in graph terms: eliminating vertex v connects all of v's remaining
neighbours to each other, and every new edge is a fill-in entry. Nested
dissection (George, 1973) uses that rule globally:

1. Find a small set of vertices S, the **separator**, whose removal splits the
   graph into parts A and B with no edges between them.
2. Number A first, then B, then S.
3. Recurse on A and B.

A vertex of A only has neighbours in A and S. Eliminating it therefore creates
fill inside A and S and never between A and B. The fill is confined to the
diagonal blocks and the separator rows, and only the separator's own block
becomes dense. On a k × k grid the separators have O(k) vertices. That gives
O(n log n) fill and O(n^1.5) factorization work, both optimal for planar graphs
(Lipton, Rose and Tarjan, 1979).

A and B are also factored independently, so the separator tree is the task
tree a parallel factorization wants.

## How MTL5 finds separators

MTL5 uses **level structures**, George's automatic nested dissection as in
SPARSPAK. It has no external dependency:

- **Components.** Each connected component is ordered separately.
- **A pseudo-peripheral vertex** (George and Liu, 1979). Run a breadth-first
  search, restart it from the minimum-degree vertex of the deepest level, and
  stop once the number of levels stops growing.
- **The separator** is one level of that search. Every edge joins the same or
  adjacent levels, so any level separates the levels before it from those
  after. The chosen level is the narrowest one whose larger side holds at most
  `balance` (default 0.7) of the remaining vertices. Balance alone picks wide
  levels, and width alone picks lopsided splits.
- **Thinning.** A separator vertex with no neighbour in the next level touches
  only A and S, so it moves to A.
- **Hybrid with AMD.** Below `cutoff` vertices (default 64), a piece is ordered
  with AMD instead of being dissected further. METIS does the same, switching to
  minimum degree at about 200 vertices.
- **No call-stack recursion.** The recursion runs on an explicit work stack, so
  a long path, which has a deep separator tree, cannot overflow the call stack.

## Measured against AMD

Cholesky nnz(L) and factorization flops (the sum of squared column counts),
relative to AMD, with the defaults:

| 2-D k×k grid | k = 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| nnz(L) | 1.13 | 1.04 | 0.92 | 0.84 | 0.78 |
| flops | 1.15 | 0.94 | 0.69 | 0.62 | 0.54 |

| 3-D k³ grid | k = 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|
| nnz(L) | 0.95 | 0.91 | 0.79 | 0.78 | 0.68 | 0.64 |
| flops | 0.77 | 0.68 | 0.52 | 0.53 | 0.40 | 0.36 |

The growth rates tell the same story. Over 2-D grids from k = 128 to 512, as
powers of n = k²:

| | fill | flops |
|---|---|---|
| nested dissection | n^1.12 | n^1.50, the optimal rate |
| AMD | n^1.18 | n^1.59 |
| n log n, for reference | about n^1.09 | |

When to use which:

- **Small 2-D problems:** AMD is as good or better. Up to a few thousand
  unknowns, the greedy local choice beats the coarse level-structure
  separators.
- **Large 2-D problems and 3-D at any size:** nested dissection wins, and the
  savings in flops are larger than the savings in fill, because the dense
  separator blocks are where the arithmetic is.
- **Irregular graphs:** level-structure separators can be wide. A multilevel
  partitioner, with coarsening by matching and Fiduccia–Mattheyses refinement
  as in METIS, finds better ones. That is a possible follow-up.

## References

- A. George, "Nested Dissection of a Regular Finite Element Mesh", SIAM J. Numer.
  Anal. 10(2), 1973.
- A. George and J. W. H. Liu, "An Implementation of a Pseudoperipheral Node
  Finder", ACM TOMS 5(3), 1979.
- A. George and J. W. H. Liu, *Computer Solution of Large Sparse Positive
  Definite Systems*, Prentice-Hall, 1981, chapter 8.
- R. J. Lipton, D. J. Rose and R. E. Tarjan, "Generalized Nested Dissection",
  SIAM J. Numer. Anal. 16(2), 1979.
