# Nested Dissection Ordering

`mtl::sparse::ordering::nested_dissection` is a fill-reducing ordering for the
sparse factorizations, next to `amd`, `colamd` and `rcm`. Like `amd`, it works on
the pattern of A + Aᵀ, and it returns `perm` with `perm[new] = old`.

```cpp
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/ordering/nested_dissection.hpp>

auto sym = mtl::sparse::factorization::sparse_cholesky_symbolic(
    A, mtl::sparse::ordering::nested_dissection{});
// or tuned:
//   mtl::sparse::ordering::nested_dissection{/*cutoff=*/64, /*balance=*/0.7,
//       mtl::sparse::ordering::nd_separator::multilevel}
// or the cheaper level-structure separators:
//   mtl::sparse::ordering::nested_dissection{
//       mtl::sparse::ordering::nd_separator::level_structure}
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

Two separator methods share the same recursion. `nd_separator::multilevel` is
the default, and `nd_separator::level_structure` is the cheaper one. Both have
no external dependency, and in both:

- **Components.** Each connected component is ordered separately.
- **Hybrid with AMD.** Below `cutoff` vertices (default 64), a piece is ordered
  with AMD instead of being dissected further. METIS does the same, switching to
  minimum degree at about 200 vertices.
- **No call-stack recursion.** The recursion runs on an explicit work stack, so
  a long path, which has a deep separator tree, cannot overflow the call stack.

### Level structures

This is George's automatic nested dissection, as in SPARSPAK:

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

Level structures are cheap, and asymptotically optimal on grids. On an
irregular graph, though, every level can be much wider than the best
separator.

### Multilevel separators

This is the scheme of METIS (Karypis and Kumar, 1998), in
`multilevel_separator.hpp`:

1. **Coarsen.** Heavy-edge matching, in a random vertex order, collapses
   matched pairs into single vertices. The vertex weights count the original
   vertices, and the edge weights sum the edges each coarse edge replaces.
   Matching repeats until about 100 vertices remain, or until a level shrinks
   the graph by less than 15%.
2. **Initial separator.** On the coarsest graph, grow breadth-first regions from
   several seeds, the first one pseudo-peripheral. Each region holds 30% to 50%
   of the weight, and its boundary becomes a separator. The best balanced
   separator wins.
3. **Uncoarsen and refine.** The labels project to the next finer graph, and
   the separator stays valid. Vertex-separator Fiduccia–Mattheyses then
   improves it:
   - Moving separator vertex v into A pulls v's neighbours in B into the
     separator. The separator therefore shrinks by w(v) minus their weight.
   - A pass moves each vertex at most once. It accepts a bounded run of moves
     that do not improve the separator, so that it can climb out of local
     minima, and then rolls back to the best state it saw.
   - Moves keep the larger part within `balance`.
4. **Thin.** A separator vertex that touches only one side moves to that side.
   Afterwards every separator vertex touches both A and B.

The level-structure separator, refined the same way, is a second candidate, and
the smaller balanced separator wins. On grids, a level structure's lopsided
corner cuts can be shorter than anything the coarse graphs suggest. On irregular
graphs the multilevel separator wins by a wide margin.

The random choices use splitmix64 with a fixed seed, and every priority queue
breaks ties by vertex number. The same matrix therefore gives the same
permutation with every compiler and standard library.

## Measured against AMD

Cholesky nnz(L) and factorization flops (the sum of squared column counts),
relative to AMD, with cutoff 64 and balance 0.7. Each cell gives the multilevel
value, with the level structure's in parentheses:

| 2-D k×k grid | k = 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| nnz(L) | 1.02 (1.04) | 0.88 (0.92) | 0.82 (0.84) | 0.76 (0.78) |
| flops | 0.89 (0.94) | 0.62 (0.69) | 0.58 (0.62) | 0.50 (0.54) |

| 3-D k³ grid | k = 12 | 16 | 20 | 32 |
|---|---|---|---|---|
| nnz(L) | 0.88 (0.95) | 0.83 (0.91) | 0.73 (0.79) | 0.60 (0.64) |
| flops | 0.70 (0.77) | 0.60 (0.68) | 0.47 (0.52) | 0.33 (0.36) |

| Irregular graph | n | nnz(L) | flops |
|---|---|---|---|
| 256² grid with 60 round holes | 44 243 | 1.06 (1.13) | 0.96 (1.16) |
| random geometric, 2-D, mean degree 8 | 50 000 | 1.17 (2.05) | 1.38 (6.12) |
| random geometric, 3-D, mean degree 12 | 30 000 | 0.70 (1.44) | 0.44 (1.53) |

On a 2-D random geometric graph of this size, AMD still gives the least fill.
The multilevel ordering cuts the level structure's fill there by 43% and its
flops by 77%.

Ordering time:

| Graph | AMD | level structure | multilevel |
|---|---|---|---|
| 256² grid | 0.01 s | 0.04 s | 0.45 s |
| 512² grid | 0.06 s | 0.21 s | 2.0 s |
| 3-D random geometric, n = 30 000 | 0.03 s | 0.09 s | 0.46 s |

The growth rates of the level-structure ordering over 2-D grids from k = 128 to
512, as powers of n = k²:

| | fill | flops |
|---|---|---|
| nested dissection | n^1.12 | n^1.50, the optimal rate |
| AMD | n^1.18 | n^1.59 |
| n log n, for reference | about n^1.09 | |

When to use which:

- **Small 2-D problems:** AMD is as good or better. Up to a few thousand
  unknowns, the greedy local choice beats nested dissection's coarse
  separators.
- **Large 2-D problems and 3-D at any size:** nested dissection wins, and the
  savings in flops are larger than the savings in fill, because the dense
  separator blocks are where the arithmetic is.
- **Irregular graphs:** use the multilevel separators, the default. Level
  structures can be several times worse there.
- **When the ordering time matters,** for example when the same pattern is
  factored only once and is not large, `nd_separator::level_structure` is
  4 to 10 times faster, and its fill on grids is within about 8%.

## References

- G. Karypis and V. Kumar, "A Fast and High Quality Multilevel Scheme for
  Partitioning Irregular Graphs", SIAM J. Sci. Comput. 20(1), 1998.
- B. Hendrickson and R. Leland, "A Multilevel Algorithm for Partitioning
  Graphs", Supercomputing '95.
- C. Ashcraft and J. W. H. Liu, "Robust Ordering of Sparse Matrices using
  Multisection", SIAM J. Matrix Anal. Appl. 19(3), 1998.
- C. M. Fiduccia and R. M. Mattheyses, "A Linear-Time Heuristic for Improving
  Network Partitions", 19th Design Automation Conference, 1982.

- A. George, "Nested Dissection of a Regular Finite Element Mesh", SIAM J. Numer.
  Anal. 10(2), 1973.
- A. George and J. W. H. Liu, "An Implementation of a Pseudoperipheral Node
  Finder", ACM TOMS 5(3), 1979.
- A. George and J. W. H. Liu, *Computer Solution of Large Sparse Positive
  Definite Systems*, Prentice-Hall, 1981, chapter 8.
- R. J. Lipton, D. J. Rose and R. E. Tarjan, "Generalized Nested Dissection",
  SIAM J. Numer. Anal. 16(2), 1979.
