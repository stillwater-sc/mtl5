# sparse_vector: architecture of the sparse vector

How `mtl::vec::sparse_vector` is put together, and *why*. It sits at the corner
where two axes of MTL5's container design meet: it is **sparse** like
[`compressed2D`](compressed2d-architecture.md) and a **vector** like
[`dense_vector`](dense-vector-architecture.md). It borrows the vector *shape*
semantics from the former and the sorted-index *storage* technique from the
latter — and then does one thing neither sparse matrix does: it lets you mutate
it in place, one element at a time. That single relaxation is the most
interesting decision in the file, and it is affordable for exactly the reason it
is not affordable for a matrix.

The implementation is ~200 lines in
[`include/mtl/vec/sparse_vector.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/sparse_vector.hpp).

---

## The representation: dual sorted arrays

A `sparse_vector` is two parallel `std::vector`s plus a logical length — in effect
a **single compressed row**:

```text
sparse_vector<Value, Parameters = parameters<>>
   dim_     : size_type                  logical length (number of positions)
   indices_ : vector<size_type>          positions of the stored nonzeros, SORTED ascending
   values_  : vector<Value>              the value at each stored position (parallel)

   Example  length 8, entries at 1, 4, 5:
            dim_     = 8
            indices_ = [ 1, 4, 5 ]
            values_  = [ a, b, c ]        -> logical vector [ 0 a 0 0 b c 0 0 ]
```

Only the nonzeros are stored, and `indices_` is kept **sorted**. That is the same
two-array, sorted-within-a-run layout `compressed2D` uses for each of its rows —
`sparse_vector` is essentially one such row promoted to a standalone type. `nnz()`
is `indices_.size()`; `size()` is the logical `dim_`.

---

## Decision 1 — sorted indices, O(log n) access via `lower_bound`

Because `indices_` is sorted, every position query is a binary search:

```cpp
auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
if (it != indices_.end() && *it == i) return values_[it - indices_.begin()];
```

`operator()`, `exists`, and `insert` all pivot on this one `lower_bound`. Read
access is O(log nnz), the same complexity as a `compressed2D` row lookup, and the
sorted invariant is what buys it. Keeping the arrays sorted (rather than
append-and-sort-later, COO style) is the right call here because a
`sparse_vector` is used *interactively* — read, write, read again — not assembled
once and frozen, so every access wants to be fast, not just the post-freeze ones.

---

## Decision 2 — directly mutable: the read/build split, relaxed

This is the decision that separates `sparse_vector` from every sparse *matrix* in
the library. `compressed2D` forbids per-element insertion; you assemble it through
an [inserter or COO](compressed2d-architecture.md) and then freeze it, precisely
because inserting into the middle of a CSR row shifts the rest of the arrays —
O(nnz) per insert, O(nnz²) to build a whole matrix. `sparse_vector` does the
opposite: it offers `insert`, a mutating `operator[]`, `clear`, and `crop`, and
its `insert` shifts the arrays exactly as CSR insertion would:

```cpp
void insert(size_type i, const Value& v) {
    auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
    auto pos = it - indices_.begin();
    if (it != indices_.end() && *it == i) values_[pos] = v;          // overwrite
    else { indices_.insert(indices_.begin()+pos, i);                 // shift-insert
           values_.insert(values_.begin()+pos, v); }
}
```

**Why the same O(nnz) shift that is banned on a matrix is fine here.** A
`sparse_vector` is *one* row's worth of data. The insert shifts within a single
short array — O(nnz of this vector) — not across the entire nonzero structure of
an n×n matrix. The pathology `compressed2D` avoids is *assembling* a matrix one
element at a time (O(nnz²) over millions of entries); building a single sparse
vector incrementally is O(nnz²) in that one vector's modest nnz, which is a cost
you can pay. So the read/build split that the matrix container enforces is
deliberately **relaxed** for the vector: the same object is both the builder and
the reader, because at vector scale it can be.

---

## Decision 3 — `operator()` reads, `operator[]` mutates

The two subscript spellings mean *different* things here — a deliberate contrast
with `dense_vector`, where `[]` is just an alias for `()`:

```cpp
value_type operator()(size_type i) const;   // READ: returns the value, or value_type{} if absent
reference   operator[](size_type i);        // WRITE: returns a ref, INSERTING a zero if absent
```

- `operator()` is `const` and **non-mutating**. A miss returns a
  default-constructed `value_type{}` (the implicit zero); the vector is unchanged.
  This is the safe read used in expressions and by generic code.
- `operator[]` is non-`const` and **materializes on access**: a miss inserts a
  zero entry at the sorted position and returns a reference to it, exactly like
  `std::map::operator[]`. `v[i] = x` therefore works and grows the structure.

Splitting the two lets a caller read without ever accidentally creating fill (use
`()`), and write through a reference when they mean to (use `[]`). Conflating them
— as the dense vector safely does, since a dense vector has every element already
— would make an innocent read of an absent index silently insert an explicit
zero, the classic `std::map` footgun. Keeping `()` read-only is the guard against
it. (One small divergence from the rest of the family: `sparse_vector` uses
`value_type{}` for the implicit zero rather than the `math::zero<T>()` the dense
and matrix containers use; for standard and most custom types these coincide.)

---

## Decision 4 — `insert` overwrites, and `crop` truncates by magnitude

Two mutators carry sparse-vector-specific policy:

- **`insert` overwrites.** A second `insert` at an existing index replaces the
  value; it does not accumulate. This is a narrower contract than the matrix
  inserter's `update_store`/`update_plus` choice or COO's always-sum — a
  `sparse_vector` is a live object you edit, so last-write-wins is the natural
  semantics, and no policy parameter is offered.
- **Indices are checked when they are stored** (#830). `insert` and
  `operator[]`, which store an index, check it in every build and throw
  `std::out_of_range` through `detail::throw_or_abort` (an abort under
  `-fno-exceptions`). This is the "check once" policy of #825 and #826, but
  per insertion, because a `sparse_vector` has no finalize step to check once
  at; the cost is one comparison next to the O(log nnz) search. `operator()`,
  a read, checks only under `bounds_checking`, like the other containers'
  element access. A signed `size_type` rejects negative indices too.
- **`crop(threshold)` drops small entries.** It compacts the arrays in place,
  keeping the entries whose magnitude is not below the threshold:

  ```cpp
  void crop(const magnitude_t<Value>& threshold);
  for (k ...)
      if (!(abs(values_[k]) < threshold)) { keep index/value }
  ```

  The threshold is a magnitude, so `crop` works for `std::complex` values,
  where comparing `abs(z)` with a complex threshold did not compile. A NaN
  entry is kept: it is not below any threshold, and silently dropping it
  would hide the NaN (#830). Before #830 the test was `abs(value) >= threshold`,
  which dropped NaNs, and each call allocated two new arrays.

  This is a genuinely sparse-vector operation with no dense or matrix analog:
  numerical **truncation** to keep a vector sparse under repeated updates — the
  drop-tolerance step at the heart of incomplete factorizations (ILUT), sparse
  approximate inverses, and truncated iterative methods. `std::abs` is called
  unqualified so ADL finds the right magnitude for custom number types.

---

## Decision 5 — iteration walks the nonzeros, not the length

`sparse_vector` exposes a `const_iterator` that yields `std::pair<index, value>`
over the **stored** entries:

```cpp
for (auto [i, v] : sv)   // visits only the nnz nonzeros, in index order
    ...
```

Iteration is over `nnz` entries, not `dim_` positions — the whole point of a
sparse representation is to skip the zeros, and the iterator makes that the
natural traversal. It is an `input_iterator` producing `(index, value)` pairs by
value (the pair is synthesized from the two parallel arrays on dereference), which
is enough for the range-based consumption sparse kernels need. `indices()` /
`values()` expose the raw arrays for code that wants them directly.

---

## Shape and dispatch: a vector shape with a sparse category

`sparse_vector` classifies exactly like `dense_vector` on the **shape** axis and
like the sparse matrices on the **category** axis — which is precisely its place
in the type system:

```cpp
template <...> struct traits::category<sparse_vector<...>> { using type = tag::sparse; };
template <...> struct ashape<sparse_vector<...>> {
    using type = std::conditional_t<col_major, cvec<Value>, rvec<Value>>;   // like dense_vector
};
```

- **ashape = `cvec`/`rvec` by orientation**, and `num_rows()`/`num_cols()` are
  orientation-aware, identical in spirit to
  [`dense_vector`](dense-vector-architecture.md): a `sparse_vector` is a column
  vector by default and presents an *n × 1* (or *1 × n*) face.
- **category = `tag::sparse`**, so operations select sparse code paths for it, as
  they do for the sparse matrices.

Note a difference from the sparse *matrices*: `compressed2D` and `ell_matrix`
carry a `static_assert` rejecting `col_major`, because their `orientation` tag was
inert and a `col_major` instantiation silently meant the transpose (the #355
bug). Here orientation is **live** — it genuinely selects `cvec` vs `rvec` — so
both orientations are valid and meaningful, and no such guard is needed. The tag
is doing real work, which is exactly the condition under which it is safe.

---

## What it deliberately does *not* use

`sparse_vector` is a lean, purpose-built container, and several mechanisms the
dense types rely on are simply absent:

- **No `contiguous_memory_block`.** It holds plain `std::vector`s (which must grow
  and shift), not the aligned stack-or-heap block. Alignment and stack storage are
  meaningless for a shifting sparse array.
- **No expression-template assignment.** There is no concept-constrained `Expr`
  constructor or `parallel_ewise` sweep as in `dense_vector`; a sparse vector is
  edited through its mutators, not materialized from a fused expression tree.
- **No fixed-size or stack storage.** It reads only `orientation` and `size_type`
  from the `Parameters` bundle; the `dimensions_type` and `storage` axes are
  inert here — a `sparse_vector` is always dynamic and heap-backed. (Unlike the
  #355 case, an inert axis here manufactures no wrong answer; a fixed/stack
  request is simply ignored, and the natural sparse vector has no compile-time nnz
  to fix anyway.)

The container is intentionally smaller than `dense_vector`: it solves the narrow
problem of an editable sorted sparse 1-D array, and leaves fusion, alignment, and
compile-time sizing to the dense side where they pay off.

---

## Where it sits: the {dense, sparse} × {matrix, vector} grid

`sparse_vector` completes a 2×2 of core containers, each combining a density with
a rank:

| | matrix | vector |
|---|---|---|
| **dense** | [`dense2D`](dense2d-architecture.md) | [`dense_vector`](dense-vector-architecture.md) |
| **sparse** | [`compressed2D`](compressed2d-architecture.md) / [`coordinate2D`](coordinate2d-architecture.md) / [`ell_matrix`](ell-matrix-architecture.md) | **`sparse_vector`** |

It inherits its **shape** behavior (orientation → `cvec`/`rvec`, matrix-shaped
`num_rows`/`num_cols`) from the vector column and its **storage** technique
(sorted index/value arrays, `lower_bound` lookup) from the sparse row — then
parts ways with the sparse matrices on mutability, because a single vector is
small enough to edit in place. That combination is the whole design.

---

## File map

| File | Role |
|---|---|
| [`vec/sparse_vector.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/sparse_vector.hpp) | the container: dual sorted arrays, `()`/`[]`, `insert`/`crop`, nonzero iterator, `rvec`/`cvec` ashape |
| [`vec/parameter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/parameter.hpp) | the shared vector `parameters<>` bundle (only `orientation`/`size_type` are used) |
| [`traits/ashape.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/traits/ashape.hpp) | the `scal`/`rvec`/`cvec`/`mat` shape taxonomy |

For the dense counterpart it shares shape semantics with, and the sparse matrix
it shares storage technique with, see the
[dense_vector](dense-vector-architecture.md) and
[compressed2D](compressed2d-architecture.md) architecture docs.
