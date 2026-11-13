# dense2D: architecture of the primary dense matrix

How `mtl::mat::dense2D` is put together, and *why* each piece is shaped the way
it is. The class is small — ~350 lines in
[`include/mtl/mat/dense2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/dense2D.hpp) —
but almost every line encodes a deliberate policy decision inherited from MTL4
and re-expressed in C++20. This document is the map of those decisions.

The organizing idea is **compile-time configuration with zero runtime cost**: a
single class template serves stack- and heap-allocated matrices, fixed- and
runtime-sized matrices, row- and column-major layouts, and 0- or 1-based
indexing, and it resolves every one of those axes at compile time so the
generated code is exactly what a hand-written matrix for that configuration
would be.

---

## The class in one picture

```text
dense2D<Value, Parameters = parameters<>>
   │
   │  Parameters bundles five compile-time policies:
   │     orientation      row_major | col_major
   │     index_type       c_index (0-based) | f_index (1-based)
   │     dimensions_type  fixed::dimensions<R,C> | non_fixed::dimensions
   │     storage          on_heap | on_stack
   │     size_type        std::size_t (default)
   │
   ├── memory_type mem_    contiguous_memory_block<Value, storage, static_size>
   │                       └─ the ONLY member that owns bytes
   ├── dim_type    dims_   [[no_unique_address]] — 0 bytes when fixed
   └── size_type   ldim_   leading dimension (fast-axis extent)
```

Two template parameters, not seven: `Value` and a single `Parameters` bundle.
Everything else is reached through `Parameters::`. The class itself holds only
three data members, and one of them (`dims_`) is free for fixed-size matrices.

---

## Decision 1 — a policy *bundle*, not a long template list

`dense2D` is `template <typename Value, typename Parameters>`. All five layout
policies live inside
[`parameters<>`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/parameter.hpp):

```cpp
template <typename Orientation = tag::row_major,
          typename Index       = detail::c_index,
          typename Dimensions  = non_fixed::dimensions,
          typename Storage     = tag::on_heap,
          typename SizeType    = std::size_t>
struct parameters { /* re-exports each as a named alias + is_fixed */ };
```

**Why a bundle.** A dense matrix has many orthogonal compile-time attributes.
Spelling them as direct template parameters — `dense2D<double, row_major,
c_index, non_fixed, on_heap, size_t>` — is unreadable, order-dependent, and
forces every downstream template (`compressed2D`, views, expressions) to repeat
the same parameter list. Bundling them into one struct gives:

- **One name to pass around.** Views, transposes, and expression results carry a
  single `Parameters` type instead of re-threading five arguments.
- **Named, defaulted axes.** `parameters<tag::col_major>` changes only
  orientation; the other four keep their defaults. The common case —
  `dense2D<double>` — is a heap, row-major, 0-based, runtime-sized matrix.
- **A place to enforce invariants.** The bundle static-asserts the combinations
  that cannot exist, so an illegal type never reaches `dense2D`:

  ```cpp
  static_assert(!std::is_same_v<Storage, tag::on_stack> || is_fixed,
      "Stack storage requires fixed-size dimensions");
  ```

  Stack storage needs a compile-time size; the constraint lives with the policy,
  not scattered through the matrix body.

This is the MTL4 `matrix::parameters` idea, kept intact because it was the right
shape — only the `enable_if`/MPL scaffolding around it changed.

---

## Decision 2 — composition over inheritance

`dense2D` **has-a** memory block; it is not built from a CRTP base hierarchy:

```cpp
private:
    memory_type mem_;                       // owns the bytes
    [[no_unique_address]] dim_type dims_;   // shape
    size_type ldim_;                        // leading dimension
```

MTL4 layered dense matrices on a stack of CRTP base classes
(`base_matrix` → `crtp_base_matrix` → …) so that shared behavior could be mixed
in statically. MTL5 drops that: the one genuinely reusable concern — a block of
contiguous, aligned bytes with value semantics — is factored into
`detail::contiguous_memory_block` and held **by value**.

**Why composition wins here.** The memory block is the only part with real
invariants (allocation, ownership, alignment, deep copy). Making it a member
rather than a base means:

- `dense2D`'s copy/move/destroy are almost entirely *defaulted* — they delegate
  to the member's value semantics. The copy constructor and copy assignment are
  `= default`; only move is written out, and only to reset the moved-from shape.
- No CRTP curiously-recurring template parameter threads through the type, so
  the type name stays `dense2D<Value, Parameters>` with nothing self-referential.
- The memory concern is independently testable and reused by other containers
  without inheriting a matrix interface they don't want.

---

## Decision 3 — one memory type, stack *or* heap, chosen by `if constexpr`

[`contiguous_memory_block<Value, Storage, StaticSize>`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/contiguous_memory_block.hpp)
is a single class that is *either* a stack array *or* a heap pointer, decided at
compile time:

```cpp
static constexpr bool on_stack = std::is_same_v<Storage, tag::on_stack>;

struct stack_data { alignas(block_alignment<Value>) Value data_[StaticSize>0?StaticSize:1]; };
struct heap_data  { Value* data_=nullptr; std::size_t size_=0; memory_category category_=own; };

std::conditional_t<on_stack, stack_data, heap_data> store_;
```

Every method (`allocate`, `realloc`, copy, move, `swap`, `data`) has an
`if constexpr (on_stack)` fork. MTL4 achieved the same split with *two* template
specializations of the memory block; MTL5 folds them into one body with
`if constexpr`, so the stack and heap paths sit side by side and cannot drift.

Two properties of this block matter to the rest of the library:

- **Over-alignment for SIMD.** Both the stack array (`alignas`) and heap
  allocations are aligned to `block_alignment<Value>` — at least
  `default_alignment = 64` bytes (a cache line, and enough for the widest SIMD
  registers). So `data()` from *any* `dense2D`, fixed or dynamic, is a legal
  aligned load/store address for the native GEMM kernels. Alignment is a storage
  concern, so it lives in the storage type, not in every kernel.
- **Three ownership modes on the heap.** `memory_category ∈ {own, external,
  view}`. Only `own` blocks are freed in the destructor; `external` (a
  user-supplied buffer, via `dense2D(rows, cols, Value* ptr)`) and `view` (an
  alias into someone else's storage) are left untouched. This is what lets
  `dense2D` wrap caller memory — e.g. a buffer from a C API or BLAS — without
  copying and without ever freeing memory it did not allocate.

---

## Decision 4 — fixed vs runtime dimensions, at zero cost when fixed

Shape is itself a policy, with two implementations in
[`dimension.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/dimension.hpp):

| Type | State | `is_fixed` | Cost |
|---|---|---|---|
| `fixed::dimensions<R,C>` | none — `R`,`C` are `constexpr` | `true` | **0 bytes** |
| `non_fixed::dimensions` | two `size_t` | `false` | 16 bytes |

`dense2D` stores the shape as `[[no_unique_address]] dim_type dims_`. For a
fixed-size matrix, `fixed::dimensions` is empty, and `[[no_unique_address]]`
lets the compiler give it **zero** size in the object — a `4×4` stack matrix
stores its 16 elements inline with *no* separate shape object taking up room;
only the leading-dimension word and alignment padding sit alongside the array.
For a dynamic matrix the same member is a real 16-byte `non_fixed::dimensions`.

The class reads shape through `if constexpr`, so the fixed case compiles to a
constant and the dynamic case to a member read:

```cpp
size_type num_rows() const {
    if constexpr (is_fixed) return dim_type::rows;   // constant
    else                    return dims_.num_rows();  // member load
}
```

`static_size` (rows×cols for fixed, else 0) is computed the same way and handed
to the memory block, so a fixed matrix's stack array is sized at compile time
while a dynamic matrix passes `StaticSize = 0` and takes the heap path. **One
class, both regimes, and the fixed regime pays for nothing it doesn't use.**

---

## Decision 5 — orientation and the leading dimension

Row- vs column-major is the `orientation` policy. The address arithmetic is a
single `if constexpr` and a stored leading dimension `ldim_`:

```cpp
static constexpr size_type compute_offset(size_type r, size_type c, size_type ldim) {
    if constexpr (std::is_same_v<orientation, tag::row_major>) return r * ldim + c;
    else                                                       return c * ldim + r;
}
void set_ldim() {  // fast-axis extent
    if constexpr (row_major) ldim_ = num_cols();
    else                     ldim_ = num_rows();
}
```

**Why store `ldim_` explicitly** rather than always recomputing `num_cols()`?
The leading dimension is exactly the parameter BLAS calls `lda`. Keeping it as a
named member (a) makes the `data()` + `get_ldim()` pair a drop-in BLAS operand —
see the `interface/blas.hpp` bindings — and (b) documents the layout contract:
`dense2D` is **tightly packed**, `ldim` equals the fast-axis length, there is no
padding. Padded sub-blocks and strided views are deliberately *not* this type's
job (see Tradeoffs); making `ldim_` explicit is what keeps that boundary clean.

---

## Decision 6 — index base as a policy

`index_type` is `c_index` (0-based, default) or `f_index` (1-based), from
[`detail/index.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/index.hpp).
Element access funnels external indices through `to_internal`:

```cpp
reference operator()(size_type r, size_type c) {
    auto ri = index_type::to_internal(r);   // c_index: identity; f_index: i-1
    auto ci = index_type::to_internal(c);
    ...
    return mem_[compute_offset(ri, ci, ldim_)];
}
```

For the default `c_index`, `to_internal` is the identity and folds away
entirely. The policy exists so Fortran-style 1-based code and libraries can be
mirrored exactly without a separate matrix type — again, one class covering a
configuration axis at zero cost in the common case.

---

## Element access and the compile-time bounds check

Bounds checking is gated on the global `bounds_checking` constant from
[`config.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/config.hpp),
which is `false` under `NDEBUG`:

```cpp
if constexpr (bounds_checking) {
    if (ri >= num_rows() || ci >= num_cols())
        throw std::out_of_range("dense2D: index out of range");
}
```

Because it is `if constexpr`, a release build contains *no* branch, no compare,
no throw path — `operator()` is a multiply-add and a load. Debug builds get full
checking. The safety/speed trade is made once, by the build type, not per call
site.

---

## Expression templates, assignment, and mixed precision

`dense2D` participates in MTL5's expression-template system through
concept-constrained templated constructors and assignment operators:

```cpp
template <typename Expr>
    requires (Matrix<Expr> && traits::is_expression_v<Expr>
              && std::convertible_to<typename Expr::value_type, Value>)
dense2D& operator=(const Expr& expr) {
    change_dim(expr.num_rows(), expr.num_cols());
    detail::parallel_ewise(num_rows(), num_cols(), [&](std::size_t r) {
        for (size_type c = 0; c < num_cols(); ++c)
            (*this)(r, c) = static_cast<Value>(expr(r, c));   // fused, element-wise
    });
    return *this;
}
```

Three design points are packed into that signature and body:

- **Concepts, not `enable_if`.** The `requires` clause reads as a sentence:
  *the right-hand side must be a matrix expression whose element type converts to
  ours*. This replaces MTL4's nested `boost::enable_if<is_matrix<...>>` chains.
- **Materialization is the fusion point.** The expression (`A + B*2`, a lazy
  tree) is only evaluated *here*, one element at a time, straight into
  `dense2D`'s storage — no temporaries per sub-expression. `change_dim` first
  resizes to match, so assignment also defines shape.
- **Mixed precision is first-class.** The constraint is
  `std::convertible_to<Expr::value_type, Value>`, and the body does an explicit
  `static_cast<Value>`. An expression in one precision can be evaluated into a
  matrix of another (the library's stated purpose: mixed-precision algorithm
  design with custom number types). The cast is explicit so narrowing is
  intentional, not silent.

`parallel_ewise` chunks by rows; each `(r,c)` is independent, so the result is
bit-identical to the serial nested loop at `MTL5_NUM_THREADS=1`. `operator+=`
and `operator-=` follow the same pattern for accumulation.

### When the expression reads the target

Writing straight into the storage is only right if the expression reads the
target at `(r, c)` alone. `A = A + trans(A)` doesn't: it reads `A(0,1)` to
compute `A(1,0)`, after `A(0,1)` has been overwritten. Before #824 that gave
`A(1,0) = 10` for `{{1,2,3},{4,5,6},{7,8,9}}`, where 6 is right, and it was a
data race under threads.

So `operator=`, `+=` and `-=` first ask
`detail::alias_reads_target_elsewhere` (`mat/expr/expr_alias.hpp`) whether
evaluating in place could read an element other than the one being written.
If it could, the expression is evaluated into a temporary, which is then
copied into the target's storage. An external-pointer matrix keeps writing
the caller's buffer.

The walk over the expression tree is conservative:

| reached in the tree | in place? |
|---|---|
| `+`, `-`, negation, scaling | as their operands are |
| the target itself, as a leaf | yes: read at `(r, c)` only, so `A = A + B` and `A = 2*A - A` stay in place |
| another contiguous matrix | yes, unless its storage overlaps the target's |
| a view (`trans`, `hermitian`, `upper`, ...) or a lazy product | no, if it reaches the target's storage at all |
| any other matrix (sparse, block, generated) | no: a `block_diagonal2D` can hold the target as one of its blocks |

---

## Move semantics and the valid-empty state

Copy and destroy are defaulted (the memory block handles them). Move is written
out for one reason — to leave the source in a valid, empty state:

```cpp
dense2D(dense2D&& other) noexcept
    : mem_(std::move(other.mem_)), dims_(other.dims_), ldim_(other.ldim_) {
    if constexpr (!is_fixed) other.dims_.set_dimensions(0, 0);
    other.ldim_ = 0;
}
```

The block's own move nulls the moved-from pointer; `dense2D` additionally zeroes
the shape so a moved-from dynamic matrix reports `0×0` rather than stale
dimensions. (For fixed matrices the shape is a compile-time constant and can't
be zeroed, which is consistent — a fixed matrix always has its declared shape.)
`swap` is provided member-wise so containers and algorithms get an efficient,
`noexcept` exchange.

---

## Trait plumbing: category and ashape

Two specializations connect `dense2D` to the compile-time dispatch machinery:

```cpp
template <...> struct traits::category<dense2D<...>> { using type = tag::dense; };
template <...> struct ashape::ashape<dense2D<...>>   { using type = mat<Value>; };
```

- `category = tag::dense` lets free operations (`mult`, norms, factorizations)
  pick the dense code path via `if constexpr` / tag dispatch without knowing the
  concrete type.
- `ashape = mat<Value>` places `dense2D` in the *algebraic shape* system, which
  tracks whether a symbol is a scalar, vector, or matrix so that expression types
  compose correctly (a `mat × vec` is a `vec`, etc.).

Keeping these as external trait specializations rather than member typedefs
means the same tags can be attached to third-party or adapted types, and the
operation layer depends only on the trait, not on `dense2D` itself.

---

## What the design deliberately does *not* do

Some boundaries are intentional, and naming them is part of the architecture:

- **No internal padding / arbitrary strides.** `ldim` always equals the fast-axis
  extent. Aligned *tiles* with a padded leading dimension, and strided
  sub-matrix views, are the province of separate view types — not fields on
  `dense2D`. This keeps the common container tight, contiguous, and trivially
  BLAS-compatible.
- **`realloc` does not preserve contents.** `change_dim` reallocates when the
  element count changes and does *not* copy old data across; it is a resize of a
  container, not a matrix-preserving grow. Callers that need the old contents
  copy first. This keeps resize O(1) in the no-change case and avoids paying for
  a copy nobody asked for.
- **Ownership is a runtime enum, not a type.** `own`/`external`/`view` is a field
  on the heap block, checked at destruction, rather than three distinct types.
  The trade buys a single `dense2D` type that can wrap external memory, at the
  cost of one branch in the destructor — worthwhile because interop buffers and
  owned buffers must otherwise be the same static type to flow through the same
  operations.

---

## Relationship to MTL4 and the C++20 rewrite

`dense2D` is a port, and the port is where the modern idioms show up. The design
*intent* is MTL4's; the *mechanism* is C++20:

| Concern | MTL4 mechanism | MTL5 mechanism |
|---|---|---|
| Enable/disable on type traits | `boost::enable_if<is_X<T>>` | `requires X<T>` concepts |
| Compile-time branch on policy | tag dispatch / class specialization | `if constexpr` |
| Stack vs heap block | two template specializations | one body + `std::conditional_t` |
| Empty dimension object | EBO via base class | `[[no_unique_address]]` member |
| Shared matrix behavior | CRTP base hierarchy | composition (memory block member) |
| Algebraic identities | `math::zero(ref)` | `math::zero<T>()` |

The result is a class whose behavior is easy to read top-to-bottom — the policy
forks are visible `if constexpr` blocks rather than resolved across a web of
specializations — while compiling to the same tight code MTL4 produced.

---

## File map

| File | Role |
|---|---|
| [`mat/dense2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/dense2D.hpp) | the matrix class, free functions, trait specializations |
| [`mat/parameter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/parameter.hpp) | the `parameters<>` policy bundle + invariants |
| [`mat/dimension.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/dimension.hpp) | `fixed::dimensions<R,C>` / `non_fixed::dimensions` |
| [`detail/contiguous_memory_block.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/contiguous_memory_block.hpp) | unified stack/heap storage, ownership modes |
| [`detail/aligned_allocator.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/aligned_allocator.hpp) | 64-byte over-aligned allocation for SIMD |
| [`detail/index.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/index.hpp) | `c_index` / `f_index` base policy |
| [`tag/orientation.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/tag/orientation.hpp), [`tag/storage.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/tag/storage.hpp) | orientation and storage tags |
| [`detail/ewise.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/ewise.hpp) | `parallel_ewise` used by assignment |

For how the *kernels* that consume `dense2D::data()`/`get_ldim()` are structured,
see [BLAS kernel architecture](../../design/blas-kernel-architecture.md).
