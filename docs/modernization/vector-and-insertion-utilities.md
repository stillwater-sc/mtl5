# Phase 11: Vector & Insertion Utilities

## Context

MTL5 Phases 1-10 are complete (57 tests passing). Phase 11 ports four utility components from MTL4 that enable practical workflows: unit vector creation, strided data access (e.g., column extraction from row-major matrices), offset-shifted insertion for FEM assembly, and permutation-based matrix reordering. All four are modernized from MTL4's Boost-heavy style to clean C++20.

## Files to Create (4 headers + 4 tests)

### 1. `include/mtl/vec/unit_vector.hpp` — Factory function

Simple factory returning `dense_vector<T>` with 1 at position k, 0 elsewhere.

```cpp
namespace mtl::vec {
template <typename Value = double>
dense_vector<Value> unit_vector(std::size_t n, std::size_t k) {
    dense_vector<Value> v(n, math::zero<Value>());
    v(k) = math::one<Value>();
    return v;
}
} // namespace mtl::vec
namespace mtl { using vec::unit_vector; }
```

~15 lines. Depends on: `dense_vector.hpp`, `identity.hpp`.

### 2. `include/mtl/vec/strided_vector_ref.hpp` — Non-owning strided view

Lightweight non-owning reference to strided data. Primary use: extract columns from row-major `dense2D` without copy.

**Design** (simplified from MTL4's 275-line CRTP version):
- Non-owning: pointer + length + stride (no clone, no ownership transfer)
- Element access: `operator()(i)`, `operator[](i)` — both return `data_[i * stride_]`
- Custom iterator: simple pointer-with-stride iterator class (nested)
- `begin()`/`end()` for range-based for loops
- `size()`, `stride()`, `data()`
- `sub_vector(start, finish)` free function
- Trait specializations: `category → tag::dense`, `ashape → cvec<Value>`

**What we omit** vs MTL4:
- No CRTP base classes (`crtp_base_vector`, `vec_expr`)
- No clone constructor / owned mode
- No range_generator specializations (not used in MTL5)
- No expression template assignment (this is a view, not an owner)

~80 lines.

### 3. `include/mtl/mat/shifted_inserter.hpp` — Offset decorator for inserters

Wraps any inserter and shifts row/col indices. Direct port from MTL4 adapted to MTL5's `compressed2D_inserter` proxy chain pattern.

**Design:**
```cpp
namespace mtl::mat {
template <typename BaseInserter>
class shifted_inserter {
    // bracket_proxy for ins[r][c] << value pattern
    // operator[](row) adds row_offset
    // bracket_proxy operator[](col) adds col_offset
    // set_row_offset(), set_col_offset(), get_row_offset(), get_col_offset()
};
} // namespace mtl::mat
```

The base inserter is constructed internally (owns it). Constructor takes `(matrix, slot_size, row_offset, col_offset)`.

~60 lines. Depends on: `inserter.hpp`.

### 4. `include/mtl/operation/reorder.hpp` — Permutation-based matrix reordering

Free functions that permute rows/columns of a dense2D using a permutation vector. Builds on existing `permutation_matrix`.

**Design:**
```cpp
namespace mtl::mat {
// reorder_rows(A, perm): B(i,:) = A(perm[i],:)
template <typename Value, typename Params>
dense2D<Value, Params> reorder_rows(const dense2D<Value, Params>& A,
                                     const std::vector<std::size_t>& perm);

// reorder_cols(A, perm): B(:,j) = A(:,perm[j])
template <typename Value, typename Params>
dense2D<Value, Params> reorder_cols(const dense2D<Value, Params>& A,
                                     const std::vector<std::size_t>& perm);

// Symmetric reorder: PAP^T (rows and columns by same permutation)
template <typename Value, typename Params>
dense2D<Value, Params> reorder(const dense2D<Value, Params>& A,
                                const std::vector<std::size_t>& perm);

// P * A operator for permutation_matrix * dense2D
template <typename PV, typename MV, typename MP>
dense2D<...> operator*(const permutation_matrix<PV>& P, const dense2D<MV, MP>& A);
}
```

~80 lines. Depends on: `dense2D.hpp`, `permutation_matrix.hpp`.

## Files to Modify (2 headers)

### 5. `include/mtl/mtl_fwd.hpp` — Add forward declarations

Add to `namespace mtl::vec`:
```cpp
template <typename Value, typename Parameters> class strided_vector_ref;
```

### 6. Include umbrella (if one exists for vec/)

Check if there's a `types.hpp` or similar that should include the new headers. If so, add includes.

## Test Files to Create (4 files)

All in `tests/unit/vec/` and `tests/unit/mat/`:

### `tests/unit/vec/test_unit_vector.cpp`
- Unit vector has correct size
- Element at k is 1, all others are 0
- Works with different types (double, float, int)
- k=0 (first), k=n-1 (last), k in middle

### `tests/unit/vec/test_strided_vector_ref.cpp`
- Construct from raw pointer with stride
- Element access via operator() and operator[]
- Size and stride queries
- Modification through reference (non-const)
- Iterator-based iteration (begin/end)
- Sub-vector extraction
- Column extraction from row-major dense2D (primary use case)

### `tests/unit/mat/test_shifted_inserter.cpp`
- Insert with zero offset = normal insertion
- Insert with row offset shifts rows correctly
- Insert with col offset shifts columns correctly
- Insert with both offsets
- Change offsets mid-insertion
- FEM-style subblock assembly pattern

### `tests/unit/operation/test_reorder.cpp`
- reorder_rows with identity permutation = original
- reorder_rows with reversal permutation = reversed rows
- reorder_cols with specific permutation
- Symmetric reorder (PAP^T) preserves eigenvalues (trace check)
- P * A operator produces same result as reorder_rows

## Implementation Order

1. **unit_vector.hpp** + test — simplest, no dependencies beyond dense_vector
2. **strided_vector_ref.hpp** + test — self-contained, needs custom iterator
3. **shifted_inserter.hpp** + test — depends on inserter.hpp
4. **reorder.hpp** + test — depends on dense2D, permutation_matrix
5. Update `mtl_fwd.hpp` with forward declarations
6. Full build + full test suite

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `unit_vector` returns `dense_vector` by value | Simple, RVO eliminates copies, matches MTL5 style |
| `strided_vector_ref` is view-only, no ownership | Simpler than MTL4's clone pattern; lifetime is caller's responsibility |
| `shifted_inserter` owns its base inserter | Matches MTL4 pattern; RAII finalization via base destructor |
| `reorder` functions return new matrices | Pure functions, no mutation; caller decides what to do with result |
| Trait registration for `strided_vector_ref` | Needed so norms and other operations dispatch correctly |

## Cross-Platform Notes

- No file I/O, no POSIX APIs, no temp paths — all arithmetic/pointer operations
- `std::size_t` throughout for sizes
- No compiler-specific extensions
- Strided iterator uses pointer arithmetic only — portable across all targets

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake --build build -j$(nproc)
ctest --test-dir build
ctest --test-dir build -R "vec_test_unit_vector|vec_test_strided|mat_test_shifted|op_test_reorder"
```

## Estimated Size

| Component | Header | Test | Total |
|-----------|--------|------|-------|
| unit_vector | ~15 | ~40 | ~55 |
| strided_vector_ref | ~80 | ~80 | ~160 |
| shifted_inserter | ~60 | ~70 | ~130 |
| reorder | ~80 | ~70 | ~150 |
| mtl_fwd.hpp update | ~3 | — | ~3 |
| **Total** | **~238** | **~260** | **~498** |
