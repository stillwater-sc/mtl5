# Phase 9: Remaining Stubs — Sparse Vector, Recursion & External Interfaces

## Context

Phases 1–8 are complete (32/32 tests passing). Phase 9 is the final stub-completion phase, filling the 7 remaining empty stubs. Per the roadmap (`docs/modernization/mtl5-stub-completion-plan.md` §Phase 9), this covers three subsystems:

1. **Sparse vector** (2 stubs) — most broadly useful
2. **Recursion** (3 stubs) — block-recursive matrix traversal
3. **External interfaces** (2 stubs) — optional BLAS/LAPACK dispatch

These are niche/optional features. The MTL4 sparse_vector was explicitly marked incomplete ("DO NOT USE in numeric code yet!!!"), so MTL5 gets a proper modern implementation.

---

## Part 1: Sparse Vector (~150 lines)

### 1.1 `include/mtl/vec/sparse_vector.hpp`

A proper sparse vector using dual sorted arrays (indices + values).

**Class:** `sparse_vector<Value, Parameters>`

**API:**
- Constructor: `sparse_vector(size_type n)` — logical size, initially empty
- `size()` — logical dimension
- `nnz()` — number of stored entries
- `operator()(size_type i) const` — read access (returns 0 if absent)
- `operator[](size_type i)` — read/write access (inserts if absent)
- `exists(size_type i) const` — check if index is stored
- `insert(size_type i, Value v)` — insert maintaining sorted order
- `clear()` — remove all entries
- `crop(Value threshold)` — drop entries below threshold
- `indices() const` / `values() const` — raw access to internal arrays
- `begin()` / `end()` — iterator over `(index, value)` pairs

**Storage:** Two `std::vector<>` — one for indices, one for values — kept sorted by index. Uses `std::lower_bound` for O(log n) lookup.

**Concept satisfaction:** Satisfy `Vector` concept (needs `size()`, `num_rows()`, `num_cols()`, `value_type`, `size_type`). `operator()(i)` provides read access.

**Reference:** MTL4 `boost/numeric/mtl/vector/sparse_vector.hpp` (176 lines). Modernize: drop `zip_it`, use structured bindings, `std::ranges` where helpful.

### 1.2 `include/mtl/vec/inserter.hpp`

RAII inserter for sparse vector construction (parallels `mat::inserter`).

**Class:** `sparse_vector_inserter<Value, Parameters, Updater>`

**API:**
- Constructor: `explicit sparse_vector_inserter(sparse_vector<V,P>& vec)`
- `operator[](size_type i)` → returns proxy
- Proxy supports `operator<<(Value)` for insertion
- Destructor sorts and merges entries

**Updater functors:** Reuse `update_store<T>` and `update_plus<T>` from `mat/inserter.hpp`.

**Convenience alias:** `template <typename Vec, typename Updater> using vec_inserter = ...`

---

## Part 2: Recursion Module (~200 lines)

Block-recursive matrix traversal for cache-oblivious algorithms. Modernizes MTL4's Boost.MPL-heavy approach with C++20 `constexpr`.

### 2.1 `include/mtl/recursion/base_case_test.hpp`

Callable functors that decide when recursion bottoms out.

**Classes (all `constexpr`):**
- `min_dim_test{threshold}` — stops when `min(rows, cols) <= threshold`
- `max_dim_test{threshold}` — stops when `max(rows, cols) <= threshold`
- `max_dim_test_static<N>` — compile-time threshold version

Each is a callable: `bool operator()(const auto& recursator) const`

**Reference:** MTL4 `recursion/base_case_test.hpp` (126 lines). Drop `bound_test` variants (tied to obsolete recursator_s).

### 2.2 `include/mtl/recursion/matrix_recursator.hpp`

The core recursator: wraps a matrix and provides quad-tree subdivision.

**Class:** `recursator<Matrix>`

**API:**
- Constructor: `explicit recursator(Matrix& m)` — wraps entire matrix
- Private constructor for sub-regions: stores row/col offset+size
- `north_west()`, `north_east()`, `south_west()`, `south_east()` — return child recursators
- `operator()` or `get()` — return the sub-matrix view
- `num_rows()`, `num_cols()`, `is_empty()`

**Splitting:** Use `first_part(n)` (largest power of 2 ≤ n) for row split, similarly for cols. This is simpler than MTL4's virtual-bound approach.

**Utility free functions:**
- `first_part(std::size_t n)` — largest power of 2 ≤ n (bit manipulation)
- `is_power_of_2(std::size_t n)`
- `for_each(recursator, function, base_case_test)` — recursively apply function to all base cases

**Reference:** MTL4 `recursion/matrix_recursator.hpp` (543 lines, ~half obsolete). Port only the modern `recursator` class (~150 lines). Drop `recursator_s`. Replace `boost::shared_ptr` → unnecessary (use value semantics — recursators are lightweight).

### 2.3 `include/mtl/recursion/predefined_masks.hpp`

Compile-time bitmasks for Morton Z-order and related space-filling curves.

**Approach:** Replace MTL4's `generate_mask<>` Boost.MPL template chain with `constexpr` functions.

```cpp
constexpr unsigned long morton_z_mask = 0x5555'5555'5555'5555UL; // interleaved bits
constexpr unsigned long morton_mask   = ~morton_z_mask;
```

Plus doppled/shark variants as `constexpr` values. These are just constants — ~40 lines.

**Reference:** MTL4 `recursion/predefined_masks.hpp` (91 lines) + `recursion/bit_masking.hpp` (260 lines). Collapse both into simple `constexpr` constants and a few helper functions.

---

## Part 3: External Interfaces (~150 lines)

### 3.1 `include/mtl/interface/blas.hpp`

Optional BLAS dispatch, guarded by `MTL5_HAS_BLAS` (set by CMake).

**Contents:**
- `extern "C"` declarations for key BLAS routines:
  - Level 1: `daxpy_`, `ddot_`, `dnrm2_`, `dscal_`, `dcopy_`
  - Level 2: `dgemv_`, `dtrsv_`
  - Level 3: `dgemm_`, `sgemm_` (float/double at minimum)
- C++ wrappers in `mtl::interface::blas` namespace with overloads for float/double
- All guarded by `#ifdef MTL5_HAS_BLAS`

**Reference:** MTL4 `interface/blas.hpp` (166 lines).

### 3.2 `include/mtl/interface/lapack.hpp`

Optional LAPACK dispatch, guarded by `MTL5_HAS_LAPACK`.

**Contents:**
- `extern "C"` declarations:
  - `dpotrf_` (Cholesky), `dgetrf_` (LU), `dgeqrf_` (QR)
  - `dgesvd_` (SVD), `dsyev_` (symmetric eigenvalue)
  - Float variants: `spotrf_`, `sgetrf_`, etc.
- C++ wrappers in `mtl::interface::lapack` namespace

**Reference:** MTL4 `interface/lapack.hpp` (49 lines — only Cholesky). Expand to cover our full decomposition set.

### 3.3 CMake Integration

Modify top-level `CMakeLists.txt`:
```cmake
option(MTL5_ENABLE_BLAS "Enable BLAS acceleration" OFF)
option(MTL5_ENABLE_LAPACK "Enable LAPACK acceleration" OFF)

if(MTL5_ENABLE_BLAS)
    find_package(BLAS REQUIRED)
    target_compile_definitions(mtl5 INTERFACE MTL5_HAS_BLAS)
    target_link_libraries(mtl5 INTERFACE ${BLAS_LIBRARIES})
endif()
# Similar for LAPACK
```

---

## Part 4: Tests (~200 lines)

### `tests/unit/vec/test_sparse_vector.cpp`
- Construction, `nnz()`, `size()`
- Insert elements, verify sorted order
- `operator()` read (present + absent indices)
- `operator[]` write (insert + overwrite)
- `exists()`, `clear()`, `crop()`
- Iterator access
- Inserter pattern: RAII construction with `update_plus`

### `tests/unit/recursion/test_recursion.cpp`
- `first_part()` / `is_power_of_2()` utilities
- `recursator` on small dense2D: verify quadrant dimensions
- `for_each` with base case test: count base cases, verify coverage
- `min_dim_test` / `max_dim_test` thresholds

No test for BLAS/LAPACK interfaces (system-dependent; tested via existing operation tests when enabled).

---

## Part 5: Integration

### Update `include/mtl/mtl.hpp`
Add includes for the new headers:
```cpp
// Sparse vector
#include <mtl/vec/sparse_vector.hpp>
// Recursion
#include <mtl/recursion/base_case_test.hpp>
#include <mtl/recursion/matrix_recursator.hpp>
#include <mtl/recursion/predefined_masks.hpp>
// Interfaces (conditional)
#include <mtl/interface/blas.hpp>
#include <mtl/interface/lapack.hpp>
```

### Update `include/mtl/mtl_fwd.hpp`
Forward declaration for `sparse_vector` already exists (line 49). No changes needed.

---

## Part 6: Examples (2 pedagogical examples)

### `examples/phase9a_sparse_vector.cpp`
- Construct sparse vectors, demonstrate inserter pattern
- SpMV (sparse matrix × sparse vector) or dot product with dense
- Compare with dense_vector for sparsity advantage
- Show crop/threshold functionality

### `examples/phase9b_recursive_traversal.cpp`
- Build a matrix, create recursator
- Demonstrate quadrant decomposition visually
- Use `for_each` to apply a function to base cases
- Show how block-recursive algorithms improve cache locality

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `include/mtl/vec/sparse_vector.hpp` | **Implement** (currently stub) |
| `include/mtl/vec/inserter.hpp` | **Implement** (currently stub) |
| `include/mtl/recursion/base_case_test.hpp` | **Implement** (currently stub) |
| `include/mtl/recursion/matrix_recursator.hpp` | **Implement** (currently stub) |
| `include/mtl/recursion/predefined_masks.hpp` | **Implement** (currently stub) |
| `include/mtl/interface/blas.hpp` | **Implement** (currently stub) |
| `include/mtl/interface/lapack.hpp` | **Implement** (currently stub) |
| `include/mtl/mtl.hpp` | **Modify** — add new includes |
| `tests/unit/vec/test_sparse_vector.cpp` | **Create** |
| `tests/unit/recursion/test_recursion.cpp` | **Create** |
| `tests/CMakeLists.txt` | **Modify** — add 2 new test targets |
| `examples/phase9a_sparse_vector.cpp` | **Create** |
| `examples/phase9b_recursive_traversal.cpp` | **Create** |
| `examples/CMakeLists.txt` | **Modify** — add 2 new example targets |
| `CMakeLists.txt` (top-level) | **Modify** — add BLAS/LAPACK options |

---

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake -B build
cmake --build build -j$(nproc)

# All tests must pass (32 existing + 2 new = 34)
ctest --test-dir build

# Run new examples
./build/examples/phase9a_sparse_vector
./build/examples/phase9b_recursive_traversal
```
