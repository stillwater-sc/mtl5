# Phase 12 Supplemental: Missing I/O, Setup Utilities & Convenience

## Context

Phase 12 was originally planned as "I/O, Setup Utilities & Convenience" but was implemented as element-wise transcendental functions instead. Most of the originally planned features already existed (Matrix Market I/O, laplacian generators, basic print), but several are genuinely missing. This supplemental commit fills those gaps.

**Already implemented** (no work needed):
- Matrix Market I/O (`include/mtl/io/matrix_market.hpp`) — mm_read, mm_read_dense, mm_write, mm_write_sparse
- Laplacian 1D/2D generators (`include/mtl/generators/laplacian.hpp`)
- `diagonal(A) -> dense_vector` extraction (`include/mtl/operation/diagonal.hpp`)
- Basic `operator<<` for vectors and matrices (`include/mtl/operation/print.hpp`)

**Missing features** to implement:

1. **`poisson2d_dirichlet(nx, ny)` generator** — Poisson-specific 2D discretization with h^2 scaling
2. **`diag(vec) -> compressed2D`** — Construct diagonal sparse matrix from a vector (inverse of `diagonal()`)
3. **CSV/whitespace element I/O** — Read/write dense and sparse matrices from simple delimited files
4. **Pretty-print enhancements** — Configurable precision, sparse triplet format, MATLAB-style output

## 12S.1 Poisson 2D Generator

**File**: `include/mtl/generators/poisson.hpp`

Differs from `laplacian_2d` by including the `h^2 = 1/((nx+1)*(ny+1))` scaling, making it the actual discretization of `-nabla^2 u = f` with Dirichlet BCs. Pattern follows `laplacian.hpp` exactly.

```cpp
namespace mtl::generators {
template <typename T = double>
auto poisson2d_dirichlet(std::size_t nx, std::size_t ny) -> mat::compressed2D<T>;
}
```

- `h_x = 1/(nx+1)`, `h_y = 1/(ny+1)`, diagonal = `2/h_x^2 + 2/h_y^2`, off-diag x = `-1/h_x^2`, off-diag y = `-1/h_y^2`
- For uniform grid (nx==ny): equivalent to `(nx+1)^2 * laplacian_2d(nx, ny)` scaled appropriately
- Uses `mat::inserter` + 5-point stencil loop (same structure as `laplacian_2d`)
- Add `#include` to `generators/generators.hpp` in "Sparse factory" tier (after laplacian)

## 12S.2 Diagonal Matrix Construction

**File**: `include/mtl/operation/diagonal.hpp` (add to existing file)

Add overload `diag(dense_vector<T>) -> compressed2D<T>` as inverse of existing `diagonal(Matrix) -> dense_vector`.

```cpp
namespace mtl {
template <typename T>
auto diag(const vec::dense_vector<T>& v) -> mat::compressed2D<T>;
}
```

- Returns n×n compressed2D with v(i) on diagonal
- Uses `mat::inserter` for construction
- Requires additional includes: `<mtl/mat/compressed2D.hpp>`, `<mtl/mat/inserter.hpp>`

## 12S.3 CSV / Whitespace Element I/O

### `include/mtl/io/read_el.hpp` — Read matrices from delimited files

```cpp
namespace mtl::io {
// Read dense matrix from CSV/whitespace file (one row per line)
template <typename Value = double>
mat::dense2D<Value> read_dense(const std::string& filename, char delimiter = ',');

// Read sparse matrix from triplet file (row col val per line, 0-based)
template <typename Value = double>
mat::compressed2D<Value> read_sparse(const std::string& filename,
                                      std::size_t nrows, std::size_t ncols);
}
```

- `read_dense`: Opens file, reads lines, splits by delimiter, populates dense2D. Auto-detects dimensions from file content.
- `read_sparse`: Reads triplet format (row col value per line), uses `coordinate2D` to build then `.compress()`. Follows `mm_read` pattern.
- Throws `std::runtime_error` on open failure (matching matrix_market.hpp pattern).

### `include/mtl/io/write_el.hpp` — Write matrices to delimited files

```cpp
namespace mtl::io {
// Write dense matrix to CSV/whitespace file
template <typename Matrix>
void write_dense(const std::string& filename, const Matrix& A, char delimiter = ',');

// Write sparse matrix to triplet file (0-based indices)
template <typename Value, typename Parameters>
void write_sparse(const std::string& filename,
                  const mat::compressed2D<Value, Parameters>& A);
}
```

- `write_dense`: Writes rows of values separated by delimiter, one row per line.
- `write_sparse`: Writes row col value triplets, one per line (0-based). Iterates CRS structure via `ref_major/ref_minor/ref_data`.
- Both set precision to 17 for full double reproducibility (matching matrix_market.hpp).

Update `include/mtl/mtl.hpp` to include both after the existing matrix_market include.

## 12S.4 Pretty-Print Enhancements

**File**: `include/mtl/operation/print.hpp` (extend existing 36-line file)

Add free functions alongside existing `operator<<`:

```cpp
namespace mtl {

/// Print with configurable precision
template <Matrix M>
void print(std::ostream& os, const M& m, int precision = 6);

template <Vector V>
void print(std::ostream& os, const V& v, int precision = 6);

/// Print sparse matrix in triplet format: (row, col) = value
template <typename Value, typename Parameters>
void print_sparse(std::ostream& os, const mat::compressed2D<Value, Parameters>& A,
                  int precision = 6);

/// Print matrix in MATLAB format: A = [row1; row2; ...]
template <Matrix M>
void print_matlab(std::ostream& os, const M& m, const std::string& name = "A",
                  int precision = 6);
}
```

- `print()`: Uses `os << std::setprecision(precision)`, otherwise same row-by-row / bracket format as existing `operator<<`. Requires `<iomanip>`.
- `print_sparse()`: Iterates CRS structure, outputs `(i, j) = val` lines. Efficient — skips zeros.
- `print_matlab()`: Outputs `name = [v1 v2 ...; v3 v4 ...; ...]` suitable for pasting into MATLAB/Octave.

## Files to Create (3 new headers)

| File | Description |
|------|-------------|
| `include/mtl/generators/poisson.hpp` | Poisson 2D Dirichlet generator |
| `include/mtl/io/read_el.hpp` | CSV/whitespace matrix reader |
| `include/mtl/io/write_el.hpp` | CSV/whitespace matrix writer |

## Files to Modify (3 existing headers)

| File | Change |
|------|--------|
| `include/mtl/operation/diagonal.hpp` | Add `diag(vec) -> compressed2D` function |
| `include/mtl/operation/print.hpp` | Add `print()`, `print_sparse()`, `print_matlab()` |
| `include/mtl/generators/generators.hpp` | Add `#include <mtl/generators/poisson.hpp>` |
| `include/mtl/mtl.hpp` | Add includes for `read_el.hpp` and `write_el.hpp` |

## Tests (2 new test files)

### `tests/unit/generators/test_poisson2d.cpp`
- Poisson 2D 3x3 grid: correct dimensions (9x9), SPD, diagonal values match h^2-scaled formula
- Poisson 2D: solve with CG converges (using `itl::cg` + identity PC)

### `tests/unit/io/test_element_io.cpp`
- Write/read dense roundtrip: values match within 1e-10
- Write/read sparse roundtrip: values and nnz match
- `read_dense` throws on missing file
- Cross-platform temp files: `std::filesystem::temp_directory_path() / "mtl5_test_..."`, cleanup with `std::remove()`

### `tests/unit/operation/test_diag_print.cpp`
- `diag(v)`: creates n×n matrix with correct diagonal values, nnz == n
- `diag(diagonal(A)) ≈ diag_part_of(A)` roundtrip on diagonal-dominant matrix
- `print()` outputs to stringstream with configurable precision
- `print_sparse()` outputs triplet format to stringstream
- `print_matlab()` produces parseable MATLAB format

## Implementation Order

1. Poisson 2D generator + test
2. `diag(vec)` function (add to existing `diagonal.hpp`)
3. CSV element I/O (`read_el.hpp`, `write_el.hpp`) + test
4. Pretty-print enhancements (extend `print.hpp`) + test
5. Update umbrella headers (`generators.hpp`, `mtl.hpp`)
6. Full build + full test suite

## Cross-Platform Notes

- Temp files: `std::filesystem::temp_directory_path()` only, never `/tmp/`
- No `M_PI` — use `std::numbers::pi` if needed
- File I/O: `std::ifstream`/`std::ofstream` with `std::string` paths

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake -B build && cmake --build build -j$(nproc) && ctest --test-dir build
```
