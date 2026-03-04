# MTL5 Project Skeleton Plan

## Context

MTL4's principal can no longer maintain the library. After a technical assessment, we decided to preserve the architecture (which is strong for mixed-precision/custom number type work with posits and LNS) but modernize it as MTL5 in a new repository. MTL4 remains as a reference. This plan creates the initial MTL5 project skeleton — directory structure, CMake build system, C++20 concepts, and stub headers — so that a developer can immediately begin porting MTL4 algorithms.

## Design Decisions

- **C++20 floor** — concepts, `if constexpr`, `constexpr`, `std::span`, ranges
- **Zero Boost dependency** — all Boost replaced with C++20 standard equivalents
- **ITL included from the start** — iterative solvers live at `mtl/itl/`
- **Catch2** for testing (via FetchContent)
- **Include root is `mtl/`** — no `boost/numeric/` prefix
- **Namespace remains `mtl::`** — ITL moves from `itl::` to `mtl::itl::`

## What Will Be Created

The skeleton creates the full directory tree with real (minimal but compilable) content in the foundational files, and stub headers with purpose comments for everything else. This gives a new developer the complete map of where everything goes.

### Directory Structure

```
mtl5/
├── CMakeLists.txt                    # Top-level: project, options, INTERFACE library target
├── CMakePresets.json                 # dev/release/CI presets
├── LICENSE
├── README.md
├── CLAUDE.md
├── .gitignore
├── .clang-format
│
├── cmake/
│   └── MTL5Config.cmake.in          # For find_package(MTL5) after install
│
├── include/mtl/
│   ├── mtl.hpp                      # Kitchen-sink: types + operations + itl
│   ├── mtl_fwd.hpp                  # Forward declarations for all key types
│   ├── version.hpp.in               # Configured at cmake time
│   ├── config.hpp                   # Tuning knobs (inline constexpr)
│   │
│   ├── concepts/                    # C++20 concepts (replace pseudo-concepts + enable_if)
│   │   ├── collection.hpp           # Collection, MutableCollection
│   │   ├── scalar.hpp               # Scalar, Field, OrderedField
│   │   ├── matrix.hpp               # Matrix, DenseMatrix, SparseMatrix
│   │   ├── vector.hpp               # Vector, DenseVector, ColumnVector, RowVector
│   │   ├── magnitude.hpp            # Magnitude trait + concept, magnitude_t<T>
│   │   ├── linear_operator.hpp      # For ITL: anything supporting A * x
│   │   └── preconditioner.hpp       # For ITL: solve() and adjoint_solve()
│   │
│   ├── tag/                         # Compile-time tags (flat structs, no virtual inheritance)
│   │   ├── orientation.hpp          # row_major, col_major
│   │   ├── sparsity.hpp             # dense, sparse
│   │   ├── traversal.hpp            # nz, all, row, col, major, minor (absorbs glas_tag)
│   │   ├── shape.hpp                # scalar, vector, matrix shape tags
│   │   └── storage.hpp              # on_stack, on_heap
│   │
│   ├── traits/                      # Type traits and metafunctions
│   │   ├── category.hpp             # category<T> -> tag mapping (std::conditional_t)
│   │   ├── ashape.hpp               # Algebraic shape: scal, rvec<>, cvec<>, mat<>
│   │   ├── flatcat.hpp              # Flat category for efficient dispatch
│   │   └── transposed_orientation.hpp
│   │
│   ├── math/                        # Type-safe algebraic identities
│   │   ├── identity.hpp             # identity_t<Op, T>, zero<T>(), one<T>()
│   │   └── operations.hpp           # add<T>, mult<T>, max<T>, min<T> operation tags
│   │
│   ├── detail/                      # Internal implementation
│   │   ├── contiguous_memory_block.hpp  # if constexpr stack/heap via std::array or unique_ptr
│   │   └── index.hpp                # c_index, f_index
│   │
│   ├── mat/                         # Matrix types
│   │   ├── parameter.hpp            # parameters<Orientation, Index, Dimensions, OnStack, SizeType>
│   │   ├── dimension.hpp            # fixed::dimensions<R,C>, non_fixed::dimensions
│   │   ├── dense2D.hpp              # Dense row/col-major matrix
│   │   ├── compressed2D.hpp         # CSR/CSC sparse matrix
│   │   ├── coordinate2D.hpp         # COO sparse matrix
│   │   ├── ell_matrix.hpp           # ELLPACK sparse matrix
│   │   ├── identity2D.hpp           # Implicit identity
│   │   ├── inserter.hpp
│   │   ├── operators.hpp            # operator+, -, * for matrices
│   │   ├── expr/                    # Expression templates
│   │   │   ├── mat_expr.hpp         # CRTP base
│   │   │   ├── dmat_expr.hpp
│   │   │   ├── smat_expr.hpp
│   │   │   └── mat_mat_times_expr.hpp
│   │   └── view/                    # Thin wrapper views
│   │       ├── transposed_view.hpp
│   │       ├── hermitian_view.hpp
│   │       ├── map_view.hpp
│   │       └── banded_view.hpp
│   │
│   ├── vec/                         # Vector types
│   │   ├── parameter.hpp
│   │   ├── dimension.hpp            # fixed::dimension<N>, non_fixed::dimension
│   │   ├── dense_vector.hpp
│   │   ├── sparse_vector.hpp
│   │   ├── inserter.hpp
│   │   ├── operators.hpp
│   │   └── expr/                    # Expression templates
│   │       └── vec_expr.hpp         # CRTP base
│   │
│   ├── operation/                   # Free-function operations
│   │   ├── dot.hpp
│   │   ├── mult.hpp                 # Multi-dispatch multiplication
│   │   ├── norms.hpp                # one_norm, two_norm, infinity_norm, frobenius_norm
│   │   ├── set_to_zero.hpp
│   │   ├── fill.hpp
│   │   ├── scale.hpp
│   │   ├── trans.hpp
│   │   ├── conj.hpp
│   │   ├── abs.hpp
│   │   ├── sqrt.hpp
│   │   ├── lu.hpp
│   │   ├── qr.hpp
│   │   ├── lq.hpp
│   │   ├── cholesky.hpp
│   │   ├── svd.hpp
│   │   ├── eigenvalue.hpp
│   │   ├── eigenvalue_symmetric.hpp
│   │   ├── householder.hpp
│   │   ├── givens.hpp
│   │   ├── lower_trisolve.hpp
│   │   ├── upper_trisolve.hpp
│   │   ├── inv.hpp
│   │   ├── kron.hpp
│   │   ├── trace.hpp
│   │   ├── diagonal.hpp
│   │   ├── sum.hpp
│   │   ├── product.hpp
│   │   ├── min.hpp
│   │   ├── max.hpp
│   │   ├── print.hpp
│   │   ├── num_rows.hpp
│   │   ├── num_cols.hpp
│   │   ├── size.hpp
│   │   ├── resource.hpp
│   │   ├── lazy.hpp
│   │   ├── fuse.hpp
│   │   ├── random.hpp
│   │   └── operators.hpp            # Global operator overloads
│   │
│   ├── functor/                     # Scalar/typed functors
│   │   ├── scalar/                  # sfunctor replacements
│   │   │   ├── plus.hpp
│   │   │   ├── minus.hpp
│   │   │   ├── times.hpp
│   │   │   ├── divide.hpp
│   │   │   ├── assign.hpp
│   │   │   ├── conj.hpp
│   │   │   ├── abs.hpp
│   │   │   ├── negate.hpp
│   │   │   └── sqrt.hpp
│   │   └── typed/                   # tfunctor replacements
│   │       ├── scale.hpp
│   │       ├── rscale.hpp
│   │       └── divide_by.hpp
│   │
│   ├── recursion/                   # Block-recursive infrastructure
│   │   ├── matrix_recursator.hpp
│   │   ├── predefined_masks.hpp
│   │   └── base_case_test.hpp
│   │
│   ├── io/                          # I/O
│   │   └── matrix_market.hpp
│   │
│   ├── itl/                         # Iterative Template Library
│   │   ├── itl.hpp                  # ITL umbrella include
│   │   ├── iteration/
│   │   │   ├── basic_iteration.hpp
│   │   │   ├── cyclic_iteration.hpp
│   │   │   └── noisy_iteration.hpp
│   │   ├── krylov/
│   │   │   ├── cg.hpp
│   │   │   ├── bicg.hpp
│   │   │   ├── bicgstab.hpp
│   │   │   ├── gmres.hpp
│   │   │   ├── tfqmr.hpp
│   │   │   ├── qmr.hpp
│   │   │   └── idr_s.hpp
│   │   ├── pc/                      # Preconditioners
│   │   │   ├── identity.hpp
│   │   │   ├── diagonal.hpp
│   │   │   ├── ilu_0.hpp
│   │   │   ├── ic_0.hpp
│   │   │   └── solver.hpp
│   │   └── smoother/
│   │       ├── gauss_seidel.hpp
│   │       ├── jacobi.hpp
│   │       └── sor.hpp
│   │
│   └── interface/                   # Optional external library interfaces
│       ├── blas.hpp
│       └── lapack.hpp
│
├── tests/
│   ├── CMakeLists.txt               # Catch2 FetchContent, test registration
│   ├── unit/
│   │   ├── CMakeLists.txt
│   │   ├── concepts/
│   │   │   └── test_concepts.cpp    # Concept satisfaction smoke tests
│   │   ├── mat/
│   │   │   └── test_dense2D.cpp
│   │   ├── vec/
│   │   │   └── test_dense_vector.cpp
│   │   ├── operation/
│   │   │   └── test_dot.cpp
│   │   ├── math/
│   │   │   └── test_zero_one.cpp
│   │   └── itl/
│   │       └── test_cg.cpp
│   └── integration/
│       └── CMakeLists.txt
│
└── examples/
    ├── CMakeLists.txt
    └── hello_mtl5.cpp               # Minimal working example
```

### Files With Real Content (not stubs)

These files will be fully implemented in the skeleton so the project compiles and the first test passes:

1. **`CMakeLists.txt`** (root) — project definition, INTERFACE library target, C++20 requirement, FetchContent for Catch2, install rules
2. **`CMakePresets.json`** — dev preset (Debug, warnings) and release preset
3. **`cmake/MTL5Config.cmake.in`** — for downstream `find_package(MTL5)`
4. **`.gitignore`** — build/, .cache/, compile_commands.json, etc.
5. **`include/mtl/version.hpp.in`** — `MTL5_VERSION_MAJOR/MINOR/PATCH` macros
6. **`include/mtl/config.hpp`** — `inline constexpr` tuning knobs ported from MTL4
7. **`include/mtl/concepts/scalar.hpp`** — `Scalar`, `Field`, `OrderedField` concepts
8. **`include/mtl/concepts/collection.hpp`** — `Collection`, `MutableCollection` concepts
9. **`include/mtl/concepts/magnitude.hpp`** — `magnitude_t<T>` trait + `Magnitude` concept
10. **`include/mtl/concepts/matrix.hpp`** — `Matrix`, `DenseMatrix`, `SparseMatrix` concepts
11. **`include/mtl/concepts/vector.hpp`** — `Vector`, `DenseVector`, `ColumnVector`, `RowVector` concepts
12. **`include/mtl/concepts/linear_operator.hpp`** — `LinearOperator` concept
13. **`include/mtl/concepts/preconditioner.hpp`** — `Preconditioner` concept
14. **`include/mtl/tag/orientation.hpp`** — `row_major`, `col_major` structs
15. **`include/mtl/tag/sparsity.hpp`** — `dense`, `sparse` structs
16. **`include/mtl/tag/shape.hpp`** — `scalar`, `vector`, `matrix` shape tags
17. **`include/mtl/tag/traversal.hpp`** — `nz`, `all`, `row`, `col` etc.
18. **`include/mtl/tag/storage.hpp`** — `on_stack`, `on_heap`
19. **`include/mtl/math/operations.hpp`** — `add<T>`, `mult<T>`, `max_op<T>`, `min_op<T>`
20. **`include/mtl/math/identity.hpp`** — `identity_t`, `zero()`, `one()` with constexpr
21. **`include/mtl/detail/index.hpp`** — `c_index`, `f_index` with constexpr conversion
22. **`include/mtl/mat/dimension.hpp`** — `fixed::dimensions<R,C>`, `non_fixed::dimensions`
23. **`include/mtl/vec/dimension.hpp`** — `fixed::dimension<N>`, `non_fixed::dimension`
24. **`include/mtl/mat/parameter.hpp`** — parameter bundle with static_assert
25. **`include/mtl/vec/parameter.hpp`** — vector parameter bundle
26. **`include/mtl/mtl_fwd.hpp`** — forward declarations for all key types
27. **`include/mtl/mtl.hpp`** — umbrella include
28. **`tests/unit/concepts/test_concepts.cpp`** — Catch2 test verifying concept satisfaction
29. **`tests/unit/math/test_zero_one.cpp`** — Catch2 test for math::zero/math::one
30. **`examples/hello_mtl5.cpp`** — minimal program that includes mtl.hpp and compiles
31. **`README.md`** — project overview, build instructions, architecture overview
32. **`CLAUDE.md`** — AI assistant context for MTL5

### Stub Headers

All remaining headers (operations, matrix/vector types, ITL, etc.) will be created as stub files with:
- `#pragma once`
- Namespace declaration
- A comment describing what should be ported from MTL4 and the corresponding MTL4 file path

Example stub:
```cpp
#pragma once
// MTL5 stub — port from MTL4: boost/numeric/mtl/operation/lu.hpp
// LU factorization with partial pivoting
// Key changes from MTL4:
//   - Replace math::zero/one literals (0.0, 1.0) with constexpr math::zero<T>/one<T>
//   - Replace boost::enable_if with requires clauses
//   - Use std::abs via ADL (already correct in MTL4)
namespace mtl::operation {
} // namespace mtl::operation
```

### Key Boost-to-C++20 Replacements

| MTL4 Boost usage | MTL5 replacement |
|---|---|
| `boost::enable_if<is_matrix<T>>` | `requires Matrix<T>` |
| `boost::mpl::if_<cond, A, B>::type` | `std::conditional_t<cond, A, B>` |
| `boost::mpl::bool_<true>` | `std::true_type` or `constexpr bool` |
| `boost::is_same<A,B>` | `std::is_same_v<A,B>` |
| `boost::is_base_of<A,B>` | `std::is_base_of_v<A,B>` |
| `boost::shared_ptr<T>` | `std::shared_ptr<T>` |
| `boost::type_traits/*` | `<type_traits>` |
| `BOOST_STATIC_ASSERT` | `static_assert` |
| `boost::lambda::*` | C++20 lambdas |

## Verification

After creating the skeleton:
1. `cmake -B build -DCMAKE_CXX_STANDARD=20` — configure succeeds
2. `cmake --build build` — builds all targets (tests, examples)
3. `ctest --test-dir build` — concept satisfaction test and math::zero_one test pass
4. `./build/examples/hello_mtl5` — runs and prints version info


## Recommended Porting Roadmap

Phase 1: Core Data Types (unlocks everything)

  1. detail/contiguous_memory_block.hpp — stack/heap storage backend (std::array vs std::unique_ptr via if constexpr). Both dense_vector and dense2D depend
  on this.
  2. vec/dense_vector.hpp — simpler than dense2D, good first real port. CRTP base, orientation-aware, element access.
  3. mat/dense2D.hpp — the core matrix type. Row/col-major indexing, CRTP, parameter-driven.

Phase 2: Essential Operations (makes types usable)

  4. operation/set_to_zero.hpp, fill.hpp, print.hpp — initialization and debugging
  5. operation/size.hpp, num_rows.hpp, num_cols.hpp — free-function accessors
  6. operation/dot.hpp — first real computation, proves the vector type works
  7. operation/norms.hpp — needed by every iterative solver
  8. operation/mult.hpp — mat*vec and mat*mat, the core BLAS-level operation

Phase 3: First Decomposition + Triangular Solvers

  9. operation/lower_trisolve.hpp, upper_trisolve.hpp
  10. operation/lu.hpp — first factorization, validates the full stack

Phase 4: Sparse + ITL

  11. mat/inserter.hpp + mat/compressed2D.hpp — first sparse type (CSR/CSC)
  12. itl/iteration/basic_iteration.hpp + itl/krylov/cg.hpp — first iterative solver on a real sparse system

##  Phase 1: Core Data Types — contiguous_memory_block, dense_vector, dense2D

Context

The MTL5 skeleton is complete (141 files, all tests pass). Phase 1 ports the three foundational types that every operation depends on: the memory block,
the dense vector, and the dense matrix. Without these, no algorithms can be ported.

MTL4 uses deep CRTP inheritance chains, Boost.MPL macros, and full class specializations for stack/heap. 
MTL5 replaces all of this with if constexpr, std::conditional_t for data members, C++20 concepts for constraints, 
and composition instead of inheritance for the memory block.

### Implementation Steps

Step 1: Update dimension and parameter types

Small additions to existing files to support the memory block and data types.

 include/mtl/vec/dimension.hpp — Add value = 0 to non_fixed::dimension:
 static constexpr std::size_t value = 0;  // signals dynamic size
 This lets dense_vector query dim_type::value uniformly regardless of fixed vs non-fixed.

 include/mtl/vec/parameter.hpp — Add is_fixed and storage constraint:
 static constexpr bool is_fixed = Dimensions::is_fixed;
 static_assert(!std::is_same_v<Storage, tag::on_stack> || is_fixed,
     "Stack storage requires fixed-size dimensions");

 include/mtl/mat/parameter.hpp — Same additions:
 static constexpr bool is_fixed = Dimensions::is_fixed;
 static_assert(!std::is_same_v<Storage, tag::on_stack> || is_fixed,
     "Stack storage requires fixed-size dimensions");

Step 2: Implement contiguous_memory_block

File: include/mtl/detail/contiguous_memory_block.hpp (replace existing stub)

Single class template with std::conditional_t for data members, if constexpr for logic:

```cpp
 template <typename Value, typename Storage, std::size_t StaticSize = 0>
 class contiguous_memory_block
```

Key design:

 - enum class memory_category { own, external, view } — three ownership modes
 - std::conditional_t<on_stack, stack_data, heap_data> store_ — one data member
   - stack_data: raw Value data_[StaticSize] array with alignas(alignof(Value))
   - heap_data: Value* data_, std::size_t size_, memory_category category_
 - Stack: always own, size() returns StaticSize, copy is std::copy_n
 - Heap: new[]/delete[], move swaps pointer, views are shallow
 - No alignment macros — use alignas on stack, plain new[] on heap
 - No memory_crtp — data(), operator[], size() are direct members

Constructors: default, size(n), external(ptr, n, is_view), copy, move
 Rule of 5: destructor delete[] for own heap; copy/move as described above

Step 3: Implement dense_vector

File: include/mtl/vec/dense_vector.hpp (replace existing stub)

```cpp
 template <typename Value, typename Parameters = parameters<>>
 class dense_vector
```

Key design:

 - Composition not inheritance: memory_type mem_ (contiguous_memory_block)
 - Derives static_size from dim_type::value (0 for dynamic, N for fixed)
 - Derives actual_storage from Parameters::storage

Public interface:

 - value_type, size_type, reference, const_reference, pointer, const_pointer, orientation
 - Constructors: default, size(n), size+value, external(n,ptr), copy, move, initializer_list, std::vector
 - operator()(i), operator[](i) with bounds checking via config.hpp::bounds_checking
 - size(), empty(), data(), begin(), end(), stride() (always 1)
 - num_rows(), num_cols() — orientation-dependent (col_major: size x 1, row_major: 1 x size)
 - change_dim(n), checked_change_dim(n)
 - swap()

Free functions (after class, in mtl::vec): size(), fill(), num_rows(), num_cols()

Traits specializations (after class):

 - mtl::traits::category<dense_vector> → tag::dense
 - mtl::ashape::ashape<dense_vector> → cvec<Value> or rvec<Value> based on orientation

Convenience alias: namespace mtl { using vec::dense_vector; }

Step 4: Implement dense2D

File: include/mtl/mat/dense2D.hpp (replace existing stub)

```cpp
 template <typename Value, typename Parameters = parameters<>>
 class dense2D
```

Key design:

 - Composition: memory_type mem_, dim_type dims_, size_type ldim_
 - Inline compute_offset(r, c, ldim) — if constexpr on orientation:
   - row_major: r * ldim + c
   - col_major: c * ldim + r
 - set_ldim(): row_major → num_cols(), col_major → num_rows()
 - Index conversion via index_type::to_internal() in operator()(r, c)

Public interface:

 - Same type aliases as dense_vector plus index_type
 - Constructors: default, (rows,cols), (dims), external(rows,cols,ptr), copy, move, initializer_list<initializer_list>
 - operator()(r, c) with bounds checking
 - num_rows(), num_cols(), size(), get_ldim(), data(), begin(), end()
 - change_dim(r, c), checked_change_dim(r, c)
 - swap()

Free functions: num_rows(), num_cols(), size()

Traits specializations:

 - mtl::traits::category<dense2D> → tag::dense
 - mtl::ashape::ashape<dense2D> → mat<Value>

Convenience alias: namespace mtl { using mat::dense2D; }

Step 5: Update umbrella and operation stubs

```cpp
 include/mtl/mtl.hpp — Add includes:
 #include <mtl/detail/contiguous_memory_block.hpp>
 #include <mtl/vec/dense_vector.hpp>
 #include <mtl/mat/dense2D.hpp>
```

Operation stubs — Give minimal concept-constrained implementations:

 - operation/set_to_zero.hpp: std::fill(c.begin(), c.end(), math::zero<T>())
 - operation/fill.hpp: std::fill(c.begin(), c.end(), val)
 - operation/print.hpp: ostream operator<< for vector/matrix
 - operation/size.hpp, num_rows.hpp, num_cols.hpp: forward to member functions

Step 6: Write tests

 New: tests/unit/detail/test_contiguous_memory_block.cpp — heap/stack construction, copy, move, external, view, realloc

 Update: tests/unit/vec/test_dense_vector.cpp — concept satisfaction (STATIC_REQUIRE), all constructors, element access, iteration, fill, orientation-aware
 num_rows/num_cols, change_dim, fixed-size on stack, swap

 Update: tests/unit/mat/test_dense2D.cpp — concept satisfaction, constructors, row/col-major element access, initializer list, ldim correctness, change_dim,
  fixed-size on stack, data pointer, f_index support

 CMake: Add test_contiguous_memory_block to tests/unit/CMakeLists.txt, create tests/unit/detail/ directory

Files Modified (10)

|                      File                      |               Action                |
|------------------------------------------------|-------------------------------------|
| include/mtl/vec/dimension.hpp                  | Add value = 0 to non_fixed          |
| include/mtl/vec/parameter.hpp                  | Add is_fixed, storage static_assert |
| include/mtl/mat/parameter.hpp                  | Add is_fixed, storage static_assert |
| include/mtl/detail/contiguous_memory_block.hpp | Full implementation (replace stub)  |
| include/mtl/vec/dense_vector.hpp               | Full implementation (replace stub)  |
| include/mtl/mat/dense2D.hpp                    | Full implementation (replace stub)  |
| include/mtl/mtl.hpp                            | Add 3 new includes                  |
| include/mtl/operation/set_to_zero.hpp          | Minimal implementation              |
| include/mtl/operation/fill.hpp                 | Minimal implementation              |
| include/mtl/operation/print.hpp                | Minimal implementation              |

 Files Created (1)

|                        File                        |         Purpose         |
|----------------------------------------------------|-------------------------|
| tests/unit/detail/test_contiguous_memory_block.cpp | Memory block unit tests |

 Files Updated (Tests, 2)

|                 File                 |             Purpose             |
|--------------------------------------|---------------------------------|
| tests/unit/vec/test_dense_vector.cpp | Expanded with real vector tests |
| tests/unit/mat/test_dense2D.cpp      | Expanded with real matrix tests |

 CMake Updates (1)

|           File            |              Change              |
|---------------------------|----------------------------------|
| tests/unit/CMakeLists.txt | Add test_contiguous_memory_block |

 Verification

 cmake -B build -DCMAKE_CXX_STANDARD=20   # configure
 cmake --build build -j$(nproc)             # build all
 ctest --test-dir build                     # all tests pass
 ./build/examples/hello_mtl5               # still runs

 Expected: all existing tests continue to pass, new tests for contiguous_memory_block / dense_vector / dense2D pass, concept satisfaction verified via
 STATIC_REQUIRE.

##   Phase 1 Complete — Summary

All 7 tests pass, hello_mtl5 example runs correctly.

Files Modified (7)


|                      File                      |                                                Change                                     |
|------------------------------------------------|-------------------------------------------------------------------------------------------|
| include/mtl/vec/dimension.hpp                  | Added value = 0 to non_fixed::dimension                                                   |
| include/mtl/vec/parameter.hpp                  | Added is_fixed, storage static_assert                                                     |
| include/mtl/mat/parameter.hpp                  | Added is_fixed, storage static_assert                                                     |
| include/mtl/detail/contiguous_memory_block.hpp | Full implementation: stack/heap via std::conditional_t, memory_category enum, Rule of 5   |
| include/mtl/vec/dense_vector.hpp               | Full implementation: composition with memory block, all constructors, bounds checking,    |
|                                                | orientation-aware sizing, traits specializations                                          |
| include/mtl/mat/dense2D.hpp                    | Full implementation: row/col-major layout, compute_offset, index_type::to_internal,       |
|                                                | initializer_list, traits                                                                  |
| include/mtl/mtl.hpp                            | Added includes for new types and operations                                               |


Operations Implemented (6)


|           File            |                   Implementation                    |
|---------------------------|-----------------------------------------------------|
| operation/set_to_zero.hpp | std::fill with math::zero<T>(), concept-constrained |
| operation/fill.hpp        | std::fill with given value, concept-constrained     |
| operation/print.hpp       | operator<< for Vector and Matrix concepts           |
| operation/size.hpp        | Forwards to c.size()                                |
| operation/num_rows.hpp    | Forwards to m.num_rows()                            |
| operation/num_cols.hpp    | Forwards to m.num_cols()                            |


Files Created (1)


|                        File                        |                                           Purpose                                            |
|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| tests/unit/detail/test_contiguous_memory_block.cpp | 16 test cases: heap/stack construction, copy, move, external, view, realloc, iteration, swap |


Tests Updated (2)


|                 File                 |                                                   Test Cases                                                   |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------|
| tests/unit/vec/test_dense_vector.cpp | 32 test cases covering concepts, constructors, access, iteration, fill, orientation, change_dim, fixed/stack,  |
|                                      | swap, traits                                                                                                   |
| tests/unit/mat/test_dense2D.cpp      | 24 test cases covering concepts, constructors, row/col-major layout, ldim, bounds checking, change_dim,        |
|                                      | fixed/stack, f_index, swap, traits                                                                             |


Key Design Decisions

  - Composition over inheritance — dense_vector and dense2D hold a memory_type mem_ member, no CRTP chains
  - std::conditional_t selects stack vs heap storage at compile time in contiguous_memory_block
  - if constexpr replaces all runtime dispatch for fixed/dynamic, stack/heap, row/col-major paths
  - Custom move operations — explicitly zero source dimensions after move (defaulted moves leave stale size)
  - ::mtl::mat:: qualification — prevents name collision with ashape::mat<> template inside namespace mtl::ashape

##  Phase 2: Essential Arithmetic — Operators, Reductions, Norms, Multiply

 Context

 Phase 1 delivered contiguous_memory_block, dense_vector, and dense2D with basic operations (fill, set_to_zero, print). Phase 2 builds the arithmetic layer
 on top: everything needed to write y = A*x + b, dot(x,y), two_norm(r), etc. This is the prerequisite for Phase 3 (ITL Krylov solvers, expression
 templates).

 Design principle: Phase 2 is eager (immediate evaluation). Operators return new containers, not lazy expression templates. Expression templates are Phase
 3. This keeps Phase 2 simple, correct, and testable.

### Implementation Steps

Step 1: Scalar Functors (9 files)

 Implement all 9 stub files in include/mtl/functor/scalar/. Each is a constexpr struct with result_type, static apply(), and operator().

|           File            |    Functor    |             Signature             |
|---------------------------|---------------|-----------------------------------|
| functor/scalar/plus.hpp   | plus<T1,T2>   | (a, b) → a + b                    |
| functor/scalar/minus.hpp  | minus<T1,T2>  | (a, b) → a - b                    |
| functor/scalar/times.hpp  | times<T1,T2>  | (a, b) → a * b                    |
| functor/scalar/divide.hpp | divide<T1,T2> | (a, b) → a / b                    |
| functor/scalar/assign.hpp | assign<T1,T2> | (a, b) → a = b                    |
| functor/scalar/negate.hpp | negate<T>     | (v) → -v                          |
| functor/scalar/abs.hpp    | abs<T>        | (v) → |v|, returns magnitude_t<T> |
| functor/scalar/conj.hpp   | conj<T>       | (v) → conj(v), identity for reals |
| functor/scalar/sqrt.hpp   | sqrt<T>       | (v) → sqrt(v)                     |

Binary functors use std::common_type_t<T1,T2> for result_type.

Step 2: Typed Functors (3 files)

 Capture a scalar at construction, apply it per-element.

|            File             |   Functor    |     Pattern     |
|-----------------------------|--------------|-----------------|
| functor/typed/scale.hpp     | scale<S>     | (x) → alpha * x |
| functor/typed/rscale.hpp    | rscale<S>    | (x) → x * alpha |
| functor/typed/divide_by.hpp | divide_by<S> | (x) → x / alpha |

Step 3: Compound Assignment on dense_vector and dense2D

 Modify include/mtl/vec/dense_vector.hpp — add member operators:
 - operator+=(const dense_vector&), operator-=(const dense_vector&)
 - template<Scalar S> operator*=(const S&), template<Field S> operator/=(const S&)

Modify include/mtl/mat/dense2D.hpp — same set of compound operators.

Step 4: Reductions — dot, sum, product

operation/dot.hpp — in namespace mtl:

 - dot(v1, v2) — Hermitian: Σ conj(v1[i]) * v2[i]
 - dot_real(v1, v2) — no conjugation: Σ v1[i] * v2[i]
 - Both use math::zero<result_type>() as accumulator

 operation/sum.hpp — sum(c) → Σ c[i], accumulator math::zero<T>()

 operation/product.hpp — product(c) → Π c[i], accumulator math::one<T>()

Step 5: Norms

 operation/norms.hpp — all in namespace mtl:

|     Function      |         Formula         |        Return type         |
| one_norm(v)       | Σ |v[i]|                | magnitude_t<V::value_type> |
| two_norm(v)       | √(Σ |v[i]|²)            | magnitude_t<V::value_type> |
| infinity_norm(v)  | max |v[i]|              | magnitude_t<V::value_type> |
| frobenius_norm(m) | √(Σ |m[i,j]|²)          | magnitude_t<M::value_type> |
| one_norm(m)       | max col sum of |m[i,j]| | magnitude_t<M::value_type> |
| infinity_norm(m)  | max row sum of |m[i,j]| | magnitude_t<M::value_type> |

two_norm computes Σ|v[i]|² inline (avoids circular dep with dot.hpp).

Step 6: scale, abs, conj, negate, sqrt, max, min

 operation/scale.hpp — scale(alpha, c) in-place, scaled(alpha, v) returns copy
 operation/abs.hpp — abs(vector) → new dense_vector<magnitude_t<T>>, abs(matrix) → new dense2D<magnitude_t<T>>
 operation/conj.hpp — conj(vector) → new vector, identity for real types
 operation/negate.hpp (NEW file) — negate(vector) → new vector with -v[i]
 operation/sqrt.hpp — sqrt(vector) → new vector
 operation/max.hpp — max(collection) → max element value (reduction)
 operation/min.hpp — min(collection) → min element value (reduction)

Step 7: Transposed View + trans()

mat/view/transposed_view.hpp — lightweight non-owning view:

 - Stores const Matrix&, swaps row/col in operator()(r,c) → ref_(c,r)
 - num_rows() → ref_.num_cols(), num_cols() → ref_.num_rows()
 - Satisfies Matrix concept
 - Specialize category → tag::dense, ashape → mat<Value>

 operation/trans.hpp — trans(m) → transposed_view<M>(m)

Step 8: mult() Function

 operation/mult.hpp — in namespace mtl:
 - mult(A, x, y) — mat×vec into pre-allocated y (triple-nested loop)
 - mult(A, B, C) — mat×mat into pre-allocated C (triple-nested loop)

 These write into pre-allocated output — essential for Krylov solvers.

 Step 9: Arithmetic Operators

 vec/operators.hpp — in namespace mtl::vec (found by ADL):
 - operator+(V1, V2), operator-(V1, V2), unary operator-(V) → new dense_vector<common_type_t>
 - operator*(Scalar, V), operator*(V, Scalar), operator/(V, Scalar) → new dense_vector

 mat/operators.hpp — in namespace mtl::mat (found by ADL):
 - Same arithmetic operators for matrices
 - operator*(Matrix, Vector) → new dense_vector (mat-vec multiply)
 - operator*(Matrix, Matrix) → new dense2D (mat-mat multiply)

Concept disambiguation: 
  - Scalar requires arithmetic on the type itself; 
  - Matrix requires num_rows(), num_cols(), m(r,c). 
These are naturally disjoint — no type satisfies both.

 operation/operators.hpp — umbrella that includes vec/operators.hpp + mat/operators.hpp

Step 10: flatcat Trait + Update Umbrella

 traits/flatcat.hpp — maps T → category_t<T> (trivial, just an alias for now)

 mtl.hpp — add includes for all new headers (functors, operations, operators, views, flatcat)

Step 11: Tests

  - Update tests/unit/operation/test_dot.cpp — replace placeholder with real tests (double, int, complex, orthogonal vectors)
  - Create tests/unit/operation/test_norms.cpp — one_norm, two_norm, infinity_norm, frobenius_norm for vectors and matrices
  - Create tests/unit/operation/test_vector_ops.cpp — all vector operators (+, -, *, /), compound assignment, sum, product, max, min, abs, conj, scale
  - Create tests/unit/operation/test_matrix_ops.cpp — all matrix operators, matvec, matmat, trans(), mult()
  - Create tests/unit/functor/test_scalar_functors.cpp — all 9 scalar functors + 3 typed functors
  - Update tests/unit/CMakeLists.txt — register 4 new test executables

 Files Modified (4)

|               File               |             Change              |
|----------------------------------|---------------------------------|
| include/mtl/vec/dense_vector.hpp | Add operator+=, -=, *=, /=      |
| include/mtl/mat/dense2D.hpp      | Add operator+=, -=, *=, /=      |
| include/mtl/mtl.hpp              | Add ~27 new includes            |
| tests/unit/CMakeLists.txt        | Register 4 new test executables |

 Files Filled (existing stubs → implementation, 29)

|                                           File                                           |            Content             |
|------------------------------------------------------------------------------------------|--------------------------------|
| include/mtl/functor/scalar/{plus,minus,times,divide,assign,negate,abs,conj,sqrt}.hpp     | 9 scalar functors              |
| include/mtl/functor/typed/{scale,rscale,divide_by}.hpp                                   | 3 typed functors               |
| include/mtl/operation/{dot,sum,product,norms,scale,mult,trans,abs,conj,sqrt,max,min}.hpp | 12 operations                  |
| include/mtl/operation/operators.hpp                                                      | Umbrella for vec+mat operators |
| include/mtl/vec/operators.hpp                                                            | Vector arithmetic operators    |
| include/mtl/mat/operators.hpp                                                            | Matrix arithmetic operators    |
| include/mtl/mat/view/transposed_view.hpp                                                 | Transposed view class          |
| include/mtl/traits/flatcat.hpp                                                           | Flat category trait            |

 Files Created (5)

|                    File                     |                  Purpose                   |
|---------------------------------------------|--------------------------------------------|
| include/mtl/operation/negate.hpp            | Element-wise negation (stub doesn't exist) |
| tests/unit/functor/test_scalar_functors.cpp | Functor unit tests                         |
| tests/unit/operation/test_norms.cpp         | Norm tests                                 |
| tests/unit/operation/test_vector_ops.cpp    | Vector operation tests                     |
| tests/unit/operation/test_matrix_ops.cpp    | Matrix operation + mult + trans tests      |

 Files Updated (1)

|               File                |               Change                |
|-----------------------------------|-------------------------------------|
| tests/unit/operation/test_dot.cpp | Replace placeholder with real tests |

 NOT Touched (deferred to Phase 3)

 - include/mtl/operation/lazy.hpp — expression templates
 - include/mtl/vec/expr/vec_expr.hpp — CRTP expression base
 - include/mtl/mat/expr/*.hpp — matrix expression templates

 Verification

 cmake -B build -DCMAKE_CXX_STANDARD=20
 cmake --build build -j$(nproc)
 ctest --test-dir build --output-on-failure
 ./build/examples/hello_mtl5

 Expected: all existing tests pass, 4 new test executables pass (test_scalar_functors, test_norms, test_vector_ops, test_matrix_ops), updated test_dot
 passes, hello_mtl5 still runs.

## Phase 2 Complete: Essential Arithmetic

  All 11/11 tests pass, hello_mtl5 runs, build is clean.

What was implemented:

- Scalar Functors (9 files) — plus, minus, times, divide, assign, negate, abs, conj, sqrt in functor/scalar/
- Typed Functors (3 files) — scale, rscale, divide_by in functor/typed/
- Compound Assignment — operator+=, -=, *=, /= added to both dense_vector and dense2D
- Reductions — dot() (Hermitian), dot_real(), sum(), product() in namespace mtl
- Norms — one_norm, two_norm, infinity_norm for vectors; frobenius_norm, one_norm, infinity_norm for matrices
- Element-wise Operations — scale(), scaled(), abs(), conj(), negate(), sqrt(), max(), min()
- Transposed View — transposed_view<M> non-owning view class + trans(m) free function
- mult() — mult(A, x, y) mat-vec and mult(A, B, C) mat-mat into pre-allocated output
- Arithmetic Operators — +, -, unary -, scalar*, */scalar for vectors and matrices; operator*(Matrix, Vector) and operator*(Matrix, Matrix)
- Umbrella — mtl.hpp updated with all 27 new includes; flatcat.hpp implemented

### Notable design decision:

Scalar-collection operator overloads use std::is_arithmetic_v<S> instead of the Scalar concept to avoid recursive constraint evaluation. When dense_vector
has operator+/*/- defined, GCC tries to check if it satisfies Scalar (which requires a * b), creating infinite recursion. This will be revisited in Phase 3
when expression templates provide a cleaner architecture.

## Phase 3: ITL Krylov Solvers and Supporting Infrastructure

### Context

Phase 1 delivered core types (`dense_vector`, `dense2D`). Phase 2 delivered eager arithmetic (operators, `dot`, `norms`, `mult`, `trans`, `scale`). Phase 3 delivers working ITL Krylov solvers so users can solve `Ax = b` with `cg(A, x, b, preconditioner, iteration)`.

**Design principle**: No expression templates. All operations are eager (sequential assignments). `lazy.hpp` and expression template stubs are deferred. The solvers work with any type satisfying `LinearOperator` and `Preconditioner` concepts.

**Out of scope**: Sparse matrices (`compressed2D`), advanced preconditioners (`ic_0`, `ilu_0`), smoothers, GMRES/QMR/TFQMR/IDR(s)/BiCG, expression templates.

## Implementation Steps

### Step 1: `operation/diagonal.hpp` — Extract Diagonal from Matrix

Fill the stub. Returns `dense_vector<T>` with `A(i,i)` for `i` in `0..min(rows,cols)-1`.

```cpp
template <Matrix M>
auto diagonal(const M& A) {
    auto n = std::min(A.num_rows(), A.num_cols());
    vec::dense_vector<typename M::value_type> v(n);
    for (size_type i = 0; i < n; ++i) v(i) = A(i, i);
    return v;
}
```

In `namespace mtl`. Includes: `<algorithm>`, matrix concept, dense_vector.

### Step 2: `operation/resource.hpp` — Workspace Allocation Helper

Fill the stub. Returns `c.size()` — a named boundary for future distributed-vector extension.

```cpp
template <Collection C>
auto resource(const C& c) { return c.size(); }
```

In `namespace mtl`.

### Step 3: `itl/pc/identity.hpp` — Identity Preconditioner

Fill the stub. Satisfies `Preconditioner<identity<M>, V>`. Stores nothing.

- Constructor: `identity(const Matrix&)` — ignores input
- `solve(VectorOut& x, const VectorIn& b)` — copies `b` to `x` element-wise
- `adjoint_solve(...)` — same as `solve` (self-adjoint)

**Convention**: output first, input second — matches MTL5 `Preconditioner` concept `p.solve(x, b)` where `x` is non-const.

In `namespace mtl::itl::pc`.

### Step 4: `itl/pc/diagonal.hpp` — Jacobi (Diagonal) Preconditioner

Fill the stub. Stores `inv_diag[i] = 1 / A(i,i)` at construction.

- Constructor: `diagonal(const Matrix& A)` — extracts diagonal, inverts elements
- `solve(VectorOut& x, const VectorIn& b)` — `x(i) = inv_diag(i) * b(i)`
- `adjoint_solve(...)` — `x(i) = conj(inv_diag(i)) * b(i)`

Depends on `operation/diagonal.hpp`, `functor/scalar/conj.hpp`.

In `namespace mtl::itl::pc`.

### Step 5: `itl/iteration/basic_iteration.hpp` — Convergence Controller

Fill the stub. Core class used by all Krylov solvers.

**Class**: `basic_iteration<Real>` in `namespace mtl::itl`

**State**: `i_` (counter), `max_iter_`, `rtol_`, `atol_`, `my_norm_r0_`, `resid_`, `error_`, `is_finished_`

**Constructors**:
- From vector `r0`: computes `my_norm_r0_ = two_norm(r0)`
- From scalar `norm_r0`: stores directly

**Key methods**:
- `finished(const V& r)` — calls `two_norm(r)`, stores as `resid_`, tests convergence
- `finished(Real r)` — stores scalar, tests convergence
- Convergence: `resid_ <= rtol_ * my_norm_r0_ || resid_ <= atol_`
- `operator++()` — increments `i_`
- `first()` — returns `i_ <= 1`
- `operator int()` — returns error code (0=converged, 1=max_iter)
- Getters: `iterations()`, `resid()`, `relresid()`, `tol()`, `atol()`, `norm_r0()`
- `fail(int code, string msg)` — for solver breakdown reporting

Fields are `protected` so `cyclic_iteration` can access them.

### Step 6: `itl/iteration/cyclic_iteration.hpp` — Periodic Logging

Fill the stub. Derives from `basic_iteration<Real>`.

- Adds: `cycle_` (print interval), `out_` (ostream reference, default `std::cout`)
- Overrides `finished(r)`: calls super, prints residual every `cycle_` iterations
- Overrides `error_code()`: prints convergence summary

### Step 7: `itl/iteration/noisy_iteration.hpp` — Per-Iteration Logging

Fill the stub. Derives from `cyclic_iteration<Real>` with `cycle = 1`. Trivial.

### Step 8: `itl/krylov/cg.hpp` — Conjugate Gradient Solver

Fill the stub. Free function implementing preconditioned CG.

**Signature**:
```cpp
template <typename LinearOp, typename VecX, typename VecB, typename PC, typename Iter>
int cg(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter)
```

**Algorithm** (always preconditioned — identity PC degenerates to unpreconditioned):
```
r = b - A*x
M.solve(z, r)
rho = dot(r, z)
while (!iter.finished(two_norm(r))):
    ++iter
    if first: p = z  else: p = z + (rho/rho_1) * p
    q = A * p
    alpha = rho / dot(p, q)
    x += alpha * p
    rho_1 = rho
    r -= alpha * q
    M.solve(z, r)
    rho = dot(r, z)
return iter
```

Workspace: `r`, `z`, `p`, `q` — all `dense_vector<Scalar>(x.size())`.

### Step 9: `itl/krylov/bicgstab.hpp` — BiCGSTAB Solver

Fill the stub. Free function for non-symmetric systems.

**Signature**: same pattern as CG but `bicgstab(...)`.

**Algorithm**:
```
r = b - A*x; r_star = r
rho_1=1, alpha=1, omega=1; v=0, p=0
while (!iter.finished(r)):
    ++iter
    rho = dot(r_star, r)
    if rho==0: fail(2), return
    beta = (rho/rho_1)*(alpha/omega)
    p = r + beta*(p - omega*v)
    M.solve(phat, p);  v = A*phat
    alpha = rho / dot(r_star, v)
    s = r - alpha*v
    if converged(s): x += alpha*phat; break
    M.solve(shat, s);  t = A*shat
    omega = dot(t,s) / dot(t,t)
    x += alpha*phat + omega*shat
    r = s - omega*t;  rho_1 = rho
return iter
```

Workspace: `r`, `r_star`, `p`, `phat`, `v`, `s`, `shat`, `t` — 8 vectors.

### Step 10: Update Umbrellas

**`mtl.hpp`** — add:
- `#include <mtl/operation/diagonal.hpp>`
- `#include <mtl/operation/resource.hpp>`
- `#include <mtl/itl/itl.hpp>`

**`itl/itl.hpp`** — already includes all stubs; no changes needed.

### Step 11: Tests

**Update `tests/unit/itl/test_cg.cpp`** — replace placeholder with:
- CG convergence on 3x3 SPD system `A={{4,1,0},{1,3,1},{0,1,2}}` with identity PC
- CG with diagonal PC on 1D Laplacian tridiagonal
- `basic_iteration` error code when max_iter exceeded
- `noisy_iteration` prints to stringstream

**Create `tests/unit/itl/test_bicgstab.cpp`**:
- BiCGSTAB on non-symmetric 3x3 system with identity PC
- BiCGSTAB with diagonal PC

**Update `tests/unit/CMakeLists.txt`** — register `test_bicgstab`.

## Files Modified (3)

| File | Change |
|------|--------|
| `include/mtl/mtl.hpp` | Add 3 includes: diagonal, resource, itl |
| `tests/unit/itl/test_cg.cpp` | Replace placeholder with real convergence tests |
| `tests/unit/CMakeLists.txt` | Register `test_bicgstab` |

## Files Filled (existing stubs → implementation, 10)

| File | Content |
|------|---------|
| `include/mtl/operation/diagonal.hpp` | Extract matrix diagonal → vector |
| `include/mtl/operation/resource.hpp` | `resource(c)` → `c.size()` |
| `include/mtl/itl/pc/identity.hpp` | Identity preconditioner |
| `include/mtl/itl/pc/diagonal.hpp` | Jacobi (diagonal) preconditioner |
| `include/mtl/itl/iteration/basic_iteration.hpp` | Convergence controller |
| `include/mtl/itl/iteration/cyclic_iteration.hpp` | Periodic-print iteration |
| `include/mtl/itl/iteration/noisy_iteration.hpp` | Every-iteration print |
| `include/mtl/itl/krylov/cg.hpp` | Conjugate Gradient solver |
| `include/mtl/itl/krylov/bicgstab.hpp` | BiCGSTAB solver |
| `include/mtl/itl/pc/solver.hpp` | Thin free-function `solve(P, b)` wrapper |

## Files Created (1)

| File | Purpose |
|------|---------|
| `tests/unit/itl/test_bicgstab.cpp` | BiCGSTAB convergence tests |

## NOT Touched (deferred)

- Expression templates: `lazy.hpp`, `vec_expr.hpp`, `mat_expr.hpp`, etc.
- Sparse matrices: `compressed2D.hpp`, `coordinate2D.hpp`, `ell_matrix.hpp`
- Advanced preconditioners: `ic_0.hpp`, `ilu_0.hpp`
- Smoothers: `gauss_seidel.hpp`, `jacobi.hpp`, `sor.hpp`
- Other Krylov solvers: `gmres.hpp`, `qmr.hpp`, `tfqmr.hpp`, `idr_s.hpp`, `bicg.hpp`

## Verification

```bash
cmake -B build -DCMAKE_CXX_STANDARD=20
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected: all 11 existing tests pass, updated `test_cg` passes with real convergence verification, new `test_bicgstab` passes, `hello_mtl5` still runs.

### Phase 3 is complete. Here's a summary of what was implemented:

10 stub files filled with implementations:

|                File                |                   Content                    |
| operation/diagonal.hpp             | Extracts matrix diagonal into dense_vector   |
| operation/resource.hpp             | resource(c) → c.size() workspace helper      |
| itl/pc/identity.hpp                | Identity (no-op) preconditioner              |
| itl/pc/diagonal.hpp                | Jacobi preconditioner (stores 1/diag(A))     |
| itl/pc/solver.hpp                  | Free-function solve(P, x, b) wrapper         |
| itl/iteration/basic_iteration.hpp  | Convergence controller (residual + max_iter) |
| itl/iteration/cyclic_iteration.hpp | Periodic residual printing                   |
| itl/iteration/noisy_iteration.hpp  | Every-iteration printing (cycle=1)           |
| itl/krylov/cg.hpp                  | Preconditioned Conjugate Gradient            |
| itl/krylov/bicgstab.hpp            | BiCGSTAB for non-symmetric systems           |

  3 files modified:
  - mtl.hpp — added diagonal, resource, and itl/itl.hpp includes
  - tests/unit/itl/test_cg.cpp — replaced placeholder with 4 real tests (CG convergence with identity/Jacobi PC, max_iter detection, noisy_iteration output)
  - tests/unit/CMakeLists.txt — registered test_bicgstab

  1 file created:
  - tests/unit/itl/test_bicgstab.cpp — 3 tests (non-symmetric, diagonal PC, SPD)

  Result: 12/12 tests pass, all solvers fully eager (no expression templates).

## Phase 4: Sparse Matrices + Extended ITL

## Context

Phases 1-3 delivered core types, eager arithmetic, and basic ITL solvers (CG, BiCGSTAB). Phase 4 adds `compressed2D` (CRS sparse matrix) with an inserter pattern, two more Krylov solvers (BiCG, GMRES), and three smoothers — enabling users to solve sparse `Ax = b` efficiently.

**Design principle**: Eager operations only (no expression templates). Row-major CRS only for Phase 4.

**Out of scope**: coordinate2D, ell_matrix, sparse_vector, ilu_0, ic_0 (need triangular solvers), tfqmr/qmr/idr_s, expression templates, col_major CCS.

## Implementation Steps

### Step 1: `mat/compressed2D.hpp` — CRS Sparse Matrix

Fill the stub. Three-array CRS storage: `data_` (values), `indices_` (column indices), `starts_` (row pointers, length `nrows+1`).

**Public API:**
- Constructors: default, `(nrows, ncols)`, raw CSR `(nrows, ncols, nnz, starts*, indices*, data*)`
- `operator()(r, c) const` — binary search in row; returns zero for absent entries
- `num_rows()`, `num_cols()`, `size()`, `nnz()`
- `ref_major()`, `ref_minor()`, `ref_data()` — const/mutable references to internal arrays
- `change_dim()`, `make_empty()`

**Traits:** `category<compressed2D> → tag::sparse`, `ashape → mat<V>`
**Alias:** `namespace mtl { using mat::compressed2D; }`

### Step 2: `mat/inserter.hpp` — RAII Sparse Builder

Fill the stub. Contains updater functors and the inserter class.

**Updater functors** (in `mtl::mat` namespace):
- `update_store<T>`: `a = b` (overwrite, default)
- `update_plus<T>`: `a += b` (accumulate)

**`compressed2D_inserter<Value, Parameters, Updater>`:**
- Constructor: allocates `slot_size` slots per row in flat working arrays
- `operator[](row)` returns `row_proxy`; `row_proxy[col]` returns `col_proxy`; `col_proxy << val` inserts
- `do_insert(r, c, val)`: binary search in row slot, update or insert; overflow to `std::map`
- Destructor calls `finalize()`: merge slots + overflow → final sorted CRS arrays in matrix

**Convenience alias:** `template <Matrix, Updater> using inserter = compressed2D_inserter<...>;`

### Step 3: Refine `SparseMatrix` Concept

**File:** `concepts/matrix.hpp`

Change `SparseMatrix` from `= Matrix<T>` to require `category_t<T> == tag::sparse`. This makes sparse operator overloads win by concept subsumption. Also refine `DenseMatrix` symmetrically.

```cpp
template <typename T>
concept SparseMatrix = Matrix<T> && std::is_same_v<traits::category_t<T>, tag::sparse>;
template <typename T>
concept DenseMatrix = Matrix<T> && std::is_same_v<traits::category_t<T>, tag::dense>;
```

### Step 4: `transposed_view` Extensions

**File:** `mat/view/transposed_view.hpp`

- Add `const Matrix& base() const { return ref_; }` — needed for O(nnz) transposed sparse matvec
- Add traits specialization: `category<transposed_view<compressed2D<V,P>>> → tag::sparse`

### Step 5: Sparse Matvec Operators

**File:** `mat/operators.hpp` — add two overloads after existing operators:

**CRS matvec** (`compressed2D * dense_vector`): O(nnz) — iterate `starts[r]..starts[r+1]`, accumulate `data[k] * x(indices[k])`. Concrete template types (not concept-constrained) so they win over generic `Matrix * Vector`.

**Transposed CRS matvec** (`transposed_view<compressed2D> * dense_vector`): O(nnz) — scatter pattern: `y(indices[k]) += data[k] * x(i)` using `At.base()` to access underlying CRS. Requires Step 4's `base()` accessor.

### Step 6: `operation/givens.hpp` — Givens Rotation Utility

Fill the stub. Two free functions in `namespace mtl`:

- `apply_givens_rotation(H, g, c, s, k)` — compute new Givens rotation for column k of Hessenberg H, apply to H and RHS vector g
- `apply_stored_rotation(H, c, s, i, k)` — apply previously stored rotation to column k at rows i, i+1

### Step 7: `itl/krylov/bicg.hpp` — BiConjugate Gradient

Fill the stub. Same API pattern: `bicg(A, x, b, M, iter)`.

Algorithm: maintains shadow residuals `r_tilde`, `z_tilde`, `p_tilde`. Uses `trans(A) * p_tilde` (O(nnz) for sparse via Step 5) and `M.adjoint_solve(z_tilde, r_tilde)`. Workspace: 8 vectors.

### Step 8: `itl/krylov/gmres.hpp` — GMRES with Restart

Fill the stub. Two functions:
- `gmres_inner(A, x, b, M, iter, kmax)` — single GMRES cycle (static/internal)
- `gmres(A, x, b, M, iter, restart=30)` — outer restart loop

Left-preconditioned: `w = M^{-1}(A * V[k])`. Arnoldi basis as `std::vector<dense_vector<T>>`. Hessenberg matrix as `dense2D<T>`. Modified Gram-Schmidt with reorthogonalization. Givens rotations from Step 6. Manual upper-triangular back-substitution.

### Step 9: Smoothers — `gauss_seidel.hpp`, `jacobi.hpp`, `sor.hpp`

Fill all three stubs in `itl/smoother/`.

**Generic version** (works with any Matrix via `A(i,j)`):
- `gauss_seidel<M>`: in-place row sweep, `x[i] = dia_inv[i] * (b[i] - sum_{j≠i} A(i,j)*x[j])`
- `jacobi<M>`: same but allocates `x_new`, copies result at end (true Jacobi semantics)
- `sor<M>`: relaxed GS with `omega` parameter, `x[i] = omega * GS_update + (1-omega) * x[i]`

**`compressed2D` specialization** for each: uses CRS raw access (`ref_major/minor/data`) for O(nnz) sweeps. Precomputes `dia_pos_[i]` (offset of diagonal in CRS arrays).

Constructor stores `const Matrix& A_`, `dia_inv_` vector, and `dia_pos_` for sparse version.
API: `operator()(x, b)` — single sweep, returns `x&`.

### Step 10: Update Umbrellas

**`mtl.hpp`** — add:
```cpp
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/operation/givens.hpp>
```

**`itl/itl.hpp`** — already includes all smoother/solver stubs; no changes needed.

### Step 11: Tests

**Create `tests/unit/mat/test_compressed2D.cpp`:**
- Raw CSR construction, element access, nnz
- Inserter store mode: insert entries, verify
- Inserter accumulate mode (update_plus): insert same entry twice
- `operator()` returns zero for absent elements
- `SparseMatrix` concept satisfied
- Sparse matvec matches dense matvec on same matrix
- Transposed sparse matvec correctness

**Create `tests/unit/itl/test_bicg.cpp`:**
- BiCG on non-symmetric 3×3 dense system
- BiCG on sparse tridiagonal (via inserter)

**Create `tests/unit/itl/test_gmres.cpp`:**
- GMRES on 3×3 non-symmetric system
- GMRES with restart on 10×10 system
- GMRES with diagonal PC

**Create `tests/unit/itl/test_smoothers.cpp`:**
- Gauss-Seidel reduces residual (dense and sparse)
- Jacobi reduces residual
- SOR with omega=1.0 matches Gauss-Seidel
- Sparse specialization matches generic version

**Update `tests/unit/CMakeLists.txt`** — register 4 new tests.

## Files Summary

### Stubs Filled (9)

| File | Content |
|------|---------|
| `include/mtl/mat/compressed2D.hpp` | CRS sparse matrix |
| `include/mtl/mat/inserter.hpp` | Updater functors + RAII inserter |
| `include/mtl/operation/givens.hpp` | Givens rotation utilities |
| `include/mtl/itl/krylov/bicg.hpp` | BiConjugate Gradient solver |
| `include/mtl/itl/krylov/gmres.hpp` | GMRES with restart |
| `include/mtl/itl/smoother/gauss_seidel.hpp` | Gauss-Seidel smoother |
| `include/mtl/itl/smoother/jacobi.hpp` | Jacobi smoother |
| `include/mtl/itl/smoother/sor.hpp` | SOR smoother |

### Files Modified (5)

| File | Change |
|------|--------|
| `include/mtl/concepts/matrix.hpp` | Refine `SparseMatrix`/`DenseMatrix` concepts |
| `include/mtl/mat/view/transposed_view.hpp` | Add `base()` accessor + sparse category trait |
| `include/mtl/mat/operators.hpp` | Add sparse matvec + transposed sparse matvec |
| `include/mtl/mtl.hpp` | Add compressed2D, inserter, givens includes |
| `tests/unit/CMakeLists.txt` | Register 4 new test executables |

### Files Created (4)

| File | Purpose |
|------|---------|
| `tests/unit/mat/test_compressed2D.cpp` | Sparse matrix + inserter + matvec tests |
| `tests/unit/itl/test_bicg.cpp` | BiCG convergence tests |
| `tests/unit/itl/test_gmres.cpp` | GMRES convergence tests |
| `tests/unit/itl/test_smoothers.cpp` | Smoother tests (dense + sparse) |

### NOT Touched (deferred to Phase 5)

- `coordinate2D`, `ell_matrix`, `sparse_vector`
- `ilu_0`, `ic_0` (need triangular solvers)
- `tfqmr`, `qmr`, `idr_s` (less common solvers)
- Expression templates
- Col-major CCS support

## Verification

```bash
cmake -B build -DCMAKE_CXX_STANDARD=20
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected: all 12 existing tests pass + 4 new tests (test_compressed2D, test_bicg, test_gmres, test_smoothers) = 16 total.

## Phase 4 is complete. Here's a summary of everything implemented:

Stubs filled (9 files):

  - mat/compressed2D.hpp — CRS sparse matrix with three-array storage, raw CSR constructor, binary search element access
  - mat/inserter.hpp — RAII sparse builder with update_store/update_plus functors, slot-based insertion with overflow to std::map
  - operation/givens.hpp — Givens rotation utilities for GMRES
  - itl/krylov/bicg.hpp — BiConjugate Gradient solver with shadow residuals and trans(A) support
  - itl/krylov/gmres.hpp — GMRES with restart, Modified Gram-Schmidt, Givens QR factorization
  - itl/smoother/gauss_seidel.hpp — Generic + compressed2D specialization
  - itl/smoother/jacobi.hpp — Generic + compressed2D specialization
  - itl/smoother/sor.hpp — SOR with omega parameter, generic + compressed2D specialization

Files modified (5):

  - concepts/matrix.hpp — SparseMatrix/DenseMatrix now check category_t tag
  - mat/view/transposed_view.hpp — Added base() accessor, category inherits from underlying matrix
  - mat/operators.hpp — Added O(nnz) sparse matvec and transposed sparse matvec
  - mtl.hpp — Added compressed2D, inserter, givens includes
  - tests/unit/CMakeLists.txt — Registered 4 new test executables

Tests created (4 files):

  - test_compressed2D.cpp — 7 test cases covering CSR construction, inserter, concepts, sparse/transposed matvec
  - test_bicg.cpp — 2 test cases: dense non-symmetric + sparse tridiagonal
  - test_gmres.cpp — 3 test cases: basic, restart, diagonal preconditioner
  - test_smoothers.cpp — 5 test cases: GS dense/sparse, Jacobi, SOR=GS, sparse=generic match

Result: 16/16 tests pass.

