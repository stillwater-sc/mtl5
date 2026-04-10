# MTL5 Examples

A progressive tutorial through the Matrix Template Library, organized by implementation phase. Each phase builds on the previous ones, introducing new types, operations, and solver techniques. Start at Phase 1 and work forward, or jump directly to the phase that matches your interest.

## Building

```bash
cmake --preset dev        # or: cmake -B build -DCMAKE_CXX_STANDARD=20
cmake --build build -j$(nproc)

# Run a specific example:
./build/examples/example_vectors_and_matrices
./build/examples/example_sparse_direct_solvers
```

All examples are auto-discovered by CMake. Each `.cpp` file produces a target named `example_<filename>`.

---

## intro/

**Getting started with MTL5.** A minimal "hello world" that prints the library version, demonstrates compile-time dimensions, and verifies that basic concepts (`Scalar`, `Field`) are satisfied.

| Example | Description |
|---------|-------------|
| `hello_mtl5.cpp` | Version info, fixed dimensions, concept checks |

---

## phase01_core_types/

**Vectors, matrices, and the C++20 concept system.** Introduces the fundamental data types -- dense vectors, dense matrices, and sparse matrices via the RAII inserter pattern. Demonstrates the concept hierarchy (`Scalar`, `Field`, `OrderedField`, `Matrix`, `Vector`, `DenseMatrix`, `SparseMatrix`) and the category traits that drive compile-time dispatch. Shows how custom number types plug into MTL5 by satisfying the same concepts.

| Example | Description |
|---------|-------------|
| `vectors_and_matrices.cpp` | Dense/sparse construction, element access, inserter pattern, math identities |
| `concepts_and_traits.cpp` | Concept hierarchy, category traits, mixed-type expressions, custom number types |

---

## phase02_basic_operations/

**Norms, dot products, and matrix arithmetic.** Covers the numerical building blocks: vector norms (1-norm, 2-norm, infinity-norm), inner products, matrix-vector and matrix-matrix multiplication for both dense and sparse matrices, element-wise operations (abs, sqrt, negate), transposed matvec, trace, and diagonal extraction. Demonstrates that dense and sparse matrices produce identical results through the same operator interface.

| Example | Description |
|---------|-------------|
| `norms_and_products.cpp` | Vector norms, dot products, dense/sparse matvec, element-wise ops, transposed multiply |
| `matrix_arithmetic.cpp` | Matrix add/subtract, scalar scaling, matrix-matrix multiply, trace, diagonal, dense-sparse consistency |

---

## phase03_iterative_solvers/

**Conjugate Gradient and BiCGSTAB on physical problems.** Applies iterative solvers to PDE-derived linear systems. The heat equation produces a symmetric positive definite (SPD) system solvable by CG; the convection-diffusion equation produces a non-symmetric system where CG fails and BiCGSTAB is needed. Introduces the pattern: discretize a PDE, assemble a sparse system, solve iteratively, interpret the solution physically.

| Example | Description |
|---------|-------------|
| `heat_equation_1d.cpp` | 1D steady-state heat equation solved by CG (SPD system) |
| `convection_diffusion.cpp` | Non-symmetric system from convection -- CG fails, BiCGSTAB succeeds |

---

## phase04_sparse_assembly/

**2D sparse assembly and stationary iterative methods.** Scales up to two dimensions with the 5-point Laplacian stencil, assembling larger sparse systems with the compressed2D inserter. Introduces GMRES for non-symmetric systems and compares classical smoothers -- Jacobi, Gauss-Seidel, and SOR -- showing how relaxation parameters affect convergence.

| Example | Description |
|---------|-------------|
| `laplacian_2d.cpp` | 2D Laplacian assembly with 5-point stencil, solved by GMRES |
| `smoother_convergence.cpp` | Jacobi, Gauss-Seidel, SOR convergence comparison |

---

## phase05_dense_decompositions/

**LU, Cholesky, and QR factorizations for direct solves.** Demonstrates the three fundamental dense matrix decompositions and when to use each: LU for general systems, Cholesky for SPD systems (2x faster), and QR for least-squares and ill-conditioned problems. Includes polynomial curve fitting as a practical least-squares application.

| Example | Description |
|---------|-------------|
| `solve_three_ways.cpp` | Side-by-side comparison of LU, Cholesky, and QR on the same system |
| `least_squares_qr.cpp` | Polynomial curve fitting via QR least-squares |

---

## phase06_eigenvalue_svd/

**Eigenvalues and Singular Value Decomposition.** Connects eigenvalue problems to physics (vibrating string frequencies) and data science (PCA dimensionality reduction). Compares numerically computed eigenvalues against analytical solutions and shows how SVD decomposes data into principal components.

| Example | Description |
|---------|-------------|
| `vibrating_string.cpp` | Eigenvalues of 1D Laplacian as vibration frequencies |
| `pca_svd.cpp` | Principal Component Analysis via SVD for data reduction |

---

## phase07_sparse_formats/

**Sparse storage formats and structured matrix views.** Compares COO (coordinate), CRS (compressed row), and ELLPACK storage with their space/time tradeoffs. Introduces structured views -- Hermitian, banded, upper/lower triangular -- that reinterpret existing storage without copying data.

| Example | Description |
|---------|-------------|
| `sparse_formats.cpp` | COO vs CRS vs ELL: assembly, storage costs, matvec performance |
| `structured_views.cpp` | Hermitian, banded, and triangular views on dense matrices |

---

## phase08_expression_templates/

**Lazy evaluation and cache efficiency.** Explains how MTL5's expression templates fuse element-wise operations into single-pass traversals, eliminating temporaries and improving cache utilization. Demonstrates the type system behind expressions and benchmarks fused vs unfused evaluation.

| Example | Description |
|---------|-------------|
| `expression_benchmark.cpp` | Cache-efficient fused operations vs manual loops |
| `expression_concepts.cpp` | Expression template mechanics: lazy capture, type inspection |

---

## phase09_advanced_types/

**Sparse vectors, block-recursive algorithms, and BLAS dispatch.** Introduces sparse vectors for high-dimensional sparse data, block-recursive matrix multiplication for cache-oblivious performance, and the compile-time dispatch architecture that routes operations to BLAS/LAPACK when available.

| Example | Description |
|---------|-------------|
| `sparse_vector.cpp` | Sparse vector construction, insertion, dot products, cropping |
| `recursive_traversal.cpp` | Block-recursive GEMM with cache-oblivious subdivision |
| `blas_dispatch.cpp` | Compile-time BLAS/LAPACK dispatch via `if constexpr` and traits |

---

## phase10_matrix_views/

**Triangular views and permutation matrices.** Demonstrates extracting upper, lower, strict-upper, and strict-lower views from matrices -- essential for working with LU and Cholesky factors. Shows permutation matrices for row/column reordering with O(n) matvec.

| Example | Description |
|---------|-------------|
| `triangular_views.cpp` | upper/lower/strict views, identity decomposition A = L + U |
| `permutation_block_diagonal.cpp` | Permutation matrices, row swaps, block-diagonal construction |

---

## phase11_assembly_reorder/

**FEM-style assembly and matrix reordering.** Demonstrates the shifted inserter for overlapping finite element assembly with automatic accumulation, and strided vector references for extracting columns from row-major storage. Shows how bandwidth-reducing reorderings (RCM) improve solver performance.

| Example | Description |
|---------|-------------|
| `fem_assembly_reorder.cpp` | Shifted inserter for FEM assembly, RCM bandwidth reduction |
| `strided_views_unit_vectors.cpp` | Unit vectors as basis, strided column extraction |

---

## phase12_transcendental_functions/

**Element-wise transcendental operations for applied mathematics.** Applies MTL5's vectorized math functions -- exp, log, sin, cos, tanh, erf -- to real-world domains: signal processing (modulated waveforms, spectral analysis), neural network activation functions (sigmoid, ReLU, GELU), radioactive decay chains, and coordinate transformations in 2D/3D geometry.

| Example | Description |
|---------|-------------|
| `signal_processing.cpp` | AM/FM modulation, spectral envelopes via sin/cos/exp |
| `activation_functions.cpp` | Sigmoid, tanh, ReLU, GELU from transcendental building blocks |
| `radioactive_decay.cpp` | Exponential decay, half-life, chemical kinetics via exp/log |
| `coordinate_geometry.cpp` | Polar/spherical coordinates, hyperbolic geometry via trig functions |

---

## phase13_krylov_multigrid/

**Advanced iterative solvers: Krylov subspace methods and multigrid.** Compares Krylov solvers -- CGS, QMR, TFQMR, IDR(s) -- on non-symmetric sparse systems, showing convergence characteristics and breakdown behavior. Demonstrates geometric multigrid with restriction, prolongation, and V-cycle smoothing for the 1D Poisson equation.

| Example | Description |
|---------|-------------|
| `krylov_comparison.cpp` | CGS, QMR, TFQMR, IDR(s) convergence on non-symmetric systems |
| `multigrid_poisson.cpp` | 3-level geometric multigrid V-cycle for 1D Poisson |

---

## phase14_io_generators/

**I/O utilities and test matrix generators.** Practical tools for data exchange and solver testing: CSV import/export for interoperability with Python/MATLAB, pretty-printed matrix output for debugging, the Poisson matrix generator for creating standard test systems, and diagonal extraction/construction utilities.

| Example | Description |
|---------|-------------|
| `csv_io_prettyprint.cpp` | CSV read/write, formatted matrix printing |
| `poisson_diag_toolkit.cpp` | Poisson generator, diagonal extraction, solver integration |

---

## phase15_sparse_direct/

**Sparse direct solvers with unified dispatch.** The culmination of MTL5's sparse direct solver infrastructure: native Cholesky (LL^T), LU (PA=LU with pivoting), and QR (Householder) factorizations with AMD and COLAMD fill-reducing orderings. Demonstrates the unified dispatch that automatically selects the best backend -- SuiteSparse (UMFPACK, CHOLMOD, SPQR) for production `double` systems, native solvers for custom number types. Shows fill-in reduction from different orderings on a 2D Laplacian.

| Example | Description |
|---------|-------------|
| `sparse_direct_solvers.cpp` | Unified dispatch demo: Cholesky/LU/QR on sparse and dense systems, backend selection, fill-in comparison (natural vs RCM vs AMD) |
