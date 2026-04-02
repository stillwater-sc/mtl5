# Aggregate Types: Vector, Matrix, Tensor, and N-Dimensional Array

## Overview

MTL5 provides a hierarchy of aggregate numerical types that range from simple one-dimensional collections to general N-dimensional containers. Understanding the distinction between these types — and especially the difference between **tensors** and **multi-dimensional arrays** — is essential for choosing the right abstraction.

| Type | Rank | Mathematical Object | Primary Interface |
|---|---|---|---|
| `vec::dense_vector<T>` | 1 | Vector in R^n | Linear algebra |
| `mat::dense2D<T>` | 2 | Matrix in R^(m x n) | Linear algebra |
| `tensor<T, Rank, Dim>` | N | Tensor with index structure | Einstein notation |
| `ndarray<T, N>` | N | N-dimensional container | NumPy-compatible |

## Existing Types

### Vectors (`vec::`)

MTL5 vectors are one-dimensional collections with linear algebra semantics:

```cpp
vec::dense_vector<double> v(100);       // dense, contiguous
vec::sparse_vector<double> s(100);      // sparse, index-value pairs
```

Vectors support inner products, norms, element-wise operations, and participate in matrix-vector expressions via expression templates.

### Matrices (`mat::`)

MTL5 matrices are two-dimensional collections with storage-aware dispatch:

```cpp
mat::dense2D<double> A(m, n);                          // dense, row or column major
mat::compressed2D<double> S(m, n);                     // CSR/CSC sparse
mat::coordinate2D<double> C(m, n);                     // COO (triplet)
mat::ell_matrix<double> E(m, n, max_nnz_per_row);     // ELLPACK
```

Matrices support decompositions (LU, QR, Cholesky, SVD), iterative solvers, expression templates, and optional BLAS/LAPACK dispatch.

## Tensors vs. Multi-Dimensional Arrays

These two concepts are frequently conflated but are fundamentally different:

### Tensors are mathematical objects with transformation rules

A **tensor** of rank (p, q) is a multilinear map that transforms in a specific way under changes of basis. The key properties are:

- **Index structure matters**: indices are either *covariant* (subscript, transforms with basis) or *contravariant* (superscript, transforms inversely). A rank-(1,1) tensor T^i_j is not the same object as a rank-(0,2) tensor T_ij even though both have two indices.
- **Einstein summation**: repeated indices (one up, one down) imply contraction: `C^i_k = A^i_j * B^j_k`. This is not just a notational convenience — it encodes the mathematical structure of the operation.
- **Coordinate invariance**: the physical/geometric quantity a tensor represents is independent of the coordinate system. The *components* change under basis transformations, but the tensor itself does not.

Tensors arise in continuum mechanics (stress, strain, elasticity), electrodynamics (electromagnetic field tensor), general relativity (Riemann curvature), and machine learning (though ML "tensors" are really just ndarrays).

### Multi-dimensional arrays are indexed containers

An **ndarray** is a data structure: an N-dimensional block of memory with shape, strides, and element access. The key properties are:

- **No index semantics**: all axes are interchangeable. There is no distinction between covariant and contravariant.
- **Broadcasting**: operations between arrays of different shapes follow NumPy broadcasting rules — a powerful but purely computational concept with no tensor analogue.
- **Reshaping and slicing**: the same data can be viewed with different shapes without copying. This is a memory layout concept, not a mathematical one.
- **Strides**: generalized memory layouts (C-order, Fortran-order, arbitrary strides) for zero-copy views and interop with external libraries.

## Design: `tensor<T, Rank, Dim>`

The tensor class models true mathematical tensors with compile-time index structure:

```cpp
// Rank-2 tensor in 3D space (e.g., stress tensor)
tensor<double, 2, 3> sigma;              // 3x3 components

// Mixed-rank tensor: 1 contravariant, 1 covariant index
// using index type tags
tensor<double, index<up, down>, dim<3>> T;

// Einstein summation via expression templates
// C^i_k = A^i_j * B^j_k  (contraction over j)
auto C = contract(A(i, j), B(j, k));    // repeated index j → summation
```

### Key features

- **Compile-time index structure**: covariant/contravariant indices encoded in the type system, preventing illegal contractions at compile time
- **Einstein notation**: expression templates for index contraction, producing correct results under the summation convention
- **Metric tensor support**: raising/lowering indices via a metric `g_ij`
- **Symmetry exploitation**: symmetric and antisymmetric tensors store only independent components
- **Custom scalar types**: works with posits, LNS, and other Universal number types for mixed-precision tensor computations

### Target applications

- Finite element analysis (stress/strain tensors, elasticity tensor C_ijkl)
- Computational fluid dynamics (Reynolds stress tensor)
- Differential geometry and general relativity
- Continuum mechanics

## Design: `ndarray<T, N>`

The ndarray class models NumPy-style multi-dimensional arrays with full Python interop:

```cpp
// Static-rank array (like xtensor)
ndarray<double, 3> volume({128, 128, 64});

// Element access
volume(i, j, k) = 1.0;

// Slicing (returns a view, no copy)
auto slice = volume(all, all, 32);       // 2D slice at k=32

// Broadcasting
ndarray<double, 1> bias({64});
auto result = volume + bias;             // broadcasts bias across first two axes

// Reshaping (view, no copy when contiguous)
auto flat = volume.reshape({128 * 128 * 64});
```

### Key features

- **NumPy-compatible semantics**: shape, strides, broadcasting, slicing, reshaping — matching NumPy behavior exactly so that Python users encounter no surprises
- **Static and dynamic rank**: `ndarray<T, N>` has compile-time rank N; a separate `xarray<T>` variant supports dynamic rank for maximum flexibility
- **Zero-copy views**: slicing and reshaping return views over the same memory
- **Stride-aware**: supports C-order (row-major), Fortran-order (column-major), and arbitrary strides for interop with external data
- **Custom scalar types**: the killer differentiator vs. NumPy — run array computations with posit, lns, cfloat, or any Universal number type

### Connecting vector and matrix

`vec::dense_vector<T>` and `mat::dense2D<T>` are conceptually rank-1 and rank-2 specializations of ndarray. The design provides interop without forcing everything through a single type:

```cpp
// Implicit conversion / view construction
ndarray<double, 1> a(v);                // view of dense_vector as 1D array
ndarray<double, 2> B(M);               // view of dense2D as 2D array

// Or use the linear algebra types directly — they model the ndarray concept
// so generic ndarray algorithms work on them
template <NdArray A>
auto sum(const A& arr);                 // works on vector, matrix, and ndarray
```

### Python integration via pybind11

The primary motivation for ndarray is seamless Python interop, following the xtensor-python model:

```cpp
#include <mtl/py/ndarray.hpp>           // pybind11 + buffer protocol

// Automatic conversion: NumPy array ↔ mtl::ndarray
m.def("solve", [](py_ndarray<double, 2> A, py_ndarray<double, 1> b) {
    // A and b share memory with the NumPy arrays — zero copy
    return mtl::operation::solve(A, b);
});
```

```python
import numpy as np
import mtl5

A = np.array([[1, 2], [3, 4]], dtype=np.float64)
b = np.array([5, 6], dtype=np.float64)
x = mtl5.solve(A, b)       # returns NumPy array, computed by MTL5
```

For custom number types, the Python side uses a registered dtype (similar to how ml_dtypes registers bfloat16):

```python
from universal import posit16
import mtl5

A = np.array([[1, 2], [3, 4]], dtype=posit16)
x = mtl5.solve(A, b)       # MTL5 solves in posit<16,2> arithmetic
```

### Integration with SciPy

SciPy's sparse matrices and linear algebra routines expect specific interfaces. MTL5 can participate by:

1. **scipy.sparse interop**: convert between `mat::compressed2D` and `scipy.sparse.csr_matrix` / `csc_matrix` via the buffer protocol
2. **LinearOperator protocol**: expose MTL5 matrix types as `scipy.sparse.linalg.LinearOperator` for use with SciPy's iterative solvers
3. **Direct solver plugins**: register MTL5's sparse Cholesky/LU/QR as solvers for `scipy.sparse.linalg.spsolve`

## Comparison with xtensor

MTL5's ndarray design is informed by [xtensor](https://github.com/xtensor-stack/xtensor) but differs in key ways:

| Feature | xtensor | MTL5 ndarray |
|---|---|---|
| Static rank | `xt::xtensor<T, N>` | `ndarray<T, N>` |
| Dynamic rank | `xt::xarray<T>` | `xarray<T>` |
| Custom scalars | Limited | First-class (posit, lns, cfloat, ...) |
| Linear algebra | Separate (xtensor-blas) | Integrated (MTL5 operations + BLAS dispatch) |
| Sparse matrices | No | Full support (CSR, CSC, COO, ELLPACK) |
| Iterative solvers | No | Integrated (CG, GMRES, BiCGSTAB, ...) |
| Tensor algebra | No | Separate `tensor<>` with Einstein notation |
| Expression templates | Yes (lazy) | Yes (lazy, compatible with MTL5 expressions) |
| Python bindings | xtensor-python | pybind11 buffer protocol |

The key differentiator is that MTL5 combines N-dimensional array computation with a full linear algebra stack and custom arithmetic types — something no single existing library provides.

## Roadmap

1. **ndarray core**: shape, strides, element access, views, broadcasting
2. **Vector/matrix interop**: concept-based bridge so existing MTL5 types work with ndarray algorithms
3. **Python bindings**: pybind11 module with NumPy buffer protocol
4. **tensor core**: index algebra, contraction, metric operations
5. **SciPy integration**: sparse matrix interop and solver registration
