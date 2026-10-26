# Quaternions, Lie Algebras and Matrix Functions in MTL5

**Date**: 2026-10-25
**Status**: Assessment. The recommended work, `expm`/`logm`/`sqrtm`, is #650.

The question was whether MTL5 should have, or know about, quaternions and Lie
algebras. Neither belongs in MTL5 as a type of its own. Both point to something
MTL5 should have and doesn't: functions of a matrix, starting with the matrix
exponential.

## Quaternions

"Quaternion" means two different things.

### As a number type

A quaternion is a scalar, in the same position as a posit or an LNS value: a type
MTL5's containers and algorithms are parameterized over, not a type MTL5
defines. It belongs in Universal or a small peer library. MTL5 stays free of
number-type libraries, as it is Universal-free today.

### As a rotation

Normalization, slerp, conversion to and from a 3×3 rotation matrix, and
composition of rotations are geometry. That is a layer above linear algebra,
the way Eigen keeps its Geometry module apart from its core. It is out of
MTL5's scope.

### What MTL5 should know about: a scalar that does not commute

The quaternions are the standard example of a scalar whose multiplication does
not commute: a·b ≠ b·a in general. Each kind of operation depends on that
differently:

- **Addition, scaling and matrix multiplication** work over such a scalar as
  long as every kernel keeps operand order: `C(i,j) += A(i,k) * B(k,j)`, never
  `B(k,j) * A(i,k)`.
- **Solves, LU and Householder** depend on the order explicitly. Left and right
  division differ, and a conjugate transpose must reverse products.

MTL5's `Field` concept (`include/mtl/concepts/scalar.hpp`) is syntactic: it
requires `+`, `-`, `*`, `/` and `T{0}`, and says nothing about commutativity. A
quaternion type would satisfy it. Whether MTL5's kernels and factorizations
preserve operand order has not been checked. A test with a quaternion element
type would answer that, and it would catch any kernel that swaps operands
silently. The same test guards the SIMD and blocked GEMM paths against
reordering a product.

## Lie algebras

What Lie-group code needs from a linear algebra library is mostly general:

- **The exponential map** of a matrix Lie group is the matrix exponential of the
  algebra element, and **the logarithm map** is the matrix logarithm.
- **Jacobians and sensitivities** come from the Fréchet derivatives of those
  functions.

The group-specific parts are robotics and geometry code: Rodrigues' formula for
SO(3), the closed-form SE(3) exponential, hat and vee maps, and left and right
Jacobians. That is a layer like Sophus or manif, built on top of the general
matrix functions. Such a layer could live in a peer repository, as mp-spice
does for circuit simulation.

## What MTL5 is missing: matrix functions

MTL5 has no functions of a matrix:

- `mtl::exp(Matrix)` in `include/mtl/operation/exp.hpp` is element-wise, as are
  `log` and `sqrt`.
- `eigenvalue` runs the Francis double-shift QR to real Schur form but returns
  only the eigenvalues. Neither the Schur vectors nor `T` are available.

| Function | Method | Needs |
|---|---|---|
| `expm` | Padé scaling and squaring (Higham 2005; Al-Mohy & Higham 2009) | GEMM, LU solve, 1-norms |
| `sqrtm` | Schur method (Björck & Hammarling 1983; real form, Higham 1987) | `schur(A)` returning `Q` and `T` |
| `logm` | inverse scaling and squaring (Al-Mohy & Higham 2012) | `schur`, `sqrtm` on triangular matrices |
| Fréchet derivatives | Al-Mohy & Higham 2009 | `expm`'s Padé pieces; triangular Sylvester |
| `expm_multiply` | Al-Mohy & Higham 2011 | GEMV/SpMV, 1-norm estimation |

These serve more than Lie groups:
- linear ODEs and exponential integrators;
- discretization of continuous-time control systems;
- continuous-time Markov chains;
- network communicability;
- polar decompositions and matrix means.

The names `expm`, `logm` and `sqrtm` follow MATLAB and SciPy, and keep them
apart from the element-wise `exp`, `log` and `sqrt`.

### Why this fits MTL5's goal

How accurate scaling and squaring is depends on the Padé degree `m` and the
scaling `s`. Both are chosen against tables of `θ_m`, which are derived from
the unit roundoff; Higham (2005) derives them for IEEE double precision. MTL5
runs on float, double, double-double, posits and other custom types, so the
degree selection has to follow `numeric_limits<T>::epsilon()` rather than a
fixed table. Fasi & Higham (2019) give an arbitrary-precision scaling and
squaring algorithm that takes the unit roundoff as input. Choosing precisions
within the computation is the kind of mixed-precision algorithm design MTL5
exists for.

## Recommendation

1. **File an issue for `expm`, `logm` and `sqrtm`**, phased:
   - `expm` first, since it needs only GEMM and LU;
   - then a `schur(A)` that returns `Q` and `T`;
   - then `sqrtm` and `logm`;
   - optionally the Fréchet derivatives and `expm_multiply`.
2. **Add a test with a non-commutative scalar** to find any hidden assumption
   that multiplication commutes. A minimal quaternion type defined in the test
   is enough; MTL5 does not ship it.
3. **Add no quaternion type and no Lie-group classes to MTL5.** Point users to a
   geometry layer built on the matrix functions.

## References

- N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008.
- N. J. Higham, "The scaling and squaring method for the matrix exponential
  revisited", SIAM J. Matrix Anal. Appl. 26(4), 1179–1193, 2005.
- A. H. Al-Mohy and N. J. Higham, "A new scaling and squaring algorithm for the
  matrix exponential", SIAM J. Matrix Anal. Appl. 31(3), 970–989, 2009.
- A. H. Al-Mohy and N. J. Higham, "Improved inverse scaling and squaring
  algorithms for the matrix logarithm", SIAM J. Sci. Comput. 34(4), C152–C169,
  2012.
- Å. Björck and S. Hammarling, "A Schur method for the square root of a
  matrix", Linear Algebra Appl. 52/53, 127–140, 1983.
- M. Fasi and N. J. Higham, "An arbitrary precision scaling and squaring
  algorithm for the matrix exponential", SIAM J. Matrix Anal. Appl. 40(4),
  1233–1256, 2019.
