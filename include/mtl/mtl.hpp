#pragma once
// MTL5 -- Kitchen-sink umbrella include
// Pulls in types, operations, and ITL

// Version and configuration
#include <mtl/version.hpp>
#include <mtl/config.hpp>

// Forward declarations
#include <mtl/mtl_fwd.hpp>

// Concepts
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/linear_operator.hpp>
#include <mtl/concepts/preconditioner.hpp>

// Tags
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/tag/shape.hpp>
#include <mtl/tag/traversal.hpp>
#include <mtl/tag/storage.hpp>

// Traits
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/traits/transposed_orientation.hpp>
#include <mtl/traits/flatcat.hpp>
#include <mtl/traits/is_expression.hpp>

// Math
#include <mtl/math/operations.hpp>
#include <mtl/math/identity.hpp>

// Detail
#include <mtl/detail/index.hpp>

// Matrix and vector parameters/dimensions
#include <mtl/mat/dimension.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/vec/dimension.hpp>
#include <mtl/vec/parameter.hpp>

// Core data types
#include <mtl/detail/contiguous_memory_block.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/sparse_vector.hpp>
#include <mtl/vec/inserter.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/identity2D.hpp>
#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/ell_matrix.hpp>
#include <mtl/mat/permutation_matrix.hpp>
#include <mtl/mat/block_diagonal2D.hpp>
#include <mtl/mat/shifted_inserter.hpp>
#include <mtl/vec/unit_vector.hpp>
#include <mtl/vec/strided_vector_ref.hpp>

// Scalar functors
#include <mtl/functor/scalar/plus.hpp>
#include <mtl/functor/scalar/minus.hpp>
#include <mtl/functor/scalar/times.hpp>
#include <mtl/functor/scalar/divide.hpp>
#include <mtl/functor/scalar/assign.hpp>
#include <mtl/functor/scalar/negate.hpp>
#include <mtl/functor/scalar/abs.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/functor/scalar/sqrt.hpp>
#include <mtl/functor/scalar/exp.hpp>
#include <mtl/functor/scalar/log.hpp>
#include <mtl/functor/scalar/exp2.hpp>
#include <mtl/functor/scalar/log2.hpp>
#include <mtl/functor/scalar/log10.hpp>
#include <mtl/functor/scalar/cbrt.hpp>
#include <mtl/functor/scalar/pow.hpp>
#include <mtl/functor/scalar/sin.hpp>
#include <mtl/functor/scalar/cos.hpp>
#include <mtl/functor/scalar/tan.hpp>
#include <mtl/functor/scalar/asin.hpp>
#include <mtl/functor/scalar/acos.hpp>
#include <mtl/functor/scalar/atan.hpp>
#include <mtl/functor/scalar/sinh.hpp>
#include <mtl/functor/scalar/cosh.hpp>
#include <mtl/functor/scalar/tanh.hpp>
#include <mtl/functor/scalar/asinh.hpp>
#include <mtl/functor/scalar/acosh.hpp>
#include <mtl/functor/scalar/atanh.hpp>
#include <mtl/functor/scalar/ceil.hpp>
#include <mtl/functor/scalar/floor.hpp>
#include <mtl/functor/scalar/round.hpp>
#include <mtl/functor/scalar/signum.hpp>
#include <mtl/functor/scalar/erf.hpp>
#include <mtl/functor/scalar/erfc.hpp>
#include <mtl/functor/scalar/real.hpp>
#include <mtl/functor/scalar/imag.hpp>

// Typed functors
#include <mtl/functor/typed/scale.hpp>
#include <mtl/functor/typed/rscale.hpp>
#include <mtl/functor/typed/divide_by.hpp>

// Views
#include <mtl/mat/view/transposed_view.hpp>
#include <mtl/mat/view/banded_view.hpp>
#include <mtl/mat/view/hermitian_view.hpp>
#include <mtl/mat/view/map_view.hpp>
#include <mtl/mat/view/upper_view.hpp>
#include <mtl/mat/view/lower_view.hpp>
#include <mtl/mat/view/strict_upper_view.hpp>
#include <mtl/mat/view/strict_lower_view.hpp>

// Operations
#include <mtl/operation/set_to_zero.hpp>
#include <mtl/operation/fill.hpp>
#include <mtl/operation/print.hpp>
#include <mtl/operation/size.hpp>
#include <mtl/operation/num_rows.hpp>
#include <mtl/operation/num_cols.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/sum.hpp>
#include <mtl/operation/product.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/operation/abs.hpp>
#include <mtl/operation/conj.hpp>
#include <mtl/operation/negate.hpp>
#include <mtl/operation/sqrt.hpp>
#include <mtl/operation/max.hpp>
#include <mtl/operation/min.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/diagonal.hpp>
#include <mtl/operation/resource.hpp>
#include <mtl/operation/givens.hpp>
#include <mtl/operation/trace.hpp>
#include <mtl/operation/random.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/householder.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/kron.hpp>
#include <mtl/operation/reorder.hpp>
#include <mtl/operation/transcendental.hpp>

// Recursion
#include <mtl/recursion/base_case_test.hpp>
#include <mtl/recursion/matrix_recursator.hpp>
#include <mtl/recursion/predefined_masks.hpp>

// External interfaces (conditional on CMake options)
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/interface/blas.hpp>
#include <mtl/interface/lapack.hpp>
#include <mtl/interface/umfpack.hpp>

// Expression templates
#include <mtl/mat/expr/mat_expr.hpp>
#include <mtl/mat/expr/dmat_expr.hpp>
#include <mtl/mat/expr/smat_expr.hpp>
#include <mtl/vec/expr/vec_expr.hpp>

// Operators
#include <mtl/operation/operators.hpp>

// Lazy evaluation and fusion
#include <mtl/operation/lazy.hpp>
#include <mtl/operation/fuse.hpp>

// I/O
#include <mtl/io/matrix_market.hpp>
#include <mtl/io/read_el.hpp>
#include <mtl/io/write_el.hpp>

// Generators -- Test matrix generation facility
#include <mtl/generators/generators.hpp>

// ITL -- Iterative Template Library
#include <mtl/itl/itl.hpp>
