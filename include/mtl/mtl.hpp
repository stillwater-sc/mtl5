#pragma once
// MTL5 — Kitchen-sink umbrella include
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
#include <mtl/mat/dense2D.hpp>

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

// Typed functors
#include <mtl/functor/typed/scale.hpp>
#include <mtl/functor/typed/rscale.hpp>
#include <mtl/functor/typed/divide_by.hpp>

// Views
#include <mtl/mat/view/transposed_view.hpp>

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

// Operators
#include <mtl/operation/operators.hpp>
