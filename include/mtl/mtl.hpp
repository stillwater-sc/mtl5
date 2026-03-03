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

// Operations
#include <mtl/operation/set_to_zero.hpp>
#include <mtl/operation/fill.hpp>
#include <mtl/operation/print.hpp>
#include <mtl/operation/size.hpp>
#include <mtl/operation/num_rows.hpp>
#include <mtl/operation/num_cols.hpp>
