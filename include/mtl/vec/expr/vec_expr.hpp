#pragma once
// MTL5 -- Vector expression umbrella
// In MTL4 this was a CRTP base class for all vector expressions.
// In MTL5, C++20 concepts (Vector, DenseVector) replace the CRTP hierarchy.
// Expression types simply satisfy the Vector concept by providing
// value_type, size_type, size(), operator()(i).
//
// This header includes all vector expression types for convenience.
#include <mtl/vec/expr/vec_vec_op_expr.hpp>
#include <mtl/vec/expr/vec_scal_op_expr.hpp>
#include <mtl/vec/expr/vec_negate_expr.hpp>
