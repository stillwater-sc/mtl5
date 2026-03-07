#pragma once
// MTL5 -- Matrix expression umbrella
// In MTL4 this was a CRTP base class for all matrix expressions.
// In MTL5, C++20 concepts (Matrix, DenseMatrix, SparseMatrix) replace the
// CRTP hierarchy. Expression types simply satisfy the Matrix concept by
// providing value_type, size_type, num_rows(), num_cols(), size(), operator()(r,c).
//
// This header includes all matrix expression types for convenience.
#include <mtl/mat/expr/mat_mat_op_expr.hpp>
#include <mtl/mat/expr/mat_scal_op_expr.hpp>
#include <mtl/mat/expr/mat_negate_expr.hpp>
#include <mtl/mat/expr/mat_mat_times_expr.hpp>
