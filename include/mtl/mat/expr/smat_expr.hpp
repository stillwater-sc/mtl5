#pragma once
// MTL5 — Sparse matrix expression umbrella
// In MTL4 this was a CRTP base for sparse matrix expressions with tag dispatch.
// In MTL5, the SparseMatrix concept (category_t<T> == tag::sparse) replaces
// the tag dispatch. Sparse expression types would specialize the category
// trait to tag::sparse.
#include <mtl/mat/expr/mat_expr.hpp>
