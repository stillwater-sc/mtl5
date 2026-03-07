#pragma once
// MTL5 -- Dense matrix expression umbrella
// In MTL4 this was a CRTP base for dense matrix expressions with tag dispatch.
// In MTL5, the DenseMatrix concept (category_t<T> == tag::dense) replaces
// the tag dispatch. All expression types are tagged as dense via trait
// specializations in their respective headers.
#include <mtl/mat/expr/mat_expr.hpp>
