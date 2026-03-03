#pragma once
// MTL5 — Free function: number of columns
#include <mtl/concepts/matrix.hpp>

namespace mtl {

/// Return the number of columns
template <Matrix M>
auto num_cols(const M& m) { return m.num_cols(); }

} // namespace mtl
