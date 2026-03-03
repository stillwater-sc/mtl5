#pragma once
// MTL5 — Free function: number of rows
#include <mtl/concepts/matrix.hpp>

namespace mtl {

/// Return the number of rows
template <Matrix M>
auto num_rows(const M& m) { return m.num_rows(); }

} // namespace mtl
