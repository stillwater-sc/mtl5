#pragma once
// MTL5 stub — port from MTL4: boost/numeric/mtl/matrix/dense2D.hpp
// Dense row/column-major matrix with CRTP base
// Key changes from MTL4:
//   - Replace CRTP base classes with C++20 concepts for constraints
//   - Replace boost::enable_if with requires clauses
//   - Use contiguous_memory_block with if constexpr for stack/heap
namespace mtl::mat {
} // namespace mtl::mat
