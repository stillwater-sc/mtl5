#pragma once
// MTL5 stub — port from MTL4: boost/numeric/mtl/vector/dense_vector.hpp
// Dense vector with CRTP, contiguous storage, orientation-aware
// Key changes from MTL4:
//   - Replace CRTP bases with C++20 concepts
//   - Use contiguous_memory_block with if constexpr
//   - Replace boost::enable_if with requires clauses
namespace mtl::vec {
} // namespace mtl::vec
