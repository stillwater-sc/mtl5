#pragma once
// MTL5 stub — port from MTL4: boost/numeric/mtl/detail/contiguous_memory_block.hpp
// Unified stack/heap memory block using if constexpr
// Key changes from MTL4:
//   - Replace boost::mpl::if_ with if constexpr and std::array / std::unique_ptr
//   - Replace BOOST_STATIC_ASSERT with static_assert
//   - Use std::span for non-owning views
namespace mtl::detail {
} // namespace mtl::detail
