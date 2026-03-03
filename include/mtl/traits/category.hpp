#pragma once
// MTL5 — Category trait (replaces MTL4 boost/numeric/mtl/utility/category.hpp)
// category<T> maps a collection type to its tag (e.g., tag::dense, tag::sparse)
// Key changes from MTL4:
//   - Replace boost::is_base_of with std::is_base_of_v
//   - Simplify virtual tag inheritance to flat tag structs + concepts

namespace mtl::traits {

/// Primary: unknown category
struct unknown_tag {};

/// Specializable trait: maps T to its category tag
template <typename T>
struct category {
    using type = unknown_tag;
};

template <typename T>
using category_t = typename category<T>::type;

} // namespace mtl::traits
