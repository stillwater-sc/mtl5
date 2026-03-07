#pragma once
// MTL5 -- Min(i,j) matrix generator (implicit, no storage)
// M(i,j) = min(i+1,j+1). SPD with known inverse.
#include <algorithm>
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::generators {

/// Implicit min(i,j) matrix: M(i,j) = min(i+1,j+1).
/// Symmetric positive definite with known inverse.
template <typename Value = double>
class minij {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using const_reference = Value;
    using reference       = Value;

    explicit minij(size_type n) : n_(n) {}

    value_type operator()(size_type r, size_type c) const {
        return Value(std::min(r, c) + 1);
    }

    size_type num_rows() const { return n_; }
    size_type num_cols() const { return n_; }
    size_type size()     const { return n_ * n_; }

private:
    size_type n_;
};

} // namespace mtl::generators

namespace mtl::traits {
template <typename V>
struct category<generators::minij<V>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename V>
struct ashape<::mtl::generators::minij<V>> { using type = mat<V>; };
} // namespace mtl::ashape
