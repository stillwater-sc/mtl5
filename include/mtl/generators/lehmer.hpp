#pragma once
// MTL5 -- Lehmer matrix generator (implicit, no storage)
// L(i,j) = (min(i,j)+1)/(max(i,j)+1). SPD with positive eigenvalues.
#include <algorithm>
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::generators {

/// Implicit Lehmer matrix: L(i,j) = (min(i,j)+1)/(max(i,j)+1).
/// Symmetric positive definite with all positive eigenvalues.
template <typename Value = double>
class lehmer {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using const_reference = Value;
    using reference       = Value;

    explicit lehmer(size_type n) : n_(n) {}

    value_type operator()(size_type r, size_type c) const {
        return Value(std::min(r, c) + 1) / Value(std::max(r, c) + 1);
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
struct category<generators::lehmer<V>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename V>
struct ashape<::mtl::generators::lehmer<V>> { using type = mat<V>; };
} // namespace mtl::ashape
