#pragma once
// MTL5 -- Implicit identity matrix (no storage)
// Returns 1 on diagonal, 0 elsewhere. Satisfies Matrix concept.
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::mat {

/// Implicit identity matrix: no storage, O(1) element access.
template <typename Value = double>
class identity2D {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using const_reference = Value;
    using reference       = Value;

    explicit identity2D(size_type n) : n_(n) {}

    value_type operator()(size_type r, size_type c) const {
        return (r == c) ? math::one<Value>() : math::zero<Value>();
    }

    size_type num_rows() const { return n_; }
    size_type num_cols() const { return n_; }
    size_type size()     const { return n_ * n_; }

private:
    size_type n_;
};

} // namespace mtl::mat

namespace mtl::traits {
template <typename Value>
struct category<mat::identity2D<Value>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Value>
struct ashape<::mtl::mat::identity2D<Value>> { using type = mat<Value>; };
} // namespace mtl::ashape

namespace mtl { using mat::identity2D; }
