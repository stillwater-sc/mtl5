#pragma once
// MTL5 -- Ones matrix generator (implicit, no storage)
// O(i,j) = 1 for all i,j. Rank-1 matrix.
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::generators {

/// Implicit ones matrix: all entries equal to 1. Rank-1.
template <typename Value = double>
class ones {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using const_reference = Value;
    using reference       = Value;

    explicit ones(size_type n) : rows_(n), cols_(n) {}
    ones(size_type m, size_type n) : rows_(m), cols_(n) {}

    value_type operator()(size_type /*r*/, size_type /*c*/) const {
        return math::one<Value>();
    }

    size_type num_rows() const { return rows_; }
    size_type num_cols() const { return cols_; }
    size_type size()     const { return rows_ * cols_; }

private:
    size_type rows_;
    size_type cols_;
};

} // namespace mtl::generators

namespace mtl::traits {
template <typename V>
struct category<generators::ones<V>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename V>
struct ashape<::mtl::generators::ones<V>> { using type = mat<V>; };
} // namespace mtl::ashape
