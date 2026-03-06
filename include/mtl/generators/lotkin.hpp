#pragma once
// MTL5 — Lotkin matrix generator (implicit, no storage)
// Hilbert matrix with first row replaced by all 1s. Ill-conditioned, asymmetric.
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::generators {

/// Implicit Lotkin matrix: Hilbert with first row all 1s.
/// Ill-conditioned and asymmetric.
template <typename Value = double>
class lotkin {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using const_reference = Value;
    using reference       = Value;

    explicit lotkin(size_type n) : n_(n) {}

    value_type operator()(size_type r, size_type c) const {
        if (r == 0) return Value(1);
        return Value(1) / Value(r + c + 1);
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
struct category<generators::lotkin<V>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename V>
struct ashape<::mtl::generators::lotkin<V>> { using type = mat<V>; };
} // namespace mtl::ashape
