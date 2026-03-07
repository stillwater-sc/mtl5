#pragma once
// MTL5 -- Hilbert matrix generator (implicit, no storage)
// H(i,j) = 1/(i+j+1). Notoriously ill-conditioned SPD matrix.
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::generators {

/// Implicit Hilbert matrix: H(i,j) = 1/(i+j+1).
/// Symmetric positive definite, notoriously ill-conditioned.
template <typename Value = double>
class hilbert {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using const_reference = Value;
    using reference       = Value;

    explicit hilbert(size_type n) : n_(n) {}

    value_type operator()(size_type r, size_type c) const {
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
struct category<generators::hilbert<V>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename V>
struct ashape<::mtl::generators::hilbert<V>> { using type = mat<V>; };
} // namespace mtl::ashape
