#pragma once
// MTL5 -- Implicit permutation matrix (stores only the permutation vector)
// P(r,c) = 1 if perm[r] == c, else 0. No matrix storage.
// Efficient P*x and P*A via direct index remapping.
#include <cstddef>
#include <vector>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <mtl/math/identity.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::mat {

/// Implicit permutation matrix -- stores only perm vector, O(n) storage.
/// Row i maps to column perm[i]:  P(i, j) = (perm[i] == j) ? 1 : 0
template <typename Value = double>
class permutation_matrix {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using const_reference = Value;
    using reference       = Value;

    /// Construct from permutation vector: perm[i] = column of the 1 in row i.
    explicit permutation_matrix(std::vector<size_type> perm)
        : perm_(std::move(perm)) {}

    /// Construct identity permutation of size n.
    explicit permutation_matrix(size_type n)
        : perm_(n) {
        std::iota(perm_.begin(), perm_.end(), size_type(0));
    }

    value_type operator()(size_type r, size_type c) const {
        return (perm_[r] == c) ? math::one<Value>() : math::zero<Value>();
    }

    size_type num_rows() const { return perm_.size(); }
    size_type num_cols() const { return perm_.size(); }
    size_type size()     const { return perm_.size() * perm_.size(); }

    /// Direct access to the permutation vector.
    const std::vector<size_type>& permutation() const { return perm_; }

    /// Swap rows i and j (elementary row transposition).
    void swap_rows(size_type i, size_type j) {
        std::swap(perm_[i], perm_[j]);
    }

    /// Return the inverse permutation matrix (transpose).
    permutation_matrix inverse() const {
        std::vector<size_type> inv(perm_.size());
        for (size_type i = 0; i < perm_.size(); ++i)
            inv[perm_[i]] = i;
        return permutation_matrix(std::move(inv));
    }

private:
    std::vector<size_type> perm_;
};

// -- Efficient permutation-vector multiply ------------------------------
// P * x: y[i] = x[perm[i]]  (O(n), not the generic O(n^2) matvec)

template <typename Value, typename VV, typename VP>
auto operator*(const permutation_matrix<Value>& P,
               const vec::dense_vector<VV, VP>& x) {
    const auto& perm = P.permutation();
    assert(perm.size() == x.size());
    using result_t = std::common_type_t<Value, VV>;
    vec::dense_vector<result_t> y(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i)
        y(i) = static_cast<result_t>(x(perm[i]));
    return y;
}

} // namespace mtl::mat

namespace mtl::traits {
template <typename Value>
struct category<mat::permutation_matrix<Value>> { using type = tag::sparse; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Value>
struct ashape<::mtl::mat::permutation_matrix<Value>> { using type = mat<Value>; };
} // namespace mtl::ashape

namespace mtl { using mat::permutation_matrix; }
