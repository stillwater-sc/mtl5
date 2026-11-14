#pragma once
// MTL5 -- Implicit permutation matrix (stores only the permutation vector)
// P(r,c) = 1 if perm[r] == c, else 0. No matrix storage.
// Efficient P*x via direct index remapping, here. P*A for a dense2D A (row
// i of the result is row perm[i] of A) is in operation/reorder.hpp.
#include <cstddef>
#include <vector>
#include <cassert>
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <mtl/config.hpp>
#include <mtl/detail/exceptions.hpp>
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
    /// It must be a permutation of 0..n-1, checked once, in every build
    /// (#826): a duplicate or out-of-range entry made inverse() and P * x
    /// index out of bounds. Throws std::invalid_argument (throw_or_abort).
    explicit permutation_matrix(std::vector<size_type> perm)
        : perm_(std::move(perm)) {
        validate();
    }

    /// Construct from a braced list of entries: P({2, 0, 1}), and P({0}) is
    /// the 1x1 permutation. Without it, {0} picked the size constructor and
    /// gave a 0x0 matrix, and {2} the 2x2 identity (#826). For the identity
    /// of size n, use P(n) with parentheses.
    permutation_matrix(std::initializer_list<size_type> perm)
        : perm_(perm) {
        validate();
    }

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
        if constexpr (bounds_checking) {
            if (i >= perm_.size() || j >= perm_.size())
                ::mtl::detail::throw_or_abort(std::out_of_range("permutation_matrix: row out of range"));
        }
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

    void validate() const {
        std::vector<bool> seen(perm_.size(), false);
        for (size_type p : perm_) {
            if (p >= perm_.size() || seen[p])
                ::mtl::detail::throw_or_abort(
                    std::invalid_argument("permutation_matrix: not a permutation of 0..n-1"));
            seen[p] = true;
        }
    }
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
