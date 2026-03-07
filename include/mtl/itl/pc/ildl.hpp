#pragma once
// MTL5 — ILDL (Incomplete LDL^T) preconditioner for compressed2D
// For symmetric (possibly indefinite) matrices.
// Preserves sparsity pattern of lower triangle.
#include <cassert>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl::pc {

/// ILDL preconditioner: incomplete L*D*L^T factorization.
/// D is diagonal (may have negative entries), L is unit lower triangular.
/// A must be symmetric; only the sparsity pattern of the lower triangle is preserved.
template <typename Value, typename Parameters = mat::parameters<>>
class ildl {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    using size_type   = typename matrix_type::size_type;
public:
    explicit ildl(const matrix_type& A) : n_(A.num_rows()) {
        assert(A.num_rows() == A.num_cols());
        factorize(A);
    }

    /// Solve (L*D*L^T)*x = b
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        // Forward substitution: L*y = b (L has unit diagonal)
        for (size_type i = 0; i < n_; ++i) {
            auto sum = math::zero<value_type>();
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                sum += l_data_[k] * x(l_indices_[k]);
            x(i) = b(i) - sum;
        }

        // Diagonal solve: D*z = y
        for (size_type i = 0; i < n_; ++i)
            x(i) /= d_[i];

        // Back substitution: L^T*x = z
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            // Scan rows j > i for L(j,i) entries
            for (size_type j = i + 1; j < n_; ++j) {
                for (size_type k = l_starts_[j]; k < l_starts_[j + 1]; ++k) {
                    if (l_indices_[k] == i) {
                        x(i) -= l_data_[k] * x(j);
                        break;
                    }
                    if (l_indices_[k] > i) break;
                }
            }
        }
    }

    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);  // L*D*L^T is symmetric
    }

private:
    void factorize(const matrix_type& A) {
        const auto& a_starts  = A.ref_major();
        const auto& a_indices = A.ref_minor();
        const auto& a_data    = A.ref_data();

        // Collect lower triangle entries (including diagonal) per row
        std::vector<std::vector<std::pair<size_type, value_type>>> l_rows(n_);
        d_.resize(n_);

        for (size_type i = 0; i < n_; ++i) {
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                size_type j = a_indices[k];
                if (j <= i) {
                    l_rows[i].emplace_back(j, a_data[k]);
                }
            }
        }

        // ILDL factorization (row-by-row)
        // For each row i:
        //   For j < i at sparsity positions:
        //     l_ij = (a_ij - sum_{k<j} l_ik * d_k * l_jk) / d_j
        //   d_i = a_ii - sum_{k<i} l_ik^2 * d_k
        for (size_type i = 0; i < n_; ++i) {
            // Process off-diagonal entries L(i,j) for j < i
            for (auto& [j, val] : l_rows[i]) {
                if (j == i) continue;
                auto sum = math::zero<value_type>();
                // sum = sum_{k<j} L(i,k) * d_k * L(j,k)
                for (const auto& [ik, iv] : l_rows[i]) {
                    if (ik >= j) break;
                    for (const auto& [jk, jv] : l_rows[j]) {
                        if (jk == ik) {
                            sum += iv * d_[ik] * jv;
                            break;
                        }
                        if (jk > ik) break;
                    }
                }
                val = (val - sum) / d_[j];
            }

            // Compute diagonal: d_i = a_ii - sum_{k<i} l_ik^2 * d_k
            value_type diag_val = math::zero<value_type>();
            auto sum_sq = math::zero<value_type>();
            for (const auto& [j, v] : l_rows[i]) {
                if (j == i) {
                    diag_val = v;
                } else {
                    sum_sq += v * v * d_[j];
                }
            }
            d_[i] = diag_val - sum_sq;

            // Update L(i,i) entry to 1 (unit diagonal) — but we don't store diagonal in L
        }

        // Build CRS for strictly lower triangular L (exclude diagonal)
        l_starts_.resize(n_ + 1);
        l_starts_[0] = 0;
        size_type nnz = 0;
        for (size_type i = 0; i < n_; ++i) {
            for (const auto& [j, v] : l_rows[i]) {
                if (j < i) ++nnz;
            }
            l_starts_[i + 1] = nnz;
        }
        l_indices_.resize(nnz);
        l_data_.resize(nnz);
        size_type pos = 0;
        for (size_type i = 0; i < n_; ++i) {
            for (const auto& [j, v] : l_rows[i]) {
                if (j < i) {
                    l_indices_[pos] = j;
                    l_data_[pos] = v;
                    ++pos;
                }
            }
        }
    }

    size_type n_;
    std::vector<size_type> l_starts_, l_indices_;
    std::vector<value_type> l_data_;
    std::vector<value_type> d_;
};

} // namespace mtl::itl::pc
