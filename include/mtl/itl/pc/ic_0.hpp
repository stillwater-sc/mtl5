#pragma once
// MTL5 — IC(0) incomplete Cholesky factorization preconditioner for compressed2D
// Preserves the sparsity pattern of A (no fill-in). A must be SPD.
// Constructor performs factorization; solve does L/L^T trisolves.
#include <cassert>
#include <cmath>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl::pc {

/// IC(0) preconditioner for SPD compressed2D sparse matrices.
/// Computes L such that L*L^T approximates A with the same sparsity pattern.
template <typename Value, typename Parameters = mat::parameters<>>
class ic_0 {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    using size_type   = typename matrix_type::size_type;
public:
    explicit ic_0(const matrix_type& A) : n_(A.num_rows()) {
        assert(A.num_rows() == A.num_cols());
        factorize(A);
    }

    /// Solve (L*L^T)*x = b: forward (L*y=b), then backward (L^T*x=y)
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        // Forward substitution: L*y = b
        for (size_type i = 0; i < n_; ++i) {
            auto sum = math::zero<value_type>();
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k) {
                if (l_indices_[k] < i)
                    sum += l_data_[k] * x(l_indices_[k]);
            }
            x(i) = (b(i) - sum) / l_diag_[i];
        }
        // Back substitution: L^T*x = y
        // L^T is upper triangular. We use the stored L in column-access pattern.
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            auto sum = math::zero<value_type>();
            // Scan all rows j > i for entries L(j, i)
            for (size_type j = i + 1; j < n_; ++j) {
                for (size_type k = l_starts_[j]; k < l_starts_[j + 1]; ++k) {
                    if (l_indices_[k] == i) {
                        sum += l_data_[k] * x(j);
                        break;
                    }
                }
            }
            x(i) = (x(i) - sum) / l_diag_[i];
        }
    }

    /// adjoint_solve: same as solve (L*L^T is self-adjoint for real)
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);
    }

private:
    void factorize(const matrix_type& A) {
        using std::sqrt;
        const auto& a_starts  = A.ref_major();
        const auto& a_indices = A.ref_minor();
        const auto& a_data    = A.ref_data();

        // Copy lower triangle of A into working structure
        std::vector<std::vector<std::pair<size_type, value_type>>> l_rows(n_);
        l_diag_.resize(n_);

        for (size_type i = 0; i < n_; ++i) {
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                size_type j = a_indices[k];
                if (j <= i) {
                    l_rows[i].emplace_back(j, a_data[k]);
                }
            }
        }

        // IC(0) factorization (row-by-row Cholesky with no fill-in)
        for (size_type i = 0; i < n_; ++i) {
            // For each nonzero L(i,j) with j < i:
            //   L(i,j) = (A(i,j) - sum_{k<j} L(i,k)*L(j,k)) / L(j,j)
            for (auto& [j, val] : l_rows[i]) {
                if (j == i) continue; // handle diagonal separately
                auto sum = math::zero<value_type>();
                // Sum L(i,k)*L(j,k) for k < j
                for (const auto& [ik, iv] : l_rows[i]) {
                    if (ik >= j) break;
                    for (const auto& [jk, jv] : l_rows[j]) {
                        if (jk == ik) {
                            sum += iv * jv;
                            break;
                        }
                        if (jk > ik) break;
                    }
                }
                val = (val - sum) / l_diag_[j];
            }

            // Diagonal: L(i,i) = sqrt(A(i,i) - sum_{k<i} L(i,k)^2)
            value_type diag_val = math::zero<value_type>();
            auto sum_sq = math::zero<value_type>();
            for (const auto& [j, v] : l_rows[i]) {
                if (j == i) {
                    diag_val = v;
                } else {
                    sum_sq += v * v;
                }
            }
            l_diag_[i] = sqrt(diag_val - sum_sq);

            // Update diagonal entry in l_rows
            for (auto& [j, v] : l_rows[i]) {
                if (j == i) {
                    v = l_diag_[i];
                    break;
                }
            }
        }

        // Convert to CRS
        l_starts_.resize(n_ + 1);
        l_starts_[0] = 0;
        size_type nnz = 0;
        for (size_type i = 0; i < n_; ++i) {
            nnz += l_rows[i].size();
            l_starts_[i + 1] = nnz;
        }
        l_indices_.resize(nnz);
        l_data_.resize(nnz);
        size_type pos = 0;
        for (size_type i = 0; i < n_; ++i) {
            for (const auto& [j, v] : l_rows[i]) {
                l_indices_[pos] = j;
                l_data_[pos] = v;
                ++pos;
            }
        }
    }

    size_type n_;
    std::vector<size_type> l_starts_, l_indices_;
    std::vector<value_type> l_data_;
    std::vector<value_type> l_diag_;
};

} // namespace mtl::itl::pc
