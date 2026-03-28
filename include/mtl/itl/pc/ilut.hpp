#pragma once
// MTL5 -- ILUT (Incomplete LU with Threshold) preconditioner for compressed2D
// Saad's ILUT: allows fill-in up to p entries per row, drops entries below tau*||row||.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl::pc {

/// ILUT preconditioner for compressed2D sparse matrices.
/// fill: max extra entries per row in L and U.
/// threshold: drop tolerance relative to row 2-norm.
template <typename Value, typename Parameters = mat::parameters<>>
class ilut {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    using size_type   = typename matrix_type::size_type;
public:
    explicit ilut(const matrix_type& A, size_type fill = 10,
                  value_type threshold = value_type(1e-4))
        : n_(A.num_rows()), fill_(fill), threshold_(threshold)
    {
        assert(A.num_rows() == A.num_cols());
        factorize(A);
    }

    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        // Forward substitution: L*y = b (L has unit diagonal)
        for (size_type i = 0; i < n_; ++i) {
            auto sum = math::zero<value_type>();
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                sum += l_data_[k] * x(l_indices_[k]);
            x(i) = b(i) - sum;
        }
        // Back substitution: U*x = y
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            auto sum = math::zero<value_type>();
            for (size_type k = u_starts_[i]; k < u_starts_[i + 1]; ++k)
                sum += u_data_[k] * x(u_indices_[k]);
            x(i) = (x(i) - sum) / u_diag_[i];
        }
    }

    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);
    }

private:
    void factorize(const matrix_type& A) {
        using std::abs;
        using std::sqrt;
        const auto& a_starts  = A.ref_major();
        const auto& a_indices = A.ref_minor();
        const auto& a_data    = A.ref_data();

        std::vector<std::vector<std::pair<size_type, value_type>>> l_rows(n_);
        std::vector<std::vector<std::pair<size_type, value_type>>> u_rows(n_);
        u_diag_.resize(n_);

        // Dense work row for accumulation
        std::vector<value_type> w(n_, math::zero<value_type>());
        std::vector<bool> w_nz(n_, false);
        std::vector<size_type> nz_cols;

        for (size_type i = 0; i < n_; ++i) {
            nz_cols.clear();

            // Compute row 2-norm for drop tolerance
            value_type row_norm = math::zero<value_type>();
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                row_norm += a_data[k] * a_data[k];
            }
            row_norm = sqrt(row_norm);
            value_type drop_tol = threshold_ * row_norm;

            // Copy row i of A into work vector
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                size_type j = a_indices[k];
                w[j] = a_data[k];
                w_nz[j] = true;
                nz_cols.push_back(j);
            }

            // IKJ elimination: for each k < i where w[k] is nonzero.
            // Process columns in strictly ascending order (k = 0, 1, ..., i-1)
            // so that fill-in at lower columns is incorporated before higher
            // columns are finalized.  A nz_cols-indexed loop would process
            // fill-in entries appended by push_back out of order, producing
            // incorrect L/U factors.
            for (size_type k = 0; k < i; ++k) {
                if (!w_nz[k]) continue;

                w[k] /= u_diag_[k];

                // Drop small entries in L part
                if (abs(w[k]) < drop_tol) {
                    w[k] = math::zero<value_type>();
                    w_nz[k] = false;
                    continue;
                }

                // Update: w[j] -= w[k] * U(k,j) for j > k
                for (const auto& [uj, uv] : u_rows[k]) {
                    if (uj <= k) continue;
                    if (!w_nz[uj]) {
                        // Fill-in: new nonzero position
                        w_nz[uj] = true;
                        w[uj] = math::zero<value_type>();
                        nz_cols.push_back(uj);
                    }
                    w[uj] -= w[k] * uv;
                }
            }

            // Split into L (j < i) and U (j >= i) parts
            // Collect and sort by magnitude for truncation
            using entry = std::pair<size_type, value_type>;
            std::vector<entry> l_entries, u_entries;

            for (auto j : nz_cols) {
                if (!w_nz[j]) continue;
                if (abs(w[j]) < drop_tol && j != i) continue;  // never drop diagonal

                if (j < i) {
                    l_entries.emplace_back(j, w[j]);
                } else {
                    u_entries.emplace_back(j, w[j]);
                    if (j == i) u_diag_[i] = w[j];
                }
            }

            // Count original nnz in each part for fill limit
            size_type orig_l = 0, orig_u = 0;
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                if (a_indices[k] < i) ++orig_l;
                else ++orig_u;
            }

            // Truncate L: keep at most (orig_l + fill_) largest entries
            size_type max_l = orig_l + fill_;
            if (l_entries.size() > max_l) {
                std::partial_sort(l_entries.begin(), l_entries.begin() + static_cast<std::ptrdiff_t>(max_l),
                                  l_entries.end(),
                                  [](const entry& a, const entry& b) {
                                      using std::abs;
                                      return abs(a.second) > abs(b.second);
                                  });
                l_entries.resize(max_l);
                // Re-sort by column index for correct forward substitution
                std::sort(l_entries.begin(), l_entries.end());
            }

            // Truncate U: keep at most (orig_u + fill_) largest entries (diagonal always kept)
            size_type max_u = orig_u + fill_;
            if (u_entries.size() > max_u) {
                // Separate diagonal, sort rest by magnitude, truncate, add diagonal back
                entry diag_entry{i, u_diag_[i]};
                u_entries.erase(
                    std::remove_if(u_entries.begin(), u_entries.end(),
                                   [i](const entry& e) { return e.first == i; }),
                    u_entries.end());
                if (u_entries.size() > max_u - 1) {
                    std::partial_sort(u_entries.begin(),
                                      u_entries.begin() + static_cast<std::ptrdiff_t>(max_u - 1),
                                      u_entries.end(),
                                      [](const entry& a, const entry& b) {
                                          using std::abs;
                                          return abs(a.second) > abs(b.second);
                                      });
                    u_entries.resize(max_u - 1);
                }
                u_entries.push_back(diag_entry);
                std::sort(u_entries.begin(), u_entries.end());
            }

            l_rows[i] = std::move(l_entries);
            u_rows[i] = std::move(u_entries);

            // Reset work arrays
            for (auto j : nz_cols) {
                w[j] = math::zero<value_type>();
                w_nz[j] = false;
            }
        }

        // Convert to CRS
        build_crs(l_rows, l_starts_, l_indices_, l_data_);
        build_crs(u_rows, u_starts_, u_indices_, u_data_);
    }

    void build_crs(const std::vector<std::vector<std::pair<size_type, value_type>>>& rows,
                   std::vector<size_type>& starts,
                   std::vector<size_type>& indices,
                   std::vector<value_type>& data) {
        starts.resize(n_ + 1);
        starts[0] = 0;
        size_type nnz = 0;
        for (size_type i = 0; i < n_; ++i) {
            nnz += rows[i].size();
            starts[i + 1] = nnz;
        }
        indices.resize(nnz);
        data.resize(nnz);
        size_type pos = 0;
        for (size_type i = 0; i < n_; ++i) {
            for (const auto& [j, v] : rows[i]) {
                indices[pos] = j;
                data[pos] = v;
                ++pos;
            }
        }
    }

    size_type n_;
    size_type fill_;
    value_type threshold_;
    std::vector<size_type> l_starts_, l_indices_;
    std::vector<value_type> l_data_;
    std::vector<size_type> u_starts_, u_indices_;
    std::vector<value_type> u_data_;
    std::vector<value_type> u_diag_;
};

} // namespace mtl::itl::pc
