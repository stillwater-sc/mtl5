#pragma once
// MTL5 -- ILU(0) incomplete LU factorization preconditioner for compressed2D
// Preserves the sparsity pattern of A (no fill-in).
// Constructor performs factorization; solve/adjoint_solve do L/U trisolves.
#include <cassert>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>

namespace mtl::itl::pc {

/// ILU(0) preconditioner for compressed2D sparse matrices.
/// Computes L and U factors with the same sparsity pattern as A.
template <typename Value, typename Parameters = mat::parameters<>>
class ilu_0 {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    using size_type   = typename matrix_type::size_type;
public:
    explicit ilu_0(const matrix_type& A) : n_(A.num_rows()) {
        assert(A.num_rows() == A.num_cols());
        factorize(A);
    }

    /// Solve (L*U)*x = b: forward substitution (L*y=b), then back substitution (U*x=y)
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        // Forward substitution: L*y = b (L has unit diagonal)
        for (size_type i = 0; i < n_; ++i) {
            auto sum = math::zero<value_type>();
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                sum += l_data_[k] * x(l_indices_[k]);
            x(i) = b(i) - sum;
        }
        // Back substitution: U*x = y, i.e.
        //   x(i) = (y(i) - sum_{j>i} U(i,j)*x(j)) / U(i,i)
        // u_data_ holds only the strictly upper entries, so the sum is exactly
        // the j > i terms and the diagonal divides once (#323).
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            auto sum = math::zero<value_type>();
            for (size_type k = u_starts_[i]; k < u_starts_[i + 1]; ++k)
                sum += u_data_[k] * x(u_indices_[k]);
            x(i) = (x(i) - sum) / u_diag_[i];
        }
    }

    /// Solve (L*U)^H * x = b, i.e. x = U^-H (L^-H b).
    ///
    /// This used to delegate to solve(), commented "approximate". It is not an
    /// approximation -- M = L*U is not self-adjoint unless A is symmetric, so
    /// delegating applies the wrong operator entirely. bicg and qmr are the only
    /// callers, and with a non-symmetric A both failed to converge at all rather
    /// than converging slowly (#394).
    ///
    /// L and U are stored row-wise, but the adjoint needs COLUMN access:
    /// (U^H)(j,i) = conj(U(i,j)). Both triangular solves are therefore written
    /// in scatter form -- once a component is final, push its contribution into
    /// the entries that depend on it -- which reads the row storage exactly as
    /// it is laid out and needs no transposed copy.
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        using conj_t = functor::scalar::conj<value_type>;

        for (size_type i = 0; i < n_; ++i)
            x(i) = b(i);

        // U^H y = b. U^H is LOWER triangular with diagonal conj(U(i,i)), so this
        // is a forward substitution, scattered: finalize y(i), then subtract
        // (U^H)(j,i) * y(i) = conj(U(i,j)) * y(i) from every later j.
        for (size_type i = 0; i < n_; ++i) {
            x(i) = x(i) / conj_t::apply(u_diag_[i]);
            for (size_type k = u_starts_[i]; k < u_starts_[i + 1]; ++k)
                x(u_indices_[k]) -= conj_t::apply(u_data_[k]) * x(i);
        }

        // L^H x = y. L^H is UPPER triangular with UNIT diagonal, so this is a
        // backward substitution; x(i) is already final when reached.
        for (size_type ii = 0; ii < n_; ++ii) {
            const size_type i = n_ - 1 - ii;
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                x(l_indices_[k]) -= conj_t::apply(l_data_[k]) * x(i);
        }
    }

private:
    void factorize(const matrix_type& A) {
        const auto& a_starts  = A.ref_major();
        const auto& a_indices = A.ref_minor();
        const auto& a_data    = A.ref_data();

        // Work with dense row copies for ILU(0) factorization
        // Store result in separate L and U CRS structures
        // L: strictly lower triangular (unit diagonal implicit)
        // U: strictly upper triangular; the diagonal is held separately in
        //    u_diag_, so no consumer has to filter it out

        // First pass: copy A into a dense working matrix (row by row)
        // Then perform IKJ variant of ILU(0)
        std::vector<std::vector<std::pair<size_type, value_type>>> l_rows(n_);
        std::vector<std::vector<std::pair<size_type, value_type>>> u_rows(n_);
        u_diag_.resize(n_);

        // Copy A into row-indexed structure
        std::vector<value_type> row_work(n_, math::zero<value_type>());
        std::vector<bool> row_nz(n_, false);
        std::vector<size_type> nz_cols;

        for (size_type i = 0; i < n_; ++i) {
            // Clear work arrays
            nz_cols.clear();
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                size_type j = a_indices[k];
                row_work[j] = a_data[k];
                row_nz[j] = true;
                nz_cols.push_back(j);
            }

            // IKJ elimination: for each k < i where A(i,k) is nonzero
            for (size_type k = 0; k < i; ++k) {
                if (!row_nz[k]) continue;
                row_work[k] /= u_diag_[k];

                // Update: row(j) -= row(k) * U(k,j) for j > k, only at existing positions
                // u_rows[k] is strictly upper by construction, so every uj > k.
                for (const auto& [uj, uv] : u_rows[k]) {
                    if (row_nz[uj]) {
                        row_work[uj] -= row_work[k] * uv;
                    }
                }
            }

            // Split into L (j < i), the diagonal, and U (j > i).
            //
            // The diagonal is kept ONLY in u_diag_, never also in u_rows. It
            // used to be stored in both, and the back substitution then summed
            // all of u_rows[i] as though it were the strictly upper part and
            // divided by u_diag_[i] as well -- subtracting U(i,i)*x(i) from
            // y(i) and dividing the difference by U(i,i), which is wrong for
            // every input (#323). Storing it once removes the possibility:
            // there is no diagonal entry left in u_rows for a consumer to
            // forget to skip.
            for (auto j : nz_cols) {
                if (j < i) {
                    l_rows[i].emplace_back(j, row_work[j]);
                } else if (j > i) {
                    u_rows[i].emplace_back(j, row_work[j]);
                } else {
                    u_diag_[i] = row_work[j];
                }
            }

            // Reset work arrays
            for (auto j : nz_cols) {
                row_work[j] = math::zero<value_type>();
                row_nz[j] = false;
            }
        }

        // Convert to CRS format
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
    std::vector<size_type> l_starts_, l_indices_;
    std::vector<value_type> l_data_;
    std::vector<size_type> u_starts_, u_indices_;
    std::vector<value_type> u_data_;
    std::vector<value_type> u_diag_;
};

} // namespace mtl::itl::pc
