#pragma once
// MTL5 -- ILU(0) incomplete LU factorization preconditioner for compressed2D
// Preserves the sparsity pattern of A (no fill-in).
// Constructor performs factorization; solve/adjoint_solve do L/U trisolves.
//
// ACCUMULATOR (#585). A non-void trailing `Accumulator` routes every
// sum-of-products of the factorization and of both solves through
// math::accumulator_traits, so each stored entry and each solution component is
// one reduction rounded once:
//
//   factor:  U(i,j) = A(i,j) - sum_k L(i,k) U(k,j),
//            L(i,j) = (A(i,j) - sum_k L(i,k) U(k,j)) / U(j,j)
//   solve:   y(i)   = b(i) - sum_j L(i,j) y(j)
//            x(i)   = (y(i) - sum_j U(i,j) x(j)) / U(i,i)
//
// A(i,j), b(i) and y(i) enter the accumulator through assign(), not by being
// subtracted from a rounded sum -- rounding the sum first gives the accuracy
// back (measured on the dense solves, #594).
//
// The IKJ factorization is right-looking WITHIN a row: entry (i,j) receives
// one update per eliminated k. That works here without restructuring because
// the whole row is live while it is eliminated, so the accumulator path keeps
// one accumulator per stored entry of the CURRENT row (workspace: the longest
// row of A) and rounds each out when the row is split into L and U. The
// products and their order are the void path's, so a plain `Acc == Value`
// accumulator reproduces it bit for bit.
//
// adjoint_solve reads the row storage by column and is therefore a scatter;
// its accumulator path keeps one live accumulator per component (workspace n,
// the shape of the transposed SpMV in operation/mult.hpp; see the workspace
// note in math/accumulator_traits.hpp).
#include <mtl/concepts/scalar.hpp>   // Field (#505)
#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace mtl::itl::pc {

/// ILU(0) preconditioner for compressed2D sparse matrices.
/// Computes L and U factors with the same sparsity pattern as A.
///
/// `Accumulator` selects how the factorization and the triangular solves
/// accumulate (see math/accumulator_traits.hpp and the header note). The default
/// `void` keeps the naive value_type arithmetic unchanged.
template <Field Value, typename Parameters = mat::parameters<>, typename Accumulator = void>
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
        if constexpr (!std::is_void_v<Accumulator>) {
            solve_acc(x, b);
            return;
        }
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
        if constexpr (!std::is_void_v<Accumulator>) {
            adjoint_solve_acc(x, b);
            return;
        }
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
    using AT = math::accumulator_traits<std::conditional_t<std::is_void_v<Accumulator>,
                                                           value_type, Accumulator>,
                                        value_type>;
    using acc_type = std::conditional_t<std::is_void_v<Accumulator>, value_type, Accumulator>;

    /// solve() with the accumulator: b(i) and y(i) are assigned into the
    /// accumulator and the row sum is subtracted inside it, rounded once.
    template <typename VecX, typename VecB>
    void solve_acc(VecX& x, const VecB& b) const {
        acc_type acc;
        for (size_type i = 0; i < n_; ++i) {
            AT::assign(acc, static_cast<value_type>(b(i)));
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                AT::add_product(acc, -l_data_[k], static_cast<value_type>(x(l_indices_[k])));
            x(i) = AT::template value<value_type>(acc);
        }
        for (size_type ii = 0; ii < n_; ++ii) {
            const size_type i = n_ - 1 - ii;
            AT::assign(acc, static_cast<value_type>(x(i)));
            for (size_type k = u_starts_[i]; k < u_starts_[i + 1]; ++k)
                AT::add_product(acc, -u_data_[k], static_cast<value_type>(x(u_indices_[k])));
            x(i) = AT::template value<value_type>(acc) / u_diag_[i];
        }
    }

    /// adjoint_solve() with the accumulator. The same scatter as the void path,
    /// but into one live accumulator per component, each rounded once when its
    /// component becomes final.
    template <typename VecX, typename VecB>
    void adjoint_solve_acc(VecX& x, const VecB& b) const {
        using conj_t = functor::scalar::conj<value_type>;
        std::vector<acc_type> acc(n_);

        for (size_type i = 0; i < n_; ++i)
            AT::assign(acc[i], static_cast<value_type>(b(i)));
        for (size_type i = 0; i < n_; ++i) {                    // U^H y = b
            const value_type yi = AT::template value<value_type>(acc[i]) / conj_t::apply(u_diag_[i]);
            x(i) = yi;
            for (size_type k = u_starts_[i]; k < u_starts_[i + 1]; ++k)
                AT::add_product(acc[u_indices_[k]], -conj_t::apply(u_data_[k]), yi);
        }

        for (size_type i = 0; i < n_; ++i)
            AT::assign(acc[i], static_cast<value_type>(x(i)));
        for (size_type ii = 0; ii < n_; ++ii) {                 // L^H x = y
            const size_type i = n_ - 1 - ii;
            const value_type xi = AT::template value<value_type>(acc[i]);
            x(i) = xi;
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                AT::add_product(acc[l_indices_[k]], -conj_t::apply(l_data_[k]), xi);
        }
    }

    /// The IKJ elimination with one accumulator per stored entry of row i. Same
    /// products in the same order as factorize(); each entry rounds once, when
    /// the row is split. ILU(0) has no fill, so the eliminated k are exactly the
    /// stored j < i, taken in increasing order as factorize() takes them.
    void factorize_acc(const matrix_type& A,
                       std::vector<std::vector<std::pair<size_type, value_type>>>& l_rows,
                       std::vector<std::vector<std::pair<size_type, value_type>>>& u_rows) {
        const auto& a_starts  = A.ref_major();
        const auto& a_indices = A.ref_minor();
        const auto& a_data    = A.ref_data();
        constexpr size_type none = std::numeric_limits<size_type>::max();

        std::vector<size_type> slot(n_, none);   // column -> accumulator of row i
        std::vector<acc_type> acc;
        std::vector<value_type> lval;            // L(i,k), rounded, by slot
        std::vector<size_type> cols, lower;

        for (size_type i = 0; i < n_; ++i) {
            const size_type len = a_starts[i + 1] - a_starts[i];
            if (acc.size() < len) { acc.resize(len); lval.resize(len); }
            cols.clear();
            lower.clear();
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                const size_type j = a_indices[k];
                slot[j] = cols.size();
                AT::assign(acc[slot[j]], a_data[k]);
                cols.push_back(j);
                if (j < i) lower.push_back(j);
            }
            std::sort(lower.begin(), lower.end());

            for (auto k : lower) {
                const size_type sk = slot[k];
                const value_type l = AT::template value<value_type>(acc[sk]) / u_diag_[k];
                lval[sk] = l;
                for (const auto& [uj, uv] : u_rows[k])
                    if (slot[uj] != none)
                        AT::add_product(acc[slot[uj]], -l, uv);
            }

            for (auto j : cols) {
                const size_type sj = slot[j];
                if (j < i)       l_rows[i].emplace_back(j, lval[sj]);
                else if (j > i)  u_rows[i].emplace_back(j, AT::template value<value_type>(acc[sj]));
                else             u_diag_[i] = AT::template value<value_type>(acc[sj]);
            }
            for (auto j : cols) slot[j] = none;
        }
    }

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

        if constexpr (!std::is_void_v<Accumulator>) {
            factorize_acc(A, l_rows, u_rows);
            build_crs(l_rows, l_starts_, l_indices_, l_data_);
            build_crs(u_rows, u_starts_, u_indices_, u_data_);
            return;
        }

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
