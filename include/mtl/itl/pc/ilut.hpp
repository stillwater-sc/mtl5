#pragma once
// MTL5 -- ILUT (Incomplete LU with Threshold) preconditioner for compressed2D
// Saad's ILUT: allows fill-in up to p entries per row, drops entries below tau*||row||.
//
// ACCUMULATOR (#585). A non-void trailing `Accumulator` routes the elimination
// and both triangular solves through math::accumulator_traits. As in ilu_0 the
// IKJ elimination is right-looking within a row, but the whole work row is
// live while it is eliminated, so each live entry -- fill-in included -- gets
// its own accumulator: A(i,j) is assigned into it (fill starts cleared), each
// update is added as a product, and the entry rounds ONCE, when it is consumed
// (L(i,k) when row k eliminates it, U(i,j) when the row is split). The drop
// tests and the magnitude truncation read those single-rounded values, so the
// accumulator decides the pattern as well as the values. Workspace: n
// accumulators, since fill can reach any column (see the workspace note in
// math/accumulator_traits.hpp). A plain `Acc == Value` accumulator performs the
// void path's operations in the void path's order, bit for bit.
//
// solve() forms b(i) - sum and y(i) - sum inside the accumulator, rounded once;
// the void path rounds the sum first, so the two agree only for one-term rows.
// adjoint_solve() scatters into one live accumulator per component.
#include <mtl/concepts/magnitude.hpp>   // magnitude_t (#505 review)
#include <mtl/concepts/scalar.hpp>   // Field (#505)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>
#include <utility>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace mtl::itl::pc {

/// ILUT preconditioner for compressed2D sparse matrices.
/// fill: max extra entries per row in L and U.
/// threshold: drop tolerance relative to row 2-norm.
///
/// `Accumulator` selects how the elimination and the triangular solves
/// accumulate (see math/accumulator_traits.hpp and the header note). The default
/// `void` keeps the naive value_type arithmetic unchanged.
template <Field Value, typename Parameters = mat::parameters<>, typename Accumulator = void>
class ilut {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    /// The REAL type the drop tolerance and row norms live in.
    ///
    /// A threshold is a magnitude, and magnitudes are ordered; complex numbers
    /// are not. Storing them as `Value` made `abs(w[k]) < drop_tol` compare a
    /// double against a complex<double>, so ilut<complex<double>> named a valid
    /// type whose factorize() did not compile (#505 review). Identical to Value
    /// for every real element type, so nothing changes there.
    using magnitude_type = magnitude_t<Value>;
    using size_type   = typename matrix_type::size_type;
public:
    explicit ilut(const matrix_type& A, size_type fill = 10,
                  magnitude_type threshold = magnitude_type(1e-4))
        : n_(A.num_rows()), fill_(fill), threshold_(threshold)
    {
        assert(A.num_rows() == A.num_cols());
        factorize(A);
    }

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
        // the j > i terms and the diagonal divides once -- the #323 fix to
        // ilu_0, which this class had kept the bug of.
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
    /// This used to delegate to solve(), which applies M^-1 rather than M^-H
    /// and is wrong for every non-symmetric A -- the #394 defect, which ilu_0
    /// had fixed and this class still carried. The triangular solves are in
    /// scatter form exactly as in ilu_0::adjoint_solve: L and U are stored by
    /// row, the adjoint needs them by column, and scattering each finished
    /// component reads the row storage as it is laid out.
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        if constexpr (!std::is_void_v<Accumulator>) {
            adjoint_solve_acc(x, b);
            return;
        }
        using conj_t = functor::scalar::conj<value_type>;

        for (size_type i = 0; i < n_; ++i)
            x(i) = b(i);

        // U^H y = b: lower triangular with diagonal conj(U(i,i)).
        for (size_type i = 0; i < n_; ++i) {
            x(i) = x(i) / conj_t::apply(u_diag_[i]);
            for (size_type k = u_starts_[i]; k < u_starts_[i + 1]; ++k)
                x(u_indices_[k]) -= conj_t::apply(u_data_[k]) * x(i);
        }

        // L^H x = y: upper triangular with unit diagonal.
        for (size_type ii = 0; ii < n_; ++ii) {
            const size_type i = n_ - 1 - ii;
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                x(l_indices_[k]) -= conj_t::apply(l_data_[k]) * x(i);
        }
    }

private:
    static constexpr bool accumulated = !std::is_void_v<Accumulator>;
    using acc_type = std::conditional_t<accumulated, Accumulator, value_type>;
    using AT = math::accumulator_traits<acc_type, value_type>;

    /// solve() with the accumulator, as ilu_0::solve_acc.
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

    /// adjoint_solve() with the accumulator, as ilu_0::adjoint_solve_acc.
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
        // One accumulator per column of the work row; empty for void.
        std::vector<acc_type> wa(accumulated ? n_ : 0);

        for (size_type i = 0; i < n_; ++i) {
            nz_cols.clear();

            // Compute row 2-norm for drop tolerance
            // sum of |a|^2, not sum of a^2: they agree for real elements and
            // only the former is the 2-norm for complex ones.
            magnitude_type row_norm = math::zero<magnitude_type>();
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                const magnitude_type m = abs(a_data[k]);
                row_norm += m * m;
            }
            row_norm = sqrt(row_norm);
            const magnitude_type drop_tol = threshold_ * row_norm;

            // Copy row i of A into work vector
            for (size_type k = a_starts[i]; k < a_starts[i + 1]; ++k) {
                size_type j = a_indices[k];
                w[j] = a_data[k];
                if constexpr (accumulated) AT::assign(wa[j], a_data[k]);
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

                if constexpr (accumulated) w[k] = AT::template value<value_type>(wa[k]);
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
                        if constexpr (accumulated) AT::clear(wa[uj]);
                        nz_cols.push_back(uj);
                    }
                    if constexpr (accumulated) AT::add_product(wa[uj], -w[k], uv);
                    else                       w[uj] -= w[k] * uv;
                }
            }

            // Split into L (j < i), the diagonal, and U (j > i).
            //
            // The diagonal is kept ONLY in u_diag_, never also in u_rows, as in
            // ilu_0 (#323). It used to be stored in both, and the back
            // substitution then summed U(i,i)*x(i) into the j > i sum and
            // divided by U(i,i) as well -- wrong for every input, including the
            // no-drop ILUT that is exactly LU.
            // Collect and sort by magnitude for truncation
            using entry = std::pair<size_type, value_type>;
            std::vector<entry> l_entries, u_entries;

            // The L part (j < i) was rounded and divided as it was eliminated;
            // the diagonal and U part round out here, once, before the drop
            // test reads them.
            if constexpr (accumulated)
                for (auto j : nz_cols)
                    if (j >= i && w_nz[j]) w[j] = AT::template value<value_type>(wa[j]);

            for (auto j : nz_cols) {
                if (!w_nz[j]) continue;
                if (abs(w[j]) < drop_tol && j != i) continue;  // never drop diagonal

                if (j < i)       l_entries.emplace_back(j, w[j]);
                else if (j > i)  u_entries.emplace_back(j, w[j]);
                else             u_diag_[i] = w[j];
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
                // Re-sort by column index for correct forward substitution.
                // BY .first EXPLICITLY, not lexicographically on the pair: the
                // default compares .second when .first ties, which needs the
                // element type to be ordered. Column indices are unique within a
                // row so it never tied at runtime, but the comparison still had
                // to COMPILE -- and complex is not ordered, so ilut<complex> was
                // rejected here (#505 review). Comparing the key is also what
                // the comment above always said this does.
                std::sort(l_entries.begin(), l_entries.end(),
                          [](const entry& a, const entry& b) { return a.first < b.first; });
            }

            // Truncate U: keep at most (orig_u + fill_) entries counting the
            // diagonal, which is always kept -- so at most orig_u + fill_ - 1
            // of the largest strictly upper entries. (orig_u counts j >= i.)
            const size_type max_u = orig_u + fill_ - 1;
            if (u_entries.size() > max_u) {
                std::partial_sort(u_entries.begin(),
                                  u_entries.begin() + static_cast<std::ptrdiff_t>(max_u),
                                  u_entries.end(),
                                  [](const entry& a, const entry& b) {
                                      using std::abs;
                                      return abs(a.second) > abs(b.second);
                                  });
                u_entries.resize(max_u);
                std::sort(u_entries.begin(), u_entries.end(),
                          [](const entry& a, const entry& b) { return a.first < b.first; });
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
    magnitude_type threshold_;
    std::vector<size_type> l_starts_, l_indices_;
    std::vector<value_type> l_data_;
    std::vector<size_type> u_starts_, u_indices_;
    std::vector<value_type> u_data_;
    std::vector<value_type> u_diag_;
};

} // namespace mtl::itl::pc
