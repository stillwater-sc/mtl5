#pragma once
// MTL5 -- ILDL (Incomplete LDL^T) preconditioner for compressed2D
// For symmetric (possibly indefinite) matrices.
// Preserves sparsity pattern of lower triangle.
//
// ACCUMULATOR (#585). A non-void trailing `Accumulator` routes the
// factorization and both triangular solves through math::accumulator_traits.
// The factorization is the up-looking (row-by-row, dot-product) form, as in
// ic_0, so each entry is one reduction rounded once:
//
//   L(i,j) = (A(i,j) - sum_k (L(i,k) d_k) L(j,k)) / d_j
//   d_i    =  A(i,i) - sum_k (L(i,k) L(i,k)) d_k
//
// Each term is a product of THREE values and the accumulator takes two, so the
// first product -- L(i,k) d_k, or L(i,k)^2 -- rounds once in the element type,
// exactly as the void path forms it. When the accumulator can keep it
// (math::splits_product_terms_v: an IEEE element type and an exact or strictly
// wider built-in accumulator, #602) that product's rounding error is added as a
// second term, so the whole three-factor term reaches the accumulator; otherwise
// the accumulator covers the sum only. (Dense LDLT, #594, does the same with
// L(j,k) D(k).) A(i,j) is assigned into the
// accumulator rather than subtracted from a rounded sum. The forward solve
// forms b(i) - sum inside the accumulator; the L^T solve scatters into one
// accumulator per component (workspace n) where the void path scans later rows.
#include <mtl/concepts/scalar.hpp>   // Field (#505)
#include <cassert>
#include <type_traits>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace mtl::itl::pc {

/// ILDL preconditioner: incomplete L*D*L^T factorization.
/// D is diagonal (may have negative entries), L is unit lower triangular.
/// A must be symmetric; only the sparsity pattern of the lower triangle is preserved.
/// For a complex Value that means complex SYMMETRIC (A^T == A), not Hermitian:
/// the factorization reads only the lower triangle and conjugates nothing.
///
/// `Accumulator` selects how the factorization and the triangular solves
/// accumulate (see math/accumulator_traits.hpp and the header note). The default
/// `void` keeps the naive value_type arithmetic unchanged.
template <Field Value, typename Parameters = mat::parameters<>, typename Accumulator = void>
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
        apply<false>(x, b);
    }

    /// Solve (L*D*L^T)^H * x = b.
    ///
    /// This used to delegate to solve(), commented "L*D*L^T is symmetric". It
    /// is -- but symmetric is self-adjoint only for a REAL Value. For a complex
    /// one M^T == M gives M^H = conj(M), so M^-H is the same three solves with
    /// L and D conjugated. Delegating applied M^-1 instead: the adjoint
    /// identity <M^-1 b, c> == <b, M^-H c> was violated by 3.7e-02 on a
    /// complex-symmetric tridiagonal and 1.2e+00 on a dense one (#394's defect,
    /// for complex only). For a real Value conj is the identity, and this is
    /// solve() operation for operation.
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        apply<true>(x, b);
    }

private:
    /// (L D L^T)^-1 b, or with Conj its adjoint (conj(L) conj(D) L^H)^-1 b.
    template <bool Conj, typename VecX, typename VecB>
    void apply(VecX& x, const VecB& b) const {
        using conj_t = functor::scalar::conj<value_type>;
        auto c = [](const value_type& v) {
            if constexpr (Conj) return conj_t::apply(v);
            else                return v;
        };
        if constexpr (!std::is_void_v<Accumulator>) {
            apply_acc(x, b, c);
            return;
        }

        // Forward substitution: L*y = b (L has unit diagonal)
        for (size_type i = 0; i < n_; ++i) {
            auto sum = math::zero<value_type>();
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                sum += c(l_data_[k]) * x(l_indices_[k]);
            x(i) = b(i) - sum;
        }

        // Diagonal solve: D*z = y
        for (size_type i = 0; i < n_; ++i)
            x(i) /= c(d_[i]);

        // Back substitution: L^T*x = z
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            // Scan rows j > i for L(j,i) entries
            for (size_type j = i + 1; j < n_; ++j) {
                for (size_type k = l_starts_[j]; k < l_starts_[j + 1]; ++k) {
                    if (l_indices_[k] == i) {
                        x(i) -= c(l_data_[k]) * x(j);
                        break;
                    }
                    if (l_indices_[k] > i) break;
                }
            }
        }
    }

    using acc_type = std::conditional_t<std::is_void_v<Accumulator>, value_type, Accumulator>;
    using AT = math::accumulator_traits<acc_type, value_type>;

    /// apply() with the accumulator; `c` is the identity, or conj for the adjoint.
    template <typename VecX, typename VecB, typename C>
    void apply_acc(VecX& x, const VecB& b, C c) const {
        std::vector<acc_type> acc(n_);
        for (size_type i = 0; i < n_; ++i) {                    // L y = b
            AT::assign(acc[i], static_cast<value_type>(b(i)));
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                AT::add_product(acc[i], -c(l_data_[k]), static_cast<value_type>(x(l_indices_[k])));
            x(i) = AT::template value<value_type>(acc[i]);
        }
        for (size_type i = 0; i < n_; ++i)                      // D z = y
            AT::assign(acc[i], static_cast<value_type>(x(i) / c(d_[i])));
        for (size_type ii = 0; ii < n_; ++ii) {                 // L^T x = z
            const size_type i = n_ - 1 - ii;
            const value_type xi = AT::template value<value_type>(acc[i]);
            x(i) = xi;
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                AT::add_product(acc[l_indices_[k]], -c(l_data_[k]), xi);
        }
    }

    /// acc -= a * b * c. The product a * b is formed in value_type; when the
    /// accumulator can keep it, its rounding error goes in too (#602).
    static void add_term(acc_type& acc, const value_type& a, const value_type& b,
                         const value_type& c) {
        if constexpr (math::splits_product_terms_v<acc_type, value_type>) {
            const auto t = math::two_product(a, b);
            AT::add_product(acc, -t.hi, c);
            if (t.lo != value_type(0)) AT::add_product(acc, -t.lo, c);
        } else {
            AT::add_product(acc, -(a * b), c);
        }
    }

    /// The factorization below with each entry one accumulator (header note).
    void factorize_acc(std::vector<std::vector<std::pair<size_type, value_type>>>& l_rows) {
        acc_type acc;
        for (size_type i = 0; i < n_; ++i) {
            for (auto& [j, val] : l_rows[i]) {
                if (j == i) continue;
                AT::assign(acc, val);
                for (const auto& [ik, iv] : l_rows[i]) {
                    if (ik >= j) break;
                    for (const auto& [jk, jv] : l_rows[j]) {
                        if (jk == ik) {
                            add_term(acc, iv, d_[ik], jv);
                            break;
                        }
                        if (jk > ik) break;
                    }
                }
                val = AT::template value<value_type>(acc) / d_[j];
            }

            AT::clear(acc);
            for (const auto& [j, v] : l_rows[i])
                if (j == i) AT::assign(acc, v);
            for (const auto& [j, v] : l_rows[i])
                if (j != i) add_term(acc, v, v, d_[j]);
            d_[i] = AT::template value<value_type>(acc);
        }
    }

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

        if constexpr (!std::is_void_v<Accumulator>) {
            factorize_acc(l_rows);
        } else {
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

            // Update L(i,i) entry to 1 (unit diagonal) -- but we don't store diagonal in L
        }
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
