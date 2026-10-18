#pragma once
// MTL5 -- IC(0) incomplete Cholesky factorization preconditioner for compressed2D
// Preserves the sparsity pattern of A (no fill-in). A must be SPD.
// Constructor performs factorization; solve does L/L^T trisolves.
//
// ACCUMULATOR (#585). A non-void trailing `Accumulator` routes every
// sum-of-products of the factorization and of both substitutions through
// math::accumulator_traits, each rounded once:
//
//   factor:  L(i,j) = (A(i,j) - sum_k L(i,k) L(j,k)) / L(j,j)
//            L(i,i) = sqrt(A(i,i) - sum_k L(i,k)^2)
//   solve:   y(i)   = (b(i) - sum_j L(i,j) y(j)) / L(i,i)
//            x(i)   = (y(i) - sum_j L(j,i) x(j)) / L(i,i)
//
// The factorization is already the up-looking (row-by-row, dot-product) form,
// so each entry IS one reduction and the accumulator path needs no
// restructuring: A(i,j) is assigned into the accumulator, the products are
// subtracted inside it, and the division or square root follows the single
// round-out. The void path rounds the sum first and then subtracts it, so the
// two agree bit for bit only where a reduction has at most one term.
//
// The back substitution needs L by column. The void path finds each L(j,i) by
// scanning every later row (O(n * nnz)); the accumulator path scatters instead,
// into one live accumulator per component (workspace n, the shape of the
// transposed SpMV in operation/mult.hpp; see the workspace note in
// math/accumulator_traits.hpp).
#include <mtl/concepts/scalar.hpp>   // Field (#505)
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/functor/scalar/conj.hpp>

namespace mtl::itl::pc {

/// IC(0) preconditioner for SPD compressed2D sparse matrices.
/// Computes L such that L*L^T approximates A with the same sparsity pattern.
/// For a complex Value that is L*L^T, not L*L^H: the factorization conjugates
/// nothing, so it models a complex SYMMETRIC A (A^T == A), not a Hermitian one.
///
/// `Accumulator` selects how the factorization and the triangular solves
/// accumulate (see math/accumulator_traits.hpp and the header note). The default
/// `void` keeps the naive value_type arithmetic unchanged.
template <Field Value, typename Parameters = mat::parameters<>, typename Accumulator = void>
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
        apply<false>(x, b);
    }

    /// Solve (L*L^T)^H * x = b.
    ///
    /// This used to delegate to solve(), commented "L*L^T is self-adjoint for
    /// real" -- true, and wrong for complex: M = L*L^T has M^T == M, so
    /// M^H = conj(M), and M^-H is the same two solves with L conjugated.
    /// Delegating applied M^-1 instead, violating <M^-1 b, c> == <b, M^-H c> by
    /// 3.6e-02 on a complex-symmetric tridiagonal (#394's defect, for complex
    /// only). For a real Value conj is the identity, and this is solve()
    /// operation for operation.
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        apply<true>(x, b);
    }

private:
    /// Returns v, or conj(v) when Conj.
    template <bool Conj>
    static value_type maybe_conj(const value_type& v) {
        if constexpr (Conj) return functor::scalar::conj<value_type>::apply(v);
        else                return v;
    }

    /// (L L^T)^-1 b, or with Conj (conj(L) L^H)^-1 b.
    template <bool Conj, typename VecX, typename VecB>
    void apply(VecX& x, const VecB& b) const {
        if constexpr (!std::is_void_v<Accumulator>) {
            solve_acc<Conj>(x, b);
            return;
        }
        constexpr auto c = [](const value_type& v) { return maybe_conj<Conj>(v); };
        // Forward substitution: L*y = b
        for (size_type i = 0; i < n_; ++i) {
            auto sum = math::zero<value_type>();
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k) {
                if (l_indices_[k] < i)
                    sum += c(l_data_[k]) * x(l_indices_[k]);
            }
            x(i) = (b(i) - sum) / c(l_diag_[i]);
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
                        sum += c(l_data_[k]) * x(j);
                        break;
                    }
                }
            }
            x(i) = (x(i) - sum) / c(l_diag_[i]);
        }
    }

    using acc_type = std::conditional_t<std::is_void_v<Accumulator>, value_type, Accumulator>;
    using AT = math::accumulator_traits<acc_type, value_type>;

    /// solve() with the accumulator: the forward rows gather, the backward
    /// rows scatter into one accumulator per component. With Conj, the adjoint.
    template <bool Conj, typename VecX, typename VecB>
    void solve_acc(VecX& x, const VecB& b) const {
        constexpr auto c = [](const value_type& v) { return maybe_conj<Conj>(v); };
        std::vector<acc_type> acc(n_);
        for (size_type i = 0; i < n_; ++i) {                    // L y = b
            AT::assign(acc[i], static_cast<value_type>(b(i)));
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                if (l_indices_[k] < i)
                    AT::add_product(acc[i], -c(l_data_[k]), static_cast<value_type>(x(l_indices_[k])));
            x(i) = AT::template value<value_type>(acc[i]) / c(l_diag_[i]);
        }
        for (size_type i = 0; i < n_; ++i)
            AT::assign(acc[i], static_cast<value_type>(x(i)));
        for (size_type ii = 0; ii < n_; ++ii) {                 // L^T x = y
            const size_type i = n_ - 1 - ii;
            const value_type xi = AT::template value<value_type>(acc[i]) / c(l_diag_[i]);
            x(i) = xi;
            for (size_type k = l_starts_[i]; k < l_starts_[i + 1]; ++k)
                if (l_indices_[k] < i)
                    AT::add_product(acc[l_indices_[k]], -c(l_data_[k]), xi);
        }
    }

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

        if constexpr (!std::is_void_v<Accumulator>) {
            factorize_acc(l_rows);
        } else {
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

    /// The factorization above with each entry one accumulator: A(i,j) is
    /// assigned, the products are subtracted inside it, and the division (or
    /// square root, on the diagonal) follows the single round-out. The same
    /// products in the same order as the void loop.
    void factorize_acc(std::vector<std::vector<std::pair<size_type, value_type>>>& l_rows) {
        using std::sqrt;
        acc_type acc;
        for (size_type i = 0; i < n_; ++i) {
            for (auto& [j, val] : l_rows[i]) {
                if (j == i) continue;
                AT::assign(acc, val);
                for (const auto& [ik, iv] : l_rows[i]) {
                    if (ik >= j) break;
                    for (const auto& [jk, jv] : l_rows[j]) {
                        if (jk == ik) {
                            AT::add_product(acc, -iv, jv);
                            break;
                        }
                        if (jk > ik) break;
                    }
                }
                val = AT::template value<value_type>(acc) / l_diag_[j];
            }

            AT::clear(acc);
            for (const auto& [j, v] : l_rows[i])
                if (j == i) AT::assign(acc, v);
            for (const auto& [j, v] : l_rows[i])
                if (j != i) AT::add_product(acc, -v, v);
            l_diag_[i] = sqrt(AT::template value<value_type>(acc));

            for (auto& [j, v] : l_rows[i])
                if (j == i) { v = l_diag_[i]; break; }
        }
    }

    size_type n_;
    std::vector<size_type> l_starts_, l_indices_;
    std::vector<value_type> l_data_;
    std::vector<value_type> l_diag_;
};

} // namespace mtl::itl::pc
