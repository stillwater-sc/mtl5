#pragma once
// MTL5 -- SSOR-style preconditioner: a forward SOR sweep followed by a backward
// SOR sweep, both starting from x = b.
//
// NOTE this is the two-sweep smoother, not the classical factored operator
// M = (D/w + L)(D/w)^-1(D/w + U) applied as triangular solves. It is an
// effective preconditioner but it is NOT self-adjoint, so adjoint_solve cannot
// simply delegate to solve -- see the note on adjoint_solve below (#394, #398).
#include <cassert>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/mat/compressed2D.hpp>

namespace mtl::itl::pc {

/// SSOR preconditioner: symmetric SOR (forward + backward sweep).
/// Generic version using A(i,j) element access.
template <typename Matrix>
class ssor {
    using value_type = typename Matrix::value_type;
    using size_type  = typename Matrix::size_type;
public:
    explicit ssor(const Matrix& A, value_type omega = value_type(1))
        : A_(A), omega_(omega), n_(A.num_rows())
    {
        assert(A.num_rows() == A.num_cols());
    }

    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        assert(x.size() == n_ && b.size() == n_);

        // Copy b into x as working vector
        for (size_type i = 0; i < n_; ++i)
            x(i) = b(i);

        // Forward SOR sweep: rows 0 to n-1
        for (size_type i = 0; i < n_; ++i) {
            auto sigma = math::zero<value_type>();
            for (size_type j = 0; j < n_; ++j) {
                if (j != i)
                    sigma += A_(i, j) * x(j);
            }
            x(i) = omega_ * (b(i) - sigma) / A_(i, i)
                 + (value_type(1) - omega_) * x(i);
        }

        // Backward SOR sweep: rows n-1 to 0
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            auto sigma = math::zero<value_type>();
            for (size_type j = 0; j < n_; ++j) {
                if (j != i)
                    sigma += A_(i, j) * x(j);
            }
            x(i) = omega_ * (b(i) - sigma) / A_(i, i)
                 + (value_type(1) - omega_) * x(i);
        }
    }

    /// adjoint_solve delegates to solve(), and that is NOT because the operator
    /// is self-adjoint. Measured against <M^-1 b, c> == <b, M^-H c> on a
    /// 144x144 Laplacian, the relative violation is 1.1e-01 for a SYMMETRIC A
    /// and 4.1e-01 for a non-symmetric one (#394). Reversing the sweep order
    /// does not give the adjoint either (8.2e-02).
    ///
    /// The reason is that solve() above is not the classical SSOR operator. It
    /// runs two SOR SWEEPS starting from x = b, whereas
    ///
    ///     M = (D/w + L) (D/w)^-1 (D/w + U)
    ///
    /// is applied as two TRIANGULAR SOLVES. The implemented operator is a
    /// two-sweep smoother, which is a perfectly good preconditioner -- cg
    /// converges in 17 iterations with it against 33 with pc::diagonal, and
    /// gmres in 13 -- but it has no simple adjoint because it is not that
    /// factorization.
    ///
    /// Consequences, all measured:
    ///   - cg, gmres and the other eight solvers do not call adjoint_solve and
    ///     are unaffected.
    ///   - bicg and qmr do call it. They converge with ssor on a SYMMETRIC A
    ///     (17 iterations), and do NOT converge on a non-symmetric one.
    ///
    /// So: ssor is usable with bicg/qmr only for symmetric A. Reimplementing it
    /// in the classical factored form would make it genuinely self-adjoint and
    /// give it a real adjoint, at the cost of changing the iteration counts of
    /// every solver that uses it; that is tracked separately (#398).
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);   // NOT self-adjoint -- see the note above (#394)
    }

private:
    const Matrix& A_;
    value_type omega_;
    size_type n_;
};

/// Specialization for compressed2D: O(nnz) per sweep.
template <typename Value, typename Parameters>
class ssor<mat::compressed2D<Value, Parameters>> {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    using size_type   = typename matrix_type::size_type;
public:
    explicit ssor(const matrix_type& A, value_type omega = value_type(1))
        : A_(A), omega_(omega), n_(A.num_rows()), dia_(A.num_rows())
    {
        assert(A.num_rows() == A.num_cols());
        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data    = A.ref_data();
        for (size_type i = 0; i < n_; ++i) {
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] == i) {
                    dia_(i) = data[k];
                    break;
                }
            }
        }
    }

    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        assert(x.size() == n_ && b.size() == n_);
        const auto& starts  = A_.ref_major();
        const auto& indices = A_.ref_minor();
        const auto& data    = A_.ref_data();

        // Copy b into x
        for (size_type i = 0; i < n_; ++i)
            x(i) = b(i);

        // Forward SOR sweep
        for (size_type i = 0; i < n_; ++i) {
            auto sigma = math::zero<value_type>();
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] != i)
                    sigma += data[k] * x(indices[k]);
            }
            x(i) = omega_ * (b(i) - sigma) / dia_(i)
                 + (value_type(1) - omega_) * x(i);
        }

        // Backward SOR sweep
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            auto sigma = math::zero<value_type>();
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] != i)
                    sigma += data[k] * x(indices[k]);
            }
            x(i) = omega_ * (b(i) - sigma) / dia_(i)
                 + (value_type(1) - omega_) * x(i);
        }
    }

    /// adjoint_solve delegates to solve(), and that is NOT because the operator
    /// is self-adjoint. Measured against <M^-1 b, c> == <b, M^-H c> on a
    /// 144x144 Laplacian, the relative violation is 1.1e-01 for a SYMMETRIC A
    /// and 4.1e-01 for a non-symmetric one (#394). Reversing the sweep order
    /// does not give the adjoint either (8.2e-02).
    ///
    /// The reason is that solve() above is not the classical SSOR operator. It
    /// runs two SOR SWEEPS starting from x = b, whereas
    ///
    ///     M = (D/w + L) (D/w)^-1 (D/w + U)
    ///
    /// is applied as two TRIANGULAR SOLVES. The implemented operator is a
    /// two-sweep smoother, which is a perfectly good preconditioner -- cg
    /// converges in 17 iterations with it against 33 with pc::diagonal, and
    /// gmres in 13 -- but it has no simple adjoint because it is not that
    /// factorization.
    ///
    /// Consequences, all measured:
    ///   - cg, gmres and the other eight solvers do not call adjoint_solve and
    ///     are unaffected.
    ///   - bicg and qmr do call it. They converge with ssor on a SYMMETRIC A
    ///     (17 iterations), and do NOT converge on a non-symmetric one.
    ///
    /// So: ssor is usable with bicg/qmr only for symmetric A. Reimplementing it
    /// in the classical factored form would make it genuinely self-adjoint and
    /// give it a real adjoint, at the cost of changing the iteration counts of
    /// every solver that uses it; that is tracked separately (#398).
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);   // NOT self-adjoint -- see the note above (#394)
    }

private:
    const matrix_type& A_;
    value_type omega_;
    size_type n_;
    vec::dense_vector<value_type> dia_;
};

} // namespace mtl::itl::pc
