#pragma once
// MTL5 -- SSOR preconditioner: a forward SOR sweep followed by a backward SOR
// sweep, both starting from x = 0.
//
// Starting from ZERO is what makes this the preconditioner rather than a
// smoother step, and it is not a detail. A stationary iteration and a
// preconditioner are the same splitting matrix M viewed two ways: SOR is
// defined by A = M - N, and applying M^-1 to a vector means running one sweep
// FROM ZERO. From any other start the result is M^-1 b + G x0, where
// G = I - M^-1 A is the iteration matrix -- fine for a solver, where the
// initial guess is arbitrary, wrong for a preconditioner, where the caller
// asked for M^-1 b.
//
// From x0 = 0 the two sweeps compute exactly
//
//     M = w/(2-w) * (D/w + L) D^-1 (D/w + U)
//
// -- the classical SSOR operator of Sheldon (1955), scalar factor included.
// Sweeps and triangular solves are the same operator here, not two competing
// formulations. Derivation, with F = D/w + L and B = D/w + U:
//
//     forward from 0:  F x1 = b
//     backward:        B x2 = b + (B - A) x1 = b + (D/w - D - L) x1
//     so               x2 = B^-1 ((B - A) + F) F^-1 b,  and
//                      (B - A) + F = 2D/w - D = D (2-w)/w
//     giving           x2 = [ w/(2-w) F D^-1 B ]^-1 b.
//
// If A is Hermitian then U = L^H and D is real, so M = c X D^-1 X^H with
// X = D/w + L: Hermitian by construction, and usable as a CG preconditioner.
// That is precisely why Sheldon introduced the symmetric variant.
//
// Refs: Young (1954) Trans. AMS 76, 92-111 and Frankel (1950) MTAC 4, 65-75
// (SOR); Sheldon (1955) MTAC 9, 101-112 (SSOR); Young (1971) Iterative Solution
// of Large Linear Systems; Barrett et al. (1994) Templates 3.3; Saad (2003)
// Iterative Methods for Sparse Linear Systems 4.1, 10.2.  (#398)
//
// NOTE solve() overwrites the caller's x, so this is a preconditioner and not a
// smoother. The smoother, which relaxes an existing iterate, is a separate
// class: mtl::itl::smoother::symmetric_sor in <mtl/itl/smoother/sor.hpp>.
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

        // Start the sweeps from ZERO -- see the header note. Anything else
        // adds a G x0 term and the result is not M^-1 b.
        for (size_type i = 0; i < n_; ++i)
            x(i) = math::zero<value_type>();

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

    /// adjoint_solve delegates to solve(), which is correct exactly when M is
    /// self-adjoint -- that is, when A is Hermitian and omega is real, giving
    /// M = c X D^-1 X^H (see the header note). It is NOT correct for a
    /// non-Hermitian A, where U != L^H and no rearrangement of the sweeps gives
    /// M^-H either.
    ///
    /// Measured against <M^-1 b, c> == <b, M^-H c> on a 144x144 Laplacian:
    /// 3.4e-16 for a symmetric A, 2.6e-01 for a non-symmetric one. The second
    /// number is a property of the method, not a defect -- SSOR of a
    /// non-symmetric A simply is not self-adjoint.
    ///
    /// Consequences:
    ///   - cg, gmres and the other eight solvers never call adjoint_solve and
    ///     are unaffected either way.
    ///   - bicg and qmr do call it, so ssor is usable with them for Hermitian A
    ///     only. For a non-Hermitian A use ilu_0, which has a real adjoint.
    ///
    /// Both numbers above were 1.1e-01 and 4.1e-01 before #398, when the sweeps
    /// started from x = b: the extra G_B G_F term is not symmetric even when A
    /// and M both are, because M^-1 A is not.
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);   // self-adjoint iff A is Hermitian -- see above (#394, #398)
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

        // Start the sweeps from ZERO -- see the header note. Anything else
        // adds a G x0 term and the result is not M^-1 b.
        for (size_type i = 0; i < n_; ++i)
            x(i) = math::zero<value_type>();

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

    /// adjoint_solve delegates to solve(), which is correct exactly when M is
    /// self-adjoint -- that is, when A is Hermitian and omega is real, giving
    /// M = c X D^-1 X^H (see the header note). It is NOT correct for a
    /// non-Hermitian A, where U != L^H and no rearrangement of the sweeps gives
    /// M^-H either.
    ///
    /// Measured against <M^-1 b, c> == <b, M^-H c> on a 144x144 Laplacian:
    /// 3.4e-16 for a symmetric A, 2.6e-01 for a non-symmetric one. The second
    /// number is a property of the method, not a defect -- SSOR of a
    /// non-symmetric A simply is not self-adjoint.
    ///
    /// Consequences:
    ///   - cg, gmres and the other eight solvers never call adjoint_solve and
    ///     are unaffected either way.
    ///   - bicg and qmr do call it, so ssor is usable with them for Hermitian A
    ///     only. For a non-Hermitian A use ilu_0, which has a real adjoint.
    ///
    /// Both numbers above were 1.1e-01 and 4.1e-01 before #398, when the sweeps
    /// started from x = b: the extra G_B G_F term is not symmetric even when A
    /// and M both are, because M^-1 A is not.
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);   // self-adjoint iff A is Hermitian -- see above (#394, #398)
    }

private:
    const matrix_type& A_;
    value_type omega_;
    size_type n_;
    vec::dense_vector<value_type> dia_;
};

} // namespace mtl::itl::pc
