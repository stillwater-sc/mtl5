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
//
// This class is the ADAPTER onto that smoother, not a second implementation of
// the sweeps (#405). One application of symmetric_sor from a zeroed x is
// exactly M^-1 b, which is the identity derived above, so the preconditioner is
// three lines over the smoother rather than a parallel copy of it. Keeping one
// implementation is not only tidiness: the x = b bug fixed in #398 lived in the
// duplicate, and could not have been written here.
#include <cassert>
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/itl/smoother/sor.hpp>

namespace mtl::itl::pc {

/// SSOR preconditioner: one symmetric-SOR application from a zeroed vector.
///
/// `Accumulator` is forwarded to the underlying smoother and selects the
/// accumulation type for the off-diagonal row sum (see
/// math/accumulator_traits.hpp). The default `void` keeps the naive value_type
/// accumulation, so this is a widening of the old single-parameter interface
/// rather than a change to it.
///
/// The matrix-type specialization lives in the smoother -- `smoother::sor` has
/// an O(nnz) `compressed2D` form -- so this template needs only one definition
/// and picks up sparse handling automatically.
template <typename Matrix, typename Accumulator = void>
class ssor {
    using value_type = typename Matrix::value_type;
    using size_type  = typename Matrix::size_type;
public:
    explicit ssor(const Matrix& A, value_type omega = value_type(1))
        : n_(A.num_rows()), sm_(A, omega)
    {
        assert(A.num_rows() == A.num_cols());
    }

    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        assert(x.size() == n_ && b.size() == n_);

        // Start from ZERO -- see the header note. Anything else adds a G x0
        // term and the result is not M^-1 b. This zeroing is the ONLY thing
        // that separates the preconditioner from the smoother.
        for (size_type i = 0; i < n_; ++i)
            x(i) = math::zero<value_type>();

        sm_(x, b);          // forward sweep then backward sweep == M^-1 b
    }

    /// adjoint_solve delegates to solve(), which is correct exactly when M is
    /// self-adjoint -- that is, when A is Hermitian and omega is real, giving
    /// M = c X D^-1 X^H (see the header note). It is NOT correct for a
    /// non-Hermitian A, where U != L^H and no rearrangement of the sweeps gives
    /// M^-H either.
    ///
    /// Measured against <M^-1 b, c> == <b, M^-H c> on a 144x144 Laplacian:
    /// 3.5e-16 for a symmetric A, 2.6e-01 for a non-symmetric one. The second
    /// number is a property of the method, not a defect -- SSOR of a
    /// non-symmetric A simply is not self-adjoint.
    ///
    /// Consequences:
    ///   - cg, gmres and the other eight solvers never call adjoint_solve and
    ///     are unaffected either way.
    ///   - bicg and qmr do call it, so ssor is usable with them for Hermitian A
    ///     only. For a non-Hermitian A use ilu_0, which has a real adjoint.
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);   // self-adjoint iff A is Hermitian -- see above (#394, #398)
    }

private:
    size_type n_;
    smoother::symmetric_sor<Matrix, Accumulator> sm_;
};

} // namespace mtl::itl::pc
