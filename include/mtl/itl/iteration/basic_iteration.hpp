#pragma once
// MTL5 -- Basic iteration controller for Krylov solvers
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>
#include <mtl/concepts/vector.hpp>
#include <mtl/operation/norms.hpp>

namespace mtl::itl {

/// Convergence controller: tracks iteration count, residual, and tolerance.
/// Convergence test: resid <= rtol * norm_r0  ||  resid <= atol
///
/// Error codes: 0 converged, 1 max_iter reached, 2 to 5 a solver's breakdown
/// (see the solver), `non_finite` (6) when a residual handed to finished()
/// is NaN or infinite, and `not_positive_definite` (7) when a solver that
/// requires an SPD (HPD) preconditioner finds that M is not (minres, cg; #949).
template <typename Real>
class basic_iteration {
public:
    /// Construct from initial residual vector r0
    template <Vector V>
    basic_iteration(const V& r0, int max_iter, Real rtol, Real atol = Real(0))
        : i_{0}, max_iter_{max_iter}, rtol_{rtol}, atol_{atol},
          my_norm_r0_{mtl::two_norm(r0)}, resid_{my_norm_r0_},
          error_{0}, is_finished_{false} {}

    /// Construct from precomputed norm of r0
    basic_iteration(Real norm_r0, int max_iter, Real rtol, Real atol = Real(0))
        : i_{0}, max_iter_{max_iter}, rtol_{rtol}, atol_{atol},
          my_norm_r0_{norm_r0}, resid_{norm_r0},
          error_{0}, is_finished_{false} {}

    /// Prepare the object for another solve (issue 888): the iteration count
    /// back to 0, the tolerance's reference norm to ||r0||, the residual,
    /// error and finished state cleared. Without reset, an object reused for
    /// a second solve keeps counting toward max_iter (a budget over all the
    /// solves) and measures the relative tolerance against the first r0's
    /// norm (#866 made reuse work, and pins that behaviour); a solver can't
    /// reset it, because what r0 means (b, b - A x0, ...) is the caller's.
    template <Vector V>
    void reset(const V& r0) { reset(Real(mtl::two_norm(r0))); }

    /// reset from a precomputed norm of r0, as the second constructor takes.
    void reset(Real norm_r0) {
        i_ = 0;
        my_norm_r0_ = norm_r0;
        resid_ = norm_r0;
        error_ = 0;
        is_finished_ = false;
    }

    /// Test convergence given residual vector r
    template <Vector V>
    bool finished(const V& r) {
        return finished(mtl::two_norm(r));
    }

    /// Error code of a residual that is NaN or infinite.
    static constexpr int non_finite = 6;

    /// Error code of a preconditioner found not to be positive definite: a
    /// v^H M^{-1} v negative beyond rounding, in a solver that requires an SPD
    /// (HPD) M. The same code in every solver (#949).
    static constexpr int not_positive_definite = 7;

    /// Test convergence given scalar residual norm
    bool finished(Real r) {
        resid_ = r;
        // A NaN fails every comparison, so converged() never held and only
        // max_iter stopped a solver: cg with a NaN in b ran 100000 iterations
        // and returned the generic error 1 (issue 924). A residual norm is not
        // negative, so this is NaN or +Inf.
        if (!(r <= std::numeric_limits<Real>::max())) {
            error_ = non_finite;
            is_finished_ = true;
            return true;
        }
        if (converged(r)) {
            error_ = 0;
            is_finished_ = true;
            return true;
        }
        if (i_ >= max_iter_) {
            error_ = 1;
            is_finished_ = true;
            return true;
        }
        // Each test reports afresh: an object reused for a second solve was
        // still finished from the first, and gmres's restart loop, which asks
        // is_finished(), returned without a step (issue 866).
        is_finished_ = false;
        return false;
    }

    /// Convergence test only, without the iteration-limit test: for a check
    /// in the middle of an iteration (bicgstab's half step, bicgstab_ell's
    /// BiCG part), where reaching max_iter must not cut the iteration short
    /// (issue 866). Finishes, with error code 0, only when r converged.
    template <Vector V>
    bool converged_now(const V& r) {
        return converged_now(mtl::two_norm(r));
    }

    bool converged_now(Real r) {
        resid_ = r;
        if (!converged(r)) return false;
        error_ = 0;
        is_finished_ = true;
        return true;
    }

    /// Pre-increment iteration counter
    basic_iteration& operator++() { ++i_; return *this; }

    /// Post-increment iteration counter
    basic_iteration operator++(int) { auto tmp = *this; ++i_; return tmp; }

    /// Returns true if this is the first iteration (i <= 1). The solvers do not
    /// use it to start their recurrences: an iteration object reused for a
    /// second solve is past its first iteration (issue 866).
    bool first() const { return i_ <= 1; }

    /// Error code: 0 = converged, 1 = max_iter exceeded, 2-5 breakdown,
    /// 6 = non-finite residual
    operator int() const { return error_code(); }

    int error_code() const { return error_; }

    // -- Getters ----------------------------------------------------------

    int iterations() const { return i_; }
    Real resid() const { return resid_; }

    Real relresid() const {
        return my_norm_r0_ > Real(0) ? resid_ / my_norm_r0_ : resid_;
    }

    Real tol() const { return rtol_; }
    Real atol() const { return atol_; }
    Real norm_r0() const { return my_norm_r0_; }

    /// Signal solver breakdown or other failure
    void fail(int code, const std::string& /* msg */) {
        error_ = code;
        is_finished_ = true;
    }

    bool is_finished() const { return is_finished_; }

protected:
    bool converged(Real r) const {
        return r <= rtol_ * my_norm_r0_ || r <= atol_;
    }

    int  i_;
    int  max_iter_;
    Real rtol_;
    Real atol_;
    Real my_norm_r0_;
    Real resid_;
    int  error_;
    bool is_finished_;
};

} // namespace mtl::itl
