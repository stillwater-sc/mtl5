#pragma once
// MTL5 — Basic iteration controller for Krylov solvers
#include <cmath>
#include <string>
#include <stdexcept>
#include <mtl/concepts/vector.hpp>
#include <mtl/operation/norms.hpp>

namespace mtl::itl {

/// Convergence controller: tracks iteration count, residual, and tolerance.
/// Convergence test: resid <= rtol * norm_r0  ||  resid <= atol
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

    /// Test convergence given residual vector r
    template <Vector V>
    bool finished(const V& r) {
        return finished(mtl::two_norm(r));
    }

    /// Test convergence given scalar residual norm
    bool finished(Real r) {
        resid_ = r;
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
        return false;
    }

    /// Pre-increment iteration counter
    basic_iteration& operator++() { ++i_; return *this; }

    /// Post-increment iteration counter
    basic_iteration operator++(int) { auto tmp = *this; ++i_; return tmp; }

    /// Returns true if this is the first iteration (i <= 1)
    bool first() const { return i_ <= 1; }

    /// Error code: 0 = converged, 1 = max_iter exceeded
    operator int() const { return error_code(); }

    int error_code() const { return error_; }

    // ── Getters ──────────────────────────────────────────────────────────

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
