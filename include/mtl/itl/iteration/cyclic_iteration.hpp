#pragma once
// MTL5 -- Cyclic iteration controller with periodic residual printing
#include <iostream>
#include <stdexcept>
#include <mtl/detail/exceptions.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>

namespace mtl::itl {

/// Prints residual every `cycle` iterations (cycle >= 1); prints a summary
/// when the solver returns it as its int result, once.
template <typename Real>
class cyclic_iteration : public basic_iteration<Real> {
    using base = basic_iteration<Real>;

public:
    template <Vector V>
    cyclic_iteration(const V& r0, int max_iter, Real rtol, int cycle = 100,
                     Real atol = Real(0), std::ostream& out = std::cout)
        : base(r0, max_iter, rtol, atol), cycle_{checked_cycle(cycle)}, out_{out} {}

    cyclic_iteration(Real norm_r0, int max_iter, Real rtol, int cycle = 100,
                     Real atol = Real(0), std::ostream& out = std::cout)
        : base(norm_r0, max_iter, rtol, atol), cycle_{checked_cycle(cycle)}, out_{out} {}

    /// Test convergence; print residual every cycle iterations
    template <Vector V>
    bool finished(const V& r) {
        return finished(mtl::two_norm(r));
    }

    bool finished(Real r) {
        bool done = base::finished(r);
        if (this->i_ % cycle_ == 0 || done) {
            out_ << "iteration " << this->i_
                 << "\tresid " << this->resid_ << "\n";
        }
        return done;
    }

    /// The error code, printing the summary the first time it is asked for.
    /// basic_iteration's error_code() and operator int() are not virtual, so
    /// this class declares its own operator int: `return iter;` from a solver
    /// templated on cyclic_iteration reaches it and prints (it went through the
    /// base and printed nothing, issue 866). gmres asks inside its restarts,
    /// which no longer repeats the summary.
    int error_code() const {
        if (!reported_) {
            reported_ = true;
            if (this->error_ == 0)
                out_ << "converged in " << this->i_ << " iterations"
                     << "\tresid " << this->resid_ << "\n";
            else
                out_ << "NOT converged after " << this->i_ << " iterations"
                     << "\tresid " << this->resid_ << "\n";
        }
        return this->error_;
    }

    operator int() const { return error_code(); }

    /// basic_iteration::reset, and the summary is printed again for the next
    /// solve (it printed only for the first, issue 888). noisy_iteration
    /// inherits it.
    template <Vector V>
    void reset(const V& r0) {
        base::reset(r0);
        reported_ = false;
    }
    void reset(Real norm_r0) {
        base::reset(norm_r0);
        reported_ = false;
    }

protected:
    static int checked_cycle(int cycle) {
        // i_ % 0 is undefined (issue 866).
        if (cycle < 1)
            ::mtl::detail::throw_or_abort(std::invalid_argument("cyclic_iteration: cycle must be at least 1"));
        return cycle;
    }

    int           cycle_;
    std::ostream& out_;
    mutable bool  reported_ = false;
};

} // namespace mtl::itl
