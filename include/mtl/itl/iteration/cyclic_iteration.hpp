#pragma once
// MTL5 -- Cyclic iteration controller with periodic residual printing
#include <iostream>
#include <mtl/itl/iteration/basic_iteration.hpp>

namespace mtl::itl {

/// Prints residual every `cycle` iterations; prints summary at convergence.
template <typename Real>
class cyclic_iteration : public basic_iteration<Real> {
    using base = basic_iteration<Real>;

public:
    template <Vector V>
    cyclic_iteration(const V& r0, int max_iter, Real rtol, int cycle = 100,
                     Real atol = Real(0), std::ostream& out = std::cout)
        : base(r0, max_iter, rtol, atol), cycle_{cycle}, out_{out} {}

    cyclic_iteration(Real norm_r0, int max_iter, Real rtol, int cycle = 100,
                     Real atol = Real(0), std::ostream& out = std::cout)
        : base(norm_r0, max_iter, rtol, atol), cycle_{cycle}, out_{out} {}

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

    int error_code() const {
        if (this->error_ == 0)
            out_ << "converged in " << this->i_ << " iterations"
                 << "\tresid " << this->resid_ << "\n";
        else
            out_ << "NOT converged after " << this->i_ << " iterations"
                 << "\tresid " << this->resid_ << "\n";
        return this->error_;
    }

protected:
    int           cycle_;
    std::ostream& out_;
};

} // namespace mtl::itl
