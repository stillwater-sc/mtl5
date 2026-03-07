#pragma once
// MTL5 -- Noisy iteration controller: prints residual at every iteration
#include <iostream>
#include <mtl/itl/iteration/cyclic_iteration.hpp>

namespace mtl::itl {

/// Prints residual at every iteration (cyclic_iteration with cycle = 1).
template <typename Real>
class noisy_iteration : public cyclic_iteration<Real> {
    using base = cyclic_iteration<Real>;

public:
    template <Vector V>
    noisy_iteration(const V& r0, int max_iter, Real rtol,
                    Real atol = Real(0), std::ostream& out = std::cout)
        : base(r0, max_iter, rtol, 1, atol, out) {}

    noisy_iteration(Real norm_r0, int max_iter, Real rtol,
                    Real atol = Real(0), std::ostream& out = std::cout)
        : base(norm_r0, max_iter, rtol, 1, atol, out) {}
};

} // namespace mtl::itl
