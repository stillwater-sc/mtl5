#pragma once
// MTL5 -- TFQMR (Transpose-Free Quasi-Minimal Residual) solver
// Does not require trans(A). Based on Freund (1993).
#include <cmath>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>

namespace mtl::itl {

/// TFQMR solver for non-symmetric systems A*x = b.
/// Does not require A^T (transpose-free). Left preconditioner M.
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int tfqmr(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    using std::sqrt;
    using std::abs;
    const size_type n = x.size();

    vec::dense_vector<value_type> r(n), r_star(n), w(n), y1(n), y2(n);
    vec::dense_vector<value_type> d(n), v(n), u1(n), u2(n), tmp(n);

    // r = b - A*x
    auto Ax = A * x;
    for (size_type i = 0; i < n; ++i) {
        r(i) = b(i) - Ax(i);
        r_star(i) = r(i);
        w(i) = r(i);
        d(i) = value_type(0);
    }

    // y1 = M^{-1} r
    M.solve(y1, r);

    auto Ay1 = A * y1;
    for (size_type i = 0; i < n; ++i) {
        u1(i) = Ay1(i);
        v(i) = u1(i);
    }

    value_type tau = mtl::two_norm(r);
    value_type theta = value_type(0);
    value_type eta = value_type(0);
    value_type rho = mtl::dot(r_star, r);

    while (!iter.finished(tau)) {
        ++iter;

        value_type sigma = mtl::dot(r_star, v);
        if (sigma == value_type(0)) {
            iter.fail(2, "tfqmr breakdown: sigma == 0");
            return iter;
        }
        value_type alpha = rho / sigma;

        // Odd half-step
        for (size_type i = 0; i < n; ++i)
            w(i) -= alpha * u1(i);

        // d = y1 + (theta^2 * eta / alpha) * d
        value_type factor = theta * theta * eta / alpha;
        for (size_type i = 0; i < n; ++i)
            d(i) = y1(i) + factor * d(i);

        theta = mtl::two_norm(w) / tau;
        value_type c = value_type(1) / sqrt(value_type(1) + theta * theta);
        tau *= theta * c;
        eta = c * c * alpha;

        // x += eta * d
        for (size_type i = 0; i < n; ++i)
            x(i) += eta * d(i);

        if (iter.finished(tau)) return iter;

        // y2 = M^{-1} w
        M.solve(y2, w);

        auto Ay2 = A * y2;
        for (size_type i = 0; i < n; ++i)
            u2(i) = Ay2(i);

        // Even half-step
        for (size_type i = 0; i < n; ++i)
            w(i) -= alpha * u2(i);

        // d = y2 + (theta^2 * eta / alpha) * d
        factor = theta * theta * eta / alpha;
        for (size_type i = 0; i < n; ++i)
            d(i) = y2(i) + factor * d(i);

        theta = mtl::two_norm(w) / tau;
        c = value_type(1) / sqrt(value_type(1) + theta * theta);
        tau *= theta * c;
        eta = c * c * alpha;

        // x += eta * d
        for (size_type i = 0; i < n; ++i)
            x(i) += eta * d(i);

        value_type rho_new = mtl::dot(r_star, w);
        if (rho == value_type(0)) {
            iter.fail(2, "tfqmr breakdown: rho == 0");
            return iter;
        }
        value_type beta = rho_new / rho;
        rho = rho_new;

        // y1 = M^{-1} w
        M.solve(y1, w);

        auto Ay1n = A * y1;
        for (size_type i = 0; i < n; ++i)
            u1(i) = Ay1n(i);

        // v = u1 + beta * (u2 + beta * v)
        for (size_type i = 0; i < n; ++i)
            v(i) = u1(i) + beta * (u2(i) + beta * v(i));
    }

    return iter;
}

} // namespace mtl::itl
