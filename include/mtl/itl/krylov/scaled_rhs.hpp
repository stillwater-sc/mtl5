#pragma once
// MTL5 -- A Krylov solver's own copy of b, scaled by an exact power of two (#924)
#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <cmath>
#include <mtl/concepts/scalar.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/math/pow2_scaling.hpp>

namespace mtl::itl::detail {

/// The right-hand side a Krylov solver works with, for the length of one solve.
///
/// **A copy of b.** The solvers read b again after they have written x: every
/// true-residual check, and gmres on every restart. A call `cg(A, x, x, M, iter)`
/// therefore read its own iterates as the right-hand side and returned x ~ 6e12
/// for a 4x4 SPD system (issue 924). The copy is taken before x is touched, so x
/// may alias b, or share its storage through a view. It is also b evaluated once,
/// when b is an expression.
///
/// **Scaling.** With `scale`, the system is solved as A (s x) = s b, where s is
/// an exact power of two chosen so that ||s b|| is near 1. This happens only when
/// ||b|| is outside [2^-E, 2^E], with E a quarter of the type's max_exponent
/// (32 for float, 256 for double). Inside that band nothing changes, bit for bit.
/// The dot products of cg, bicg, bicgstab, cgs, qmr, tfqmr and bicgstab_ell are
/// quadratic in the residual. In float, b = 1e-30 * ones made dot(p, A p)
/// underflow to 0, so cg and bicgstab reported a breakdown with x = 0. b = 1e20
/// made the dots overflow and x = NaN (issue 924). minres and idr_s scale their
/// own dot products (#671, #874) and take the copy without the scaling.
///
/// Every operation of these solvers is homogeneous in (x, b) when the
/// preconditioner is linear, and multiplying by 2^k is exact in a binary type
/// away from overflow and the subnormals. So the scaled iterates are the
/// unscaled ones times s. x is multiplied by s here and divided by it again
/// when this object is destroyed, on every return and on an exception.
///
/// The iteration object sees the user's scale. Pass a residual norm through
/// `norm(r)` or `unscale(nr)`: the tolerance and norm_r0 are in the units of b.
template <typename VecX, typename VecB>
class scaled_rhs {
public:
    using value_type  = typename VecX::value_type;
    using size_type   = typename VecX::size_type;
    using vector_type = vec::dense_vector<value_type>;
    using mag_type    = decltype(mtl::two_norm(std::declval<const vector_type&>()));

    scaled_rhs(VecX& x, const VecB& b, bool scale) : x_(x), b_(x.size()) {
        const size_type n = x.size();
        for (size_type i = 0; i < n; ++i)
            b_(i) = b(i);
        if (scale) choose_scale();
    }

    scaled_rhs(const scaled_rhs&)            = delete;
    scaled_rhs& operator=(const scaled_rhs&) = delete;

    ~scaled_rhs() {
        if (!active_) return;
        const value_type inv = value_type(inv_);
        for (size_type i = 0; i < x_.size(); ++i)
            x_(i) *= inv;
    }

    /// s * b, in storage of its own.
    const vector_type& b() const { return b_; }

    /// A residual norm of the scaled system, in the units of b.
    mag_type unscale(const mag_type& r) const { return active_ ? mag_type(r * inv_) : r; }

    template <typename V>
    mag_type norm(const V& r) const { return unscale(mag_type(mtl::two_norm(r))); }

private:
    void choose_scale() {
        using limits = std::numeric_limits<mag_type>;
        using pow2   = math::pow2_scaling<mag_type>;
        constexpr int band = limits::max_exponent / 4;
        if constexpr (band < 1) {
            return;
        } else {
            const mag_type nb = mtl::two_norm(b_);
            // b = 0, or a NaN / Inf in b: nothing to scale, and the iteration
            // object reports a non-finite residual.
            if (!(nb > mag_type(0)) || !(nb <= limits::max())) return;
            const int e = pow2::exponent(nb);
            if (e >= -band && e <= band) return;
            // k must lie in [lo, hi]:
            //   - 2^k and 2^-k are normal values of the type;
            //   - s x does not overflow (a starting guess far larger than b):
            //     the largest exponent of x plus k stays below max_exponent - 1;
            //   - s x does not underflow, so that dividing by s on exit gives
            //     back every entry of x, also when the solver makes no step:
            //     the smallest nonzero part of x stays normal, and a subnormal
            //     part of x allows only k >= 0 (scaling it up is exact).
            // The overflow bound and the underflow bound are both kept: a
            // clamp to one must not break the other (#927 review). An empty
            // range, or one that reverses the direction, leaves the system
            // unscaled. An x = 0 sets neither bound.
            const int k0 = -e;
            int lo = limits::min_exponent;
            int hi = limits::max_exponent - 2;
            // Both bounds come from the finite nonzero parts of x, the real
            // and imaginary parts of a complex entry separately: they are what
            // the scaling multiplies. A modulus can overflow where its parts
            // do not (|max + max i| in float), and then gave no bound.
            int emin = 0, emax = 0;
            bool any = false;
            for (size_type i = 0; i < x_.size(); ++i)
                note_exponents(x_(i), emin, emax, any);
            if (any) {
                hi = std::min(hi, limits::max_exponent - 2 - emax);
                lo = std::max(lo, emin >= limits::min_exponent - 1 ? limits::min_exponent - 1 - emin : 0);
            }
            if (lo > hi) return;
            const int k = std::clamp(k0, lo, hi);
            if (k == 0 || (k0 > 0) != (k > 0)) return;
            const mag_type s = pow2::scale(mag_type(1), k);
            inv_ = pow2::scale(mag_type(1), -k);
            const value_type sv = value_type(s);
            for (size_type i = 0; i < b_.size(); ++i)
                b_(i) *= sv;
            for (size_type i = 0; i < x_.size(); ++i)
                x_(i) *= sv;
            active_ = true;
        }
    }

    // The smallest and largest exponents among the finite nonzero parts of v
    // (the real and imaginary parts of a complex v, which are scaled
    // separately). A NaN or Inf part is unchanged by the scaling.
    static void note_exponents(const value_type& v, int& emin, int& emax, bool& any) {
        if constexpr (is_complex_v<value_type>) {
            note_part(mag_type(v.real()), emin, emax, any);
            note_part(mag_type(v.imag()), emin, emax, any);
        } else {
            note_part(mag_type(v), emin, emax, any);
        }
    }

    static void note_part(const mag_type& part, int& emin, int& emax, bool& any) {
        using std::abs;
        const mag_type a = abs(part);
        if (!(a > mag_type(0)) || !(a <= std::numeric_limits<mag_type>::max())) return;
        const int e = math::pow2_scaling<mag_type>::exponent(a);
        if (!any || e < emin) emin = e;
        if (!any || e > emax) emax = e;
        any = true;
    }

    VecX&       x_;
    vector_type b_;
    mag_type    inv_{1};
    bool        active_ = false;
};

} // namespace mtl::itl::detail
