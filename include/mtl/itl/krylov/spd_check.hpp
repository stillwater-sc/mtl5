#pragma once
// MTL5 -- the test for a preconditioner that is not positive definite (#949)
#include <cstddef>
#include <limits>
#include <mtl/concepts/scalar.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>

namespace mtl::itl::detail {

/// True when `d`, a computed v^H M^{-1} v (MINRES's ||v||_{M^-1}^2, CG's
/// rho = r^H M^{-1} r), shows that M is not symmetric (Hermitian) positive
/// definite: its real part is negative beyond the rounding of an n-term dot
/// product, n eps ||v||_2 ||M^{-1} v||_2. `norms()` returns that product of
/// norms; it is called only when the real part is negative, so the test costs
/// nothing on an SPD M's ordinary steps.
///
/// The imaginary part of a complex `d` is not tested. It also points to an M
/// that is not Hermitian, but rounding in the preconditioner's solve can give
/// an HPD M one of order cond(M) eps ||v|| ||M^{-1} v||, which no tolerance
/// independent of M separates from that.
template <typename T, typename Norms>
bool not_positive_definite(const T& d, std::size_t n, Norms&& norms) {
    using mag = magnitude_t<T>;
    mag re;
    if constexpr (is_complex_v<T>) re = d.real();
    else                           re = d;
    if (!(re < mag(0))) return false;   // also NaN: the residual test reports that
    const mag tol = mag(n) * std::numeric_limits<mag>::epsilon() * mag(norms());
    return re < -tol;
}

/// The error code of a preconditioner that is not positive definite, as
/// basic_iteration::not_positive_definite (7).
inline constexpr int not_positive_definite_code = 7;
static_assert(basic_iteration<double>::not_positive_definite == not_positive_definite_code);

} // namespace mtl::itl::detail
