#pragma once
// MTL5 -- the test for a preconditioner that is not positive definite (#949)
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <mtl/concepts/scalar.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>

namespace mtl::itl::detail {

template <typename A> struct spd_accumulator_real { using type = A; };
template <typename R> struct spd_accumulator_real<std::complex<R>> { using type = R; };

/// The rounding unit of a dot product delivered in a value of magnitude type
/// `Mag` and summed in `Accumulator` (void: in the value type, as dot()'s
/// default). The larger of the two epsilons: a float accumulator rounds a
/// double dot product at float's. An accumulator without numeric_limits (an
/// exact or extended one) is taken to round no worse than the delivery type.
template <typename Accumulator, typename Mag>
Mag dot_rounding_unit() {
    Mag e = std::numeric_limits<Mag>::epsilon();
    if constexpr (!std::is_void_v<Accumulator>) {
        using AR = typename spd_accumulator_real<Accumulator>::type;
        if constexpr (std::numeric_limits<AR>::is_specialized) {
            const Mag ea = Mag(std::numeric_limits<AR>::epsilon());
            if (ea > e) e = ea;
        }
    }
    return e;
}

/// True when `d`, a computed v^H M^{-1} v (MINRES's ||v||_{M^-1}^2, CG's
/// rho = r^H M^{-1} r), shows that M is not symmetric (Hermitian) positive
/// definite: its real part is negative beyond the rounding of an n-term dot
/// product, n u ||v||_2 ||M^{-1} v||_2, with u the rounding unit of the
/// arithmetic that formed d (dot_rounding_unit). `norms()` returns the pair
/// (||v||_2, ||M^{-1} v||_2); it is called only when the real part is
/// negative, so the test costs nothing on an SPD M's ordinary steps. The
/// small factor n u is applied before the two norms are multiplied, so their
/// product cannot overflow while d itself is finite.
///
/// The imaginary part of a complex `d` is not tested. It also points to an M
/// that is not Hermitian, but rounding in the preconditioner's solve can give
/// an HPD M one of order cond(M) eps ||v|| ||M^{-1} v||, which no tolerance
/// independent of M separates from that.
template <typename Accumulator = void, typename T, typename Norms>
bool not_positive_definite(const T& d, std::size_t n, Norms&& norms) {
    using mag = magnitude_t<T>;
    mag re;
    if constexpr (is_complex_v<T>) re = d.real();
    else                           re = d;
    if (!(re < mag(0))) return false;   // also NaN: the residual test reports that
    const auto [nv, nz] = norms();
    const mag tol = (mag(n) * dot_rounding_unit<Accumulator, mag>() * mag(nv)) * mag(nz);
    return re < -tol;
}

/// The error code of a preconditioner that is not positive definite, as
/// basic_iteration::not_positive_definite (7).
inline constexpr int not_positive_definite_code = 7;
static_assert(basic_iteration<double>::not_positive_definite == not_positive_definite_code);

} // namespace mtl::itl::detail
