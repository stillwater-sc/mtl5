#pragma once
// MTL5 -- IDR(s) solver (Induced Dimension Reduction)
// The biorthogonal IDR(s) of van Gijzen & Sonneveld, "Algorithm 913: An
// elegant IDR(s) variant that efficiently exploits biorthogonality
// properties", ACM TOMS 38(1), 2011, with the omega of Sleijpen & van
// Gijzen, SIAM J. Sci. Comput. 32, 2010 (issue 874).
#include <mtl/detail/exceptions.hpp>
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/apply_operator.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl {

/// Options of idr_s (issue 874).
struct idrs_options {
    /// Dimension of the shadow space P. Larger s takes fewer products with A
    /// and more vector work and memory per product: 3s + 4 vectors of
    /// length n (3s + 6 with smoothing). Clamped to n.
    std::size_t s = 4;
    /// The omega of each cycle minimizes ||r - omega t||; with kappa > 0 it
    /// is enlarged when the angle between r and t is wide: when
    /// |cos(r, t)| < kappa, omega is scaled by kappa / |cos(r, t)|
    /// ("maintaining the convergence", Sleijpen & van Gijzen 2010, whose
    /// default is 0.7). The default here is 0, the minimal-residual omega,
    /// from measurements on 2D convection-diffusion, Poisson and shifted
    /// skew systems (issue 874): 0.7 was never clearly better on the first
    /// two and up to 25% slower on Poisson at s <= 2; on sigma I + S it took
    /// 25% fewer iterations at sigma = 0.5, s >= 4, but at sigma = 0.1 the
    /// enlarged omega made the residual grow without bound (1e62 at s = 1),
    /// where kappa = 0 stays bounded.
    double kappa = 0;
    /// Minimal-residual smoothing (Alg. 913): x and r are replaced by a
    /// smoothed pair whose residual norm decreases monotonically. Costs two
    /// vectors and two reductions per product with A.
    bool smoothing = false;
    /// Residual replacement ("reliable updating", Sleijpen & van der Vorst
    /// 1996; van der Vorst & Ye 2000): at the start of a cycle, once the
    /// updated residual has fallen 100x below its largest norm since the last
    /// replacement, it is replaced by b - A x. The gap between the two then
    /// stays at rounding level: without it, IDR(16) on a nonnormal
    /// convection-diffusion system stopped at an updated 9.1e-11 whose true
    /// value was 2.5e-9 (issue 874). One product with A per replacement,
    /// about one per two decades of convergence.
    bool residual_replacement = true;
    /// Seed of the shadow space's random draw. A run is reproducible: the
    /// draw uses std::mt19937_64 and integer arithmetic only, which every
    /// platform computes alike.
    std::uint64_t seed = 42;
};

namespace detail {

template <typename T> struct idrs_is_complex : std::false_type {};
template <typename T> struct idrs_is_complex<std::complex<T>> : std::true_type {};

/// Orthonormalize P in place (modified Gram-Schmidt, run twice, which keeps
/// P orthonormal to working precision). Throws std::invalid_argument when P
/// does not have full rank.
template <typename Accumulator, typename Value>
void idrs_orthonormalize(std::vector<vec::dense_vector<Value>>& P) {
    const std::size_t s = P.size();
    for (std::size_t j = 0; j < s; ++j) {
        for (int pass = 0; pass < 2; ++pass)
            for (std::size_t i = 0; i < j; ++i) {
                const Value h = mtl::dot<Accumulator, Value>(P[i], P[j]);
                for (std::size_t k = 0; k < P[j].size(); ++k)
                    P[j](k) -= h * P[i](k);
            }
        const auto nrm = mtl::two_norm(P[j]);
        if (!(nrm > decltype(nrm)(0)))
            ::mtl::detail::throw_or_abort(std::invalid_argument("idr_s: the shadow space P does not have full rank"));
        for (std::size_t k = 0; k < P[j].size(); ++k)
            P[j](k) /= Value(nrm);
    }
}

/// s vectors of length n with entries uniform in [-1, 1), drawn from seed
/// in the value type itself: 24-bit integers scaled by 2^-23, exact in any
/// type with a 24-bit significand, with no detour through double. Complex
/// value types get complex entries, the recommended shadow space for a
/// complex system.
template <typename Value>
std::vector<vec::dense_vector<Value>> idrs_random_shadow(std::size_t n, std::size_t s, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    auto draw = [&gen]<typename T>(T) {
        const std::int32_t k = static_cast<std::int32_t>(gen() >> 40) - (std::int32_t(1) << 23);
        return T(k) / T(std::int32_t(1) << 23);
    };
    std::vector<vec::dense_vector<Value>> P(s, vec::dense_vector<Value>(n));
    for (std::size_t j = 0; j < s; ++j)
        for (std::size_t i = 0; i < n; ++i) {
            if constexpr (idrs_is_complex<Value>::value) {
                using R = typename Value::value_type;
                const R re = draw(R{});
                const R im = draw(R{});
                P[j](i) = Value(re, im);
            } else {
                P[j](i) = draw(Value{});
            }
        }
    return P;
}

} // namespace detail

/// IDR(s) for nonsymmetric systems A x = b, right preconditioned by M, with
/// the shadow space P given: s = P.size() vectors of length n, of full rank.
/// P is orthonormalized first, which leaves the method unchanged (it
/// depends on the span of P only) and keeps P^H G well conditioned.
///
/// The biorthogonal IDR(s) (van Gijzen & Sonneveld, ACM TOMS 38(1), 2011,
/// Alg. 913). Each cycle is s + 1 steps, each one product with A and one
/// preconditioner solve, and each counted as one iteration. The first s
/// steps keep the new direction G_k orthogonal to P_0 .. P_{k-1}, so the
/// small system P^H G is lower triangular, solved by forward substitution
/// in O(s^2); the last step picks the cycle's omega. This replaces the
/// prototype IDR(s) of 2008 (SIAM J. Sci. Comput. 31, Alg. 3.1), whose s x s
/// system was dense and solved by elimination at every step, and whose
/// iteration counts became erratic at tight tolerances: at rtol 1e-13 on a
/// 2D convection-diffusion system with ILU(0), s = 2 took 729 iterations and
/// s = 8 took 1612, where s = 4 took 71 (issue 874).
///
/// Convergence is judged by the true residual: when the recursively updated
/// residual meets the tolerance, or the iteration limit is reached, b - A x
/// is computed, and if that has not converged the method restarts from the
/// current x, as cg, bicgstab, tfqmr and the rest do (#752, issue 873). The
/// returned iter reports the true residual.
///
/// A near-breakdown in P^H G (a diagonal entry negligible against ||G_k||)
/// restarts the method from the current x. Return codes: 0 converged, 1
/// iteration limit, 2 breakdown in P^H G at the first step after a (re)start,
/// 3 breakdown in omega (t = 0 or omega = 0).
///
/// Accumulator (optional): accumulation type for the dot products and the
/// matrix-vector products. A is a Matrix or any LinearOperator (A * x).
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int idr_s(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter,
          const idrs_options& opt,
          std::vector<vec::dense_vector<typename VecX::value_type>> P) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    using std::abs;
    const size_type n = x.size();
    const std::size_t s = P.size();
    if (s < 1)
        ::mtl::detail::throw_or_abort(std::invalid_argument("idr_s: s must be at least 1"));
    for (const auto& p : P)
        if (p.size() != n)
            ::mtl::detail::throw_or_abort(std::invalid_argument("idr_s: each shadow vector must have length n"));
    detail::idrs_orthonormalize<Accumulator>(P);

    const value_type zero = math::zero<value_type>();
    const value_type one  = math::one<value_type>();

    // Workspace: G, U (s each), r, v, t, u; xs, rs with smoothing.
    // P, G and U are 3s vectors; with r, v, t, u, 3s + 4.
    std::vector<vec::dense_vector<value_type>> G(s, vec::dense_vector<value_type>(n));
    std::vector<vec::dense_vector<value_type>> U(s, vec::dense_vector<value_type>(n));
    vec::dense_vector<value_type> r(n), v(n), t(n), u(n);
    vec::dense_vector<value_type> xs, rs;
    mat::dense2D<value_type> Ms(s, s);           // P^H G, lower triangular
    vec::dense_vector<value_type> f(s), c(s);

    using mag_type = decltype(mtl::two_norm(r));
    const mag_type eps = std::numeric_limits<mag_type>::epsilon();
    const mag_type kappa = mag_type(opt.kappa);

    // r = b - A x, the true residual.
    auto true_residual = [&]() {
        apply_operator<Accumulator>(A, x, t);
        for (size_type i = 0; i < n; ++i)
            r(i) = b(i) - t(i);
    };

    // Minimal-residual smoothing (Alg. 913): rs <- rs - gamma (rs - r),
    // xs <- xs - gamma (xs - x), gamma minimizing ||rs - gamma (rs - r)||.
    auto smooth = [&]() {
        for (size_type i = 0; i < n; ++i)
            u(i) = rs(i) - r(i);
        const value_type uu = mtl::dot<Accumulator, value_type>(u, u);
        if (uu == zero) return;
        const value_type gamma = mtl::dot<Accumulator, value_type>(u, rs) / uu;
        for (size_type i = 0; i < n; ++i) {
            rs(i) -= gamma * u(i);
            xs(i) -= gamma * (xs(i) - x(i));
        }
    };
    // The residual the loop tests: the smoothed one with smoothing.
    auto tested = [&]() -> const vec::dense_vector<value_type>& { return opt.smoothing ? rs : r; };
    // With smoothing the solution is the smoothed iterate.
    auto settle_x = [&]() {
        if (opt.smoothing)
            for (size_type i = 0; i < n; ++i) x(i) = xs(i);
    };

    true_residual();
    if (opt.smoothing) {
        xs = vec::dense_vector<value_type>(n);
        rs = vec::dense_vector<value_type>(n);
    }
    bool r_is_true = true;   // no step since r was computed from x
    for (;;) {
        // (Re)start: G = U = 0, P^H G = I, omega = 1; the smoothed pair is
        // the current one.
        for (std::size_t k = 0; k < s; ++k)
            for (size_type i = 0; i < n; ++i) {
                G[k](i) = zero;
                U[k](i) = zero;
            }
        for (std::size_t i = 0; i < s; ++i)
            for (std::size_t j = 0; j < s; ++j)
                Ms(i, j) = (i == j) ? one : zero;
        if (opt.smoothing)
            for (size_type i = 0; i < n; ++i) {
                xs(i) = x(i);
                rs(i) = r(i);
            }
        value_type omega = one;
        std::size_t steps_since_start = 0;
        bool breakdown = false;
        mag_type peak = mtl::two_norm(r);   // largest ||r|| since the last replacement

        bool done = iter.finished(tested());
        while (!done) {
            // Residual replacement, at the start of a cycle (see idrs_options).
            if (opt.residual_replacement && !r_is_true) {
                const mag_type nr = mtl::two_norm(r);
                if (nr > peak) peak = nr;
                else if (nr < peak / mag_type(100)) {
                    true_residual();
                    peak = mtl::two_norm(r);
                }
            }

            // f = P^H r
            for (std::size_t i = 0; i < s; ++i)
                f(i) = mtl::dot<Accumulator, value_type>(P[i], r);

            for (std::size_t k = 0; k < s; ++k) {
                // Solve the lower triangular Ms(k:s, k:s) c = f(k:s).
                for (std::size_t i = k; i < s; ++i) {
                    value_type sum = f(i);
                    for (std::size_t j = k; j < i; ++j)
                        sum -= Ms(i, j) * c(j);
                    c(i) = sum / Ms(i, i);
                }

                // v = r - G(:, k:s) c; v = M^{-1} v.
                for (size_type i = 0; i < n; ++i)
                    u(i) = r(i);
                for (std::size_t j = k; j < s; ++j)
                    for (size_type i = 0; i < n; ++i)
                        u(i) -= c(j) * G[j](i);
                M.solve(v, u);

                // U_k = U(:, k:s) c + omega v (into u: U_k is read here);
                // G_k = A U_k.
                for (size_type i = 0; i < n; ++i)
                    u(i) = omega * v(i);
                for (std::size_t j = k; j < s; ++j)
                    for (size_type i = 0; i < n; ++i)
                        u(i) += c(j) * U[j](i);
                for (size_type i = 0; i < n; ++i)
                    U[k](i) = u(i);
                apply_operator<Accumulator>(A, U[k], G[k]);

                // Make G_k orthogonal to P_0 .. P_{k-1}.
                for (std::size_t i = 0; i < k; ++i) {
                    const value_type alpha = mtl::dot<Accumulator, value_type>(P[i], G[k]) / Ms(i, i);
                    for (size_type l = 0; l < n; ++l) {
                        G[k](l) -= alpha * G[i](l);
                        U[k](l) -= alpha * U[i](l);
                    }
                }

                // Column k of P^H G.
                for (std::size_t i = k; i < s; ++i)
                    Ms(i, k) = mtl::dot<Accumulator, value_type>(P[i], G[k]);
                // A diagonal entry negligible against ||G_k|| (P is
                // orthonormal, so |P_k^H G_k| <= ||G_k||) is a breakdown;
                // the test also catches 0 and NaN.
                if (!(abs(Ms(k, k)) > eps * mtl::two_norm(G[k]))) {
                    // A near-breakdown of the recurrence, not of the
                    // problem: restart from x, as BiCG-type methods do (on
                    // a shifted skew system, sigma I + S, IDR(1) met one
                    // after 102 iterations and converges after restarts).
                    // Fail only when no step was made since the last start.
                    if (steps_since_start == 0) {
                        settle_x();
                        iter.fail(2, "idr_s breakdown: P^H G is singular");
                        return iter;
                    }
                    breakdown = true;
                    break;
                }

                // r -= beta G_k, x += beta U_k: r becomes orthogonal to P_k.
                const value_type beta = f(k) / Ms(k, k);
                for (size_type i = 0; i < n; ++i) {
                    r(i) -= beta * G[k](i);
                    x(i) += beta * U[k](i);
                }
                ++iter;
                ++steps_since_start;
                r_is_true = false;
                if (opt.smoothing) smooth();
                if (iter.finished(tested())) { done = true; break; }

                // The rest of f: f(k+1:s) -= beta Ms(k+1:s, k).
                for (std::size_t i = k + 1; i < s; ++i)
                    f(i) -= beta * Ms(i, k);
            }
            if (done || breakdown) break;

            // The step into the next space G_{j+1}: v = M^{-1} r, t = A v,
            // omega from (t, r); r -= omega t, x += omega v.
            M.solve(v, r);
            apply_operator<Accumulator>(A, v, t);
            const mag_type nt = mtl::two_norm(t);
            if (!(nt > mag_type(0))) {
                settle_x();
                iter.fail(3, "idr_s breakdown: t = A M^{-1} r is zero");
                return iter;
            }
            const mag_type nr = mtl::two_norm(r);
            const value_type tr = mtl::dot<Accumulator, value_type>(t, r);
            omega = (tr / value_type(nt)) / value_type(nt);
            const mag_type rho = (mag_type(abs(tr)) / nt) / nr;
            if (rho < kappa && rho > mag_type(0))
                omega *= value_type(kappa / rho);
            if (omega == zero) {
                settle_x();
                iter.fail(3, "idr_s breakdown: omega == 0");
                return iter;
            }
            for (size_type i = 0; i < n; ++i) {
                r(i) -= omega * t(i);
                x(i) += omega * v(i);
            }
            ++iter;
            ++steps_since_start;
            r_is_true = false;
            if (opt.smoothing) smooth();
            done = iter.finished(tested());
        }
        if (breakdown) {
            // Restart from x on its true residual (with smoothing, from
            // the smoothed pair).
            settle_x();
            true_residual();
            r_is_true = true;
            if (iter.finished(r)) return iter;
            continue;
        }

        // Converged by the recurrence, or at the limit: the true residual
        // decides (finished() is true again at the limit).
        if (r_is_true) return iter;
        settle_x();
        true_residual();
        r_is_true = true;
        if (iter.finished(r)) return iter;
    }
}

/// IDR(s) with the options in `opt` and a random shadow space drawn from
/// opt.seed (see idrs_options).
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int idr_s(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter,
          const idrs_options& opt) {
    // s = 0 has no shadow space (issue 866).
    if (opt.s < 1)
        ::mtl::detail::throw_or_abort(std::invalid_argument("idr_s: s must be at least 1"));
    const std::size_t n = x.size();
    // An empty system has converged; there is no shadow space to draw.
    if (n == 0) {
        iter.finished(mtl::two_norm(b));
        return iter;
    }
    // More than n orthonormal vectors do not exist; s = n already makes the
    // method exact in n + n/s products (in exact arithmetic).
    const std::size_t s = opt.s < n ? opt.s : (n > 0 ? n : 1);
    return idr_s<Accumulator>(A, x, b, M, iter, opt,
                              detail::idrs_random_shadow<typename VecX::value_type>(n, s, opt.seed));
}

/// IDR(s) with shadow space dimension s and the default options.
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int idr_s(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter,
          typename VecX::size_type s = 4) {
    idrs_options opt;
    opt.s = s;
    return idr_s<Accumulator>(A, x, b, M, iter, opt);
}

} // namespace mtl::itl
