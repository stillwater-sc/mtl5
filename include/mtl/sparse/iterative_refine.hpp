#pragma once
// MTL5 -- generic iterative refinement for sparse direct solves (issue #119)
//
// Refine an approximate solution of A x = b using an existing factorization,
// computing the residual in a (typically higher) Residual precision and solving
// for the correction through the low-precision factors. This is the reusable,
// dependency-free core of mixed-precision iterative refinement: factor cheaply
// in a low precision, then recover accuracy with a few corrections driven by an
// accurate residual.
//
//   r  = b - A x            (Residual precision)
//   dx = U \ (L \ r)        (the factorization's solve, in its own precision)
//   x += dx                 (Residual precision)
//
// MTL5 stays free of any external number library: Residual is any arithmetic
// type (float / double / long double, or a custom type when a caller composes
// one in). The factorization need only expose `solve(x, b)` over generic vector
// types (e.g. klu_numeric, lu_numeric), so the same core serves the
// mixed-precision study in the mp-spice composition layer.
//
// Scaled variant (opt-in): each residual is normalized to O(1) before the
// correction solve and the correction's magnitude is restored in Residual
// precision. This carries the correction MAGNITUDE in Residual precision while
// only its normalized SHAPE passes through the factor type -- which rescues
// narrow-exponent factor types whose corrections would otherwise underflow when
// cast into the factorization's solve.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::sparse {

/// Controls for iterative_refine.
struct refine_options {
    int    max_iter = 20;     ///< maximum correction steps
    double rel_tol  = 0.0;    ///< stop when ||r||inf/||b||inf <= rel_tol (0 = disabled)
    bool   scaled   = false;  ///< normalize the residual before each correction solve
    int    patience = 3;      ///< stop after this many consecutive non-improving steps
                              ///< (>=1; tolerates a noisy low-precision residual so IR
                              ///< reaches its floor instead of quitting on one bad step)
};

/// Outcome of iterative_refine.
struct refine_result {
    int    iters        = 0;      ///< correction steps actually applied
    double rel_residual = 0.0;    ///< best ||r||inf/||b||inf reached (double, for reporting)
    bool   converged    = false;  ///< rel_tol was met
    std::size_t num_perturbed = 0; ///< pivots the factorization perturbed (#185); set by the
                                   ///< drivers that factor (supernodal_lu_solve_refined), 0 here
};

/// Refine x in-place so that A x = b, using factorization `fac`. `x` is the
/// working (Residual) precision and may start at zero -- the first step then
/// produces the initial solve. `A` and `b` are in Residual precision; `fac` is a
/// (possibly lower-precision) factorization exposing `solve(dx, r)`.
///
/// Returns the BEST iterate found (refinement is stopped, and the best x kept,
/// once the residual stops improving), so an over-long max_iter never degrades
/// the result.
template <typename Residual, typename Parameters, typename Factorization>
refine_result iterative_refine(
    const mat::compressed2D<Residual, Parameters>& A,
    const Factorization& fac,
    const vec::dense_vector<Residual>& b,
    vec::dense_vector<Residual>& x,
    const refine_options& opt = {})
{
    using std::abs;
    const std::size_t n = A.num_rows();
    if (A.num_cols() != n)
        throw std::invalid_argument("iterative_refine: matrix must be square");
    if (static_cast<std::size_t>(b.size()) != n || static_cast<std::size_t>(x.size()) != n)
        throw std::invalid_argument("iterative_refine: b/x size does not match A");

    const auto& rp  = A.ref_major();
    const auto& ci  = A.ref_minor();
    const auto& dat = A.ref_data();

    // A non-finite entry makes the norm infinite: std::max(m, NaN) keeps m, so
    // a NaN residual measured 0, and a correction solve that overflowed (a
    // perturbed low-precision factor, #185) was taken as the best iterate and
    // reported converged with rel_residual 0.
    auto norm_inf = [&](const vec::dense_vector<Residual>& v) {
        double m = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double a = static_cast<double>(abs(v(static_cast<int>(i))));
            if (!std::isfinite(a)) return std::numeric_limits<double>::infinity();
            m = std::max(m, a);
        }
        return m;
    };
    const double bnorm = norm_inf(b);

    vec::dense_vector<Residual> r(n), dx(n, Residual{0});
    vec::dense_vector<Residual> best_x = x;
    double best_rn = std::numeric_limits<double>::infinity();
    const int patience = std::max(1, opt.patience);
    int stalls = 0;                          // consecutive non-improving steps

    refine_result res;
    // Each pass evaluates the residual of the current x, then applies one
    // correction, so max_iter corrections get max_iter + 1 evaluations: the
    // last correction is judged like the others (it was dropped, issue 864),
    // and max_iter <= 0 evaluates the supplied x only.
    for (int it = 0; ; ++it) {
        // r = b - A x   (Residual precision)
        for (std::size_t i = 0; i < n; ++i) {
            Residual ax{0};
            for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
                ax += dat[k] * x(static_cast<int>(ci[k]));
            r(static_cast<int>(i)) = b(static_cast<int>(i)) - ax;
        }
        double rn = norm_inf(r);
        // Track the best iterate; a low-precision residual is noisy, so allow up
        // to `patience` consecutive non-improving steps before giving up (the
        // returned x is always the best seen, so extra steps never degrade it).
        if (rn < best_rn || it == 0) { best_rn = rn; best_x = x; stalls = 0; }
        else                         { ++stalls; }

        double rel = (bnorm > 0.0) ? rn / bnorm : rn;
        if (opt.rel_tol > 0.0 && rel <= opt.rel_tol) { res.converged = true; break; }
        if (it >= opt.max_iter) break;       // no corrections left
        if (stalls >= patience) break;       // plateaued / diverging

        // Correction solve through the factorization (in its own precision).
        if (opt.scaled) {
            if (rn == 0.0) break;
            // rn is a double; in a narrower Residual it can round to zero.
            const Residual rho = static_cast<Residual>(rn);
            if (rho == Residual{0}) break;
            for (std::size_t i = 0; i < n; ++i) r(static_cast<int>(i)) /= rho;
            fac.solve(dx, r);
            for (std::size_t i = 0; i < n; ++i)
                x(static_cast<int>(i)) += rho * dx(static_cast<int>(i));
        } else {
            fac.solve(dx, r);
            for (std::size_t i = 0; i < n; ++i)
                x(static_cast<int>(i)) += dx(static_cast<int>(i));
        }
        ++res.iters;
    }

    x = best_x;                              // keep the best iterate
    res.rel_residual = (bnorm > 0.0) ? best_rn / bnorm : best_rn;
    return res;
}

} // namespace mtl::sparse
