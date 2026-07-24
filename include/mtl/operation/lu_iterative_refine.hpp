#pragma once
// MTL5 -- dense LU iterative refinement (mixed precision).
//
// The dense counterpart of sparse/iterative_refine.hpp. Solve A x = b by
// factoring A once in a (low) Working precision, then correcting the solution
// with a residual formed in the (higher) Residual precision:
//
//     factor  A ~= P L U            (Working precision, cheap)
//     solve   x  = U \ (L \ P b)    (Working precision)
//     repeat  r  = b - A x          (Residual precision)
//             d  = U \ (L \ P r)    (Working precision)
//             x += d                (Residual precision)
//
// The expensive factorization runs in low precision while an accurate residual
// recovers accuracy far beyond the working precision's own solve (Wilkinson;
// Higham, "Accuracy and Stability of Numerical Algorithms", ch. 12). This is
// the reusable, dependency-free core: Working and Residual are any arithmetic
// types (float / double / long double, or a custom type when a caller composes
// one in). MTL5 never depends on an external number library.
//
// Options and result mirror sparse/iterative_refine: best-iterate return
// (refinement stops once the residual stops improving, so an over-long max_iter
// never degrades the result) and patience for a noisy low-precision residual.
//
// Scaled variant (opt-in): each residual is normalized to O(1) before the
// Working-precision correction solve and the correction magnitude is restored
// in Residual precision -- carrying the magnitude in Residual precision while
// only the normalized shape passes through the low-precision factor, which
// rescues narrow-exponent Working types whose corrections would otherwise
// underflow. (This is the "scale-and-round" squeeze from the Universal study.)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>

namespace mtl {

/// Controls for lu_iterative_refine.
struct lu_refine_options {
    int    max_iter = 20;     ///< maximum correction steps
    double rel_tol  = 0.0;    ///< stop when ||r||inf/||b||inf <= rel_tol (0 = disabled)
    bool   scaled   = false;  ///< normalize the residual before each correction solve
    int    patience = 3;      ///< stop after this many consecutive non-improving steps (>=1)
};

/// Outcome of lu_iterative_refine.
struct lu_refine_result {
    int    iters        = 0;      ///< correction steps actually applied
    double rel_residual = 0.0;    ///< best ||r||inf/||b||inf reached (double, for reporting)
    bool   converged    = false;  ///< rel_tol was met
};

/// Solve A x = b by LU iterative refinement.
///   Working  -- precision of the LU factorization and correction solves
///               (explicit template argument, typically lower than Residual).
///   Residual -- precision of A, b, x and the residual (deduced from A).
/// A and b are supplied in Residual precision; the refined x is returned in
/// Residual precision (any prior contents are overwritten by the initial solve).
///
/// Returns the BEST iterate found.
template <typename Working, typename Residual, typename PA, typename PV>
lu_refine_result lu_iterative_refine(
    const mat::dense2D<Residual, PA>& A,
    const vec::dense_vector<Residual, PV>& b,
    vec::dense_vector<Residual, PV>& x,
    const lu_refine_options& opt = {})
{
    const std::size_t n = A.num_rows();
    if (A.num_cols() != n)
        throw std::invalid_argument("lu_iterative_refine: matrix must be square");
    if (static_cast<std::size_t>(b.size()) != n)
        throw std::invalid_argument("lu_iterative_refine: b size does not match A");

    using WMat = mat::dense2D<Working>;
    using WVec = vec::dense_vector<Working>;
    using RVec = vec::dense_vector<Residual>;

    // Factor a Working-precision copy of A (the cheap low-precision step).
    WMat LU(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) LU(i, j) = static_cast<Working>(A(i, j));
    std::vector<typename WMat::size_type> pivot;
    lu_factor(LU, pivot);

    // ||b - A x||_inf, accumulated in double for a precision-independent yardstick.
    auto residual_norm = [&](const RVec& xx) {
        double m = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double ax = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                ax += static_cast<double>(A(i, j)) * static_cast<double>(xx[j]);
            m = std::max(m, std::abs(static_cast<double>(b[i]) - ax));
        }
        return m;
    };
    double bnorm = 0.0;
    for (std::size_t i = 0; i < n; ++i) bnorm = std::max(bnorm, std::abs(static_cast<double>(b[i])));

    // Initial solve x = U \ (L \ b) in Working precision, cast up to Residual.
    {
        WVec bw(n), xw(n, Working(0));
        for (std::size_t i = 0; i < n; ++i) bw[i] = static_cast<Working>(b[i]);
        lu_solve(LU, pivot, xw, bw);
        x = RVec(n, Residual(0));
        for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<Residual>(xw[i]);
    }

    RVec   best_x  = x;
    double best_rn = std::numeric_limits<double>::infinity();
    const int patience = std::max(1, opt.patience);
    int    stalls = 0;

    lu_refine_result res;
    for (int it = 0; it < opt.max_iter; ++it) {
        // r = b - A x, formed in Residual precision.
        RVec r(n, Residual(0));
        for (std::size_t i = 0; i < n; ++i) {
            Residual ax{0};
            for (std::size_t j = 0; j < n; ++j) ax += A(i, j) * x[j];
            r[i] = b[i] - ax;
        }
        double rn = residual_norm(x);
        if (rn < best_rn) { best_rn = rn; best_x = x; stalls = 0; }
        else              { ++stalls; }

        double rel = (bnorm > 0.0) ? rn / bnorm : rn;
        if (opt.rel_tol > 0.0 && rel <= opt.rel_tol) { res.converged = true; break; }
        if (stalls >= patience) break;

        // Correction solve through the Working-precision factor.
        WVec rw(n, Working(0)), dw(n, Working(0));
        if (opt.scaled) {
            if (rn == 0.0) break;
            const Residual rho = static_cast<Residual>(rn);
            for (std::size_t i = 0; i < n; ++i) rw[i] = static_cast<Working>(r[i] / rho);
            lu_solve(LU, pivot, dw, rw);
            for (std::size_t i = 0; i < n; ++i) x[i] += rho * static_cast<Residual>(dw[i]);
        } else {
            for (std::size_t i = 0; i < n; ++i) rw[i] = static_cast<Working>(r[i]);
            lu_solve(LU, pivot, dw, rw);
            for (std::size_t i = 0; i < n; ++i) x[i] += static_cast<Residual>(dw[i]);
        }
        ++res.iters;
    }

    if (!std::isfinite(best_rn)) best_rn = residual_norm(x);   // max_iter <= 0

    x = best_x;                                                // keep the best iterate
    res.rel_residual = (bnorm > 0.0) ? best_rn / bnorm : best_rn;
    return res;
}

} // namespace mtl
