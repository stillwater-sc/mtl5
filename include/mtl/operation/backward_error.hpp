#pragma once
// MTL5 -- normwise backward error of an approximate linear-system solution.
//
// For an approximate solution x of A x = b, the normwise backward error is the
// size of the smallest (normwise) perturbation to the data for which x is the
// exact solution:
//
//     eta(x) = ||b - A x||_inf / ( ||A||_inf ||x||_inf + ||b||_inf )
//
// (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., Thm 7.1).
// This is the natural quality/termination metric for iterative refinement: a
// solution is backward-stable once eta(x) is at the level of the working unit
// roundoff. Universal-free -- generic over any arithmetic type; all norms are
// accumulated in double so the yardstick is precision-independent.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl {

/// Normwise backward error eta(x) = ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf).
/// A must be square and match the length of x and b.
template <typename T, typename PA, typename PV>
double normwise_backward_error(const mat::dense2D<T, PA>& A,
                               const vec::dense_vector<T, PV>& x,
                               const vec::dense_vector<T, PV>& b) {
    const std::size_t n = A.num_rows();
    // Enforce the documented contract (mirrors lu_iterative_refine): A(i,j) is
    // indexed for i,j < n and x[j]/b[i] for j,i < n, so a non-square A or a short
    // x/b would read out of bounds.
    if (A.num_cols() != n)
        throw std::invalid_argument("normwise_backward_error: matrix must be square");
    if (static_cast<std::size_t>(x.size()) != n)
        throw std::invalid_argument("normwise_backward_error: x size does not match A");
    if (static_cast<std::size_t>(b.size()) != n)
        throw std::invalid_argument("normwise_backward_error: b size does not match A");
    double rnorm = 0.0;   // ||b - A x||_inf
    double anorm = 0.0;   // ||A||_inf   (max abs row sum)
    double xnorm = 0.0;   // ||x||_inf
    double bnorm = 0.0;   // ||b||_inf
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0, rowsum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            double aij = static_cast<double>(A(i, j));
            ax     += aij * static_cast<double>(x[j]);
            rowsum += std::abs(aij);
        }
        rnorm = std::max(rnorm, std::abs(static_cast<double>(b[i]) - ax));
        anorm = std::max(anorm, rowsum);
        xnorm = std::max(xnorm, std::abs(static_cast<double>(x[i])));
        bnorm = std::max(bnorm, std::abs(static_cast<double>(b[i])));
    }
    const double denom = anorm * xnorm + bnorm;
    return denom > 0.0 ? rnorm / denom : rnorm;
}

} // namespace mtl
