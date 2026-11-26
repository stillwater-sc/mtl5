// replay.hpp -- the iterates of an iterative solver, for the examples' traces
// (#933).
//
// MTL5's Krylov solvers report an iteration count and a residual, not their
// iterates. A run capped at k iterations follows the same recurrence as the
// full run up to step k, so its x is the full run's x_k: the animations replay
// these, step by step. Used only on the --trace path, where the extra solves
// cost nothing anyone waits for.

#pragma once

#include <mtl/mtl.hpp>

#include <cstddef>
#include <vector>

namespace example_trace {

/// x after exactly k iterations of `solve(x, iter)` from x0 = 0. The solver
/// runs with rtol = 0, so only the cap stops it (or a breakdown).
template <typename Solve>
mtl::vec::dense_vector<double> iterate_after(const mtl::vec::dense_vector<double>& b, int k, Solve solve) {
    mtl::vec::dense_vector<double> x(b.size(), 0.0);
    if (k <= 0) return x;
    mtl::itl::basic_iteration<double> it(b, k, 0.0);
    solve(x, it);
    return x;
}

/// The true relative residual ||b - A x|| / ||b||.
template <typename Matrix>
double true_residual(const Matrix& A, const mtl::vec::dense_vector<double>& x,
                     const mtl::vec::dense_vector<double>& b) {
    mtl::vec::dense_vector<double> r(b - A * x);
    return mtl::two_norm(r) / mtl::two_norm(b);
}

template <typename Vector>
std::vector<double> to_std(const Vector& v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = v(i);
    return out;
}

} // namespace example_trace
