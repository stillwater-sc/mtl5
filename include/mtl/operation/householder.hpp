#pragma once
// MTL5 -- Householder reflections for QR factorization
// Computes v, beta such that (I - beta*v*v^T)*x = ||x||*e_1
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/detail/thread_pool.hpp>

namespace mtl {

/// Compute Householder vector v and scalar beta for a column vector x.
/// The reflection (I - beta*v*v^T) zeroes out x(1:end), leaving x(0) = -sign(x0)*||x||.
/// v(0) is always 1 (implicit). Returns {v, beta}.
template <typename T>
std::pair<vec::dense_vector<T>, T> householder(const vec::dense_vector<T>& x) {
    using std::sqrt;
    using std::abs;
    using size_type = typename vec::dense_vector<T>::size_type;
    const size_type n = x.size();

    vec::dense_vector<T> v(n);
    for (size_type i = 0; i < n; ++i)
        v(i) = x(i);

    // Compute sigma = sum(x(1:end)^2)
    T sigma = math::zero<T>();
    for (size_type i = 1; i < n; ++i)
        sigma += x(i) * x(i);

    v(0) = math::one<T>();

    if (sigma == math::zero<T>()) {
        // x is already along e_1
        return {v, math::zero<T>()};
    }

    T norm_x = sqrt(x(0) * x(0) + sigma);
    if (x(0) <= math::zero<T>()) {
        v(0) = x(0) - norm_x;
    } else {
        v(0) = -sigma / (x(0) + norm_x);
    }

    T beta = T(2) * v(0) * v(0) / (sigma + v(0) * v(0));

    // Normalize v so that v(0) = 1
    T v0 = v(0);
    for (size_type i = 0; i < n; ++i)
        v(i) /= v0;

    return {v, beta};
}

/// Apply Householder reflection (I - beta*v*v^T) to columns col..ncols-1
/// of matrix A, rows row..nrows-1. Modifies A in-place.
template <Matrix M, typename T>
void apply_householder_left(M& A, const vec::dense_vector<T>& v, T beta,
                            typename M::size_type row, typename M::size_type col) {
    using size_type = typename M::size_type;
    const size_type n = A.num_cols();
    const size_type vlen = v.size();

    // Each column j is updated independently (reads the shared reflector v and
    // only column j of A, writes only column j), so partitioning the columns
    // across the thread pool is bit-identical to serial. No-op at
    // MTL5_NUM_THREADS=1.
    // Empty or beyond-end target column: no work (matches the serial loop, and
    // avoids an unsigned underflow in n - col before deriving the work range).
    if (col >= n) return;
    const size_type ncols = n - col;
    const std::size_t grain = std::max<std::size_t>(
        std::size_t{1}, std::size_t{65536} / (vlen ? static_cast<std::size_t>(vlen) : std::size_t{1}));
    detail::thread_pool::instance().parallel_for(
        static_cast<std::size_t>(ncols), grain,
        [&](std::size_t b, std::size_t e) {
            for (std::size_t t = b; t < e; ++t) {
                const size_type j = col + static_cast<size_type>(t);
                // w = v^T * A(:,j)
                T w = math::zero<T>();
                for (size_type i = 0; i < vlen; ++i)
                    w += v(i) * A(row + i, j);
                // A(:,j) -= beta * v * w
                for (size_type i = 0; i < vlen; ++i)
                    A(row + i, j) -= beta * v(i) * w;
            }
        });
}

/// Apply Householder reflection on the right: A * (I - beta*v*v^T)
/// Modifies columns col..col+vlen-1 of rows row..nrows-1.
template <Matrix M, typename T>
void apply_householder_right(M& A, const vec::dense_vector<T>& v, T beta,
                             typename M::size_type row, typename M::size_type col) {
    using size_type = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type vlen = v.size();

    // Each row i is updated independently (reads the shared reflector v and only
    // its own vlen entries of A, writes only those), so partitioning the rows
    // across the thread pool is bit-identical to serial. No-op at
    // MTL5_NUM_THREADS=1.
    // Empty or beyond-end target row: no work (matches the serial loop, and
    // avoids an unsigned underflow in m - row before deriving the work range).
    if (row >= m) return;
    const size_type nrows = m - row;
    const std::size_t grain = std::max<std::size_t>(
        std::size_t{1}, std::size_t{65536} / (vlen ? static_cast<std::size_t>(vlen) : std::size_t{1}));
    detail::thread_pool::instance().parallel_for(
        static_cast<std::size_t>(nrows), grain,
        [&](std::size_t b, std::size_t e) {
            for (std::size_t t = b; t < e; ++t) {
                const size_type i = row + static_cast<size_type>(t);
                // w = A(i,:) * v
                T w = math::zero<T>();
                for (size_type j = 0; j < vlen; ++j)
                    w += A(i, col + j) * v(j);
                // A(i,:) -= beta * w * v^T
                for (size_type j = 0; j < vlen; ++j)
                    A(i, col + j) -= beta * w * v(j);
            }
        });
}

} // namespace mtl
