#pragma once
// MTL5 — Householder reflections for QR factorization
// Computes v, beta such that (I - beta*v*v^T)*x = ||x||*e_1
#include <cmath>
#include <cassert>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>

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
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type vlen = v.size();

    for (size_type j = col; j < n; ++j) {
        // w = v^T * A(:,j)
        T w = math::zero<T>();
        for (size_type i = 0; i < vlen; ++i)
            w += v(i) * A(row + i, j);
        // A(:,j) -= beta * v * w
        for (size_type i = 0; i < vlen; ++i)
            A(row + i, j) -= beta * v(i) * w;
    }
}

/// Apply Householder reflection on the right: A * (I - beta*v*v^T)
/// Modifies columns col..col+vlen-1 of rows row..nrows-1.
template <Matrix M, typename T>
void apply_householder_right(M& A, const vec::dense_vector<T>& v, T beta,
                             typename M::size_type row, typename M::size_type col) {
    using size_type = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type vlen = v.size();

    for (size_type i = row; i < m; ++i) {
        // w = A(i,:) * v
        T w = math::zero<T>();
        for (size_type j = 0; j < vlen; ++j)
            w += A(i, col + j) * v(j);
        // A(i,:) -= beta * w * v^T
        for (size_type j = 0; j < vlen; ++j)
            A(i, col + j) -= beta * w * v(j);
    }
}

} // namespace mtl
