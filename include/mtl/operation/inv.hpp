#pragma once
// MTL5 — Matrix inverse via LU factorization
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Compute the inverse of a square matrix A via LU factorization.
/// Returns inv(A) as a dense2D. Throws if A is singular.
template <Matrix M>
auto inv(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n);

    // Copy A for in-place LU
    mat::dense2D<value_type> LU(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            LU(i, j) = A(i, j);

    std::vector<size_type> pivot;
    int info = lu_factor(LU, pivot);
    if (info != 0)
        throw std::runtime_error("inv: singular matrix (zero pivot at row " +
                                 std::to_string(info - 1) + ")");

    // Solve A * X = I column by column
    mat::dense2D<value_type> Ainv(n, n);
    vec::dense_vector<value_type> ei(n, math::zero<value_type>());
    vec::dense_vector<value_type> col(n);

    for (size_type j = 0; j < n; ++j) {
        // Set up e_j
        if (j > 0) ei(j - 1) = math::zero<value_type>();
        ei(j) = math::one<value_type>();

        lu_solve(LU, pivot, col, ei);

        for (size_type i = 0; i < n; ++i)
            Ainv(i, j) = col(i);
    }

    return Ainv;
}

} // namespace mtl
