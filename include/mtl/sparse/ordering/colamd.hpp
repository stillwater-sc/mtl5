#pragma once
// MTL5 -- Column Approximate Minimum Degree (COLAMD) ordering
//
// COLAMD computes a column permutation for unsymmetric sparse matrices
// that reduces fill-in during LU or QR factorization. It operates on
// the column structure of A, effectively computing an AMD ordering on
// A^T*A without explicitly forming the product.
//
// The key idea: the fill pattern of LU(A*Q) or QR(A*Q) depends on the
// column intersection graph of A*Q, which is the same as the adjacency
// graph of Q^T * A^T * A * Q. So a good column ordering for LU/QR is
// equivalent to a good symmetric ordering for A^T*A.
//
// This implementation computes the column intersection graph (A^T*A
// structure without values) and applies AMD to it.
//
// Reference: Davis, Gilbert, Larimore, Ng, "A Column Approximate Minimum
//            Degree Ordering Algorithm", ACM Trans. Math. Softw., 2004.
//            Davis, "Direct Methods for Sparse Linear Systems", SIAM, Ch. 7.

#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/sparse/ordering/minimum_degree.hpp>

namespace mtl::sparse::ordering {

/// Column Approximate Minimum Degree ordering.
/// Callable object: given a (possibly rectangular) sparse matrix, returns
/// a column permutation that reduces fill-in during LU or QR factorization.
struct colamd {

    /// Compute the COLAMD ordering for a sparse matrix.
    /// Returns permutation p where p[new] = old (column indices).
    ///
    /// Runs the near-linear quotient-graph minimum-degree algorithm
    /// (CSparse cs_amd, order 2) on the column-intersection graph A^T*A,
    /// dropping dense rows. Works for rectangular and unsymmetric matrices.
    template <typename Value, typename Parameters>
    std::vector<std::size_t> operator()(
        const mat::compressed2D<Value, Parameters>& A) const
    {
        return detail::minimum_degree(/*order=*/2, A);
    }
};

} // namespace mtl::sparse::ordering
