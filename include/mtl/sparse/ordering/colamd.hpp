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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/mat/inserter.hpp>

namespace mtl::sparse::ordering {

/// Column Approximate Minimum Degree ordering.
/// Callable object: given a (possibly rectangular) sparse matrix, returns
/// a column permutation that reduces fill-in during LU or QR factorization.
struct colamd {

    /// Compute the COLAMD ordering for a sparse matrix.
    /// Returns permutation p where p[new] = old (column indices).
    ///
    /// For square symmetric matrices, this reduces to AMD.
    /// For rectangular or unsymmetric matrices, computes AMD on
    /// the column intersection graph (structure of A^T*A).
    template <typename Value, typename Parameters>
    std::vector<std::size_t> operator()(
        const mat::compressed2D<Value, Parameters>& A) const
    {
        using size_type = std::size_t;
        size_type m = A.num_rows();
        size_type n = A.num_cols();

        if (n == 0) return {};

        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();

        // Build the column intersection graph: A^T*A structure (n x n)
        // Two columns i and j are connected if they share a common row
        // (i.e., there exists a row r where A(r,i) != 0 and A(r,j) != 0).

        // First, build column-to-row mapping (transpose structure)
        std::vector<std::vector<size_type>> col_rows(n);
        for (size_type r = 0; r < m; ++r) {
            for (size_type k = starts[r]; k < starts[r + 1]; ++k) {
                col_rows[indices[k]].push_back(r);
            }
        }

        // Build column intersection graph as compressed2D
        // For each row, all pairs of columns in that row are connected
        mat::compressed2D<double> AtA(n, n);
        {
            mat::inserter<mat::compressed2D<double>> ins(AtA);

            // For each row, enumerate column pairs
            for (size_type r = 0; r < m; ++r) {
                // Collect columns present in this row
                std::vector<size_type> row_cols;
                for (size_type k = starts[r]; k < starts[r + 1]; ++k)
                    row_cols.push_back(indices[k]);

                // Add edges (including self-loops for diagonal)
                for (size_type ci : row_cols) {
                    for (size_type cj : row_cols) {
                        ins[ci][cj] << 1.0;
                    }
                }
            }
        }

        // Apply AMD to the column intersection graph
        amd amd_ordering;
        return amd_ordering(AtA);
    }
};

} // namespace mtl::sparse::ordering
