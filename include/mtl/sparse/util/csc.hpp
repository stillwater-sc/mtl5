#pragma once
// MTL5 -- Compressed Sparse Column (CSC) utilities for sparse direct solvers
// Most sparse direct algorithms (CSparse, UMFPACK, etc.) work in CSC format.
// This provides conversion from CRS (compressed2D) to CSC and a lightweight
// CSC representation for internal use.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>

namespace mtl::sparse::util {

/// Lightweight CSC (Compressed Sparse Column) representation.
/// Owns its data. Column-oriented storage for sparse direct solvers.
template <typename Value, typename SizeType = std::size_t>
struct csc_matrix {
    SizeType nrows;
    SizeType ncols;
    std::vector<SizeType> col_ptr;   // length ncols + 1
    std::vector<SizeType> row_ind;   // length nnz
    std::vector<Value>    values;    // length nnz

    SizeType nnz() const { return static_cast<SizeType>(values.size()); }

    /// Read-only access: binary search within column c for row r.
    /// Returns zero if element is absent.
    Value operator()(SizeType r, SizeType c) const {
        assert(r < nrows && c < ncols);
        auto begin = row_ind.begin() + col_ptr[c];
        auto end   = row_ind.begin() + col_ptr[c + 1];
        auto it = std::lower_bound(begin, end, r);
        if (it != end && *it == r) {
            return values[it - row_ind.begin()];
        }
        return Value{0};
    }
};

/// Convert a CRS compressed2D matrix to CSC format.
template <typename Value, typename Parameters>
csc_matrix<Value, typename mat::compressed2D<Value, Parameters>::size_type>
crs_to_csc(const mat::compressed2D<Value, Parameters>& A) {
    using size_type = typename mat::compressed2D<Value, Parameters>::size_type;

    size_type nrows = A.num_rows();
    size_type ncols = A.num_cols();
    size_type nnz   = A.nnz();

    const auto& row_ptr = A.ref_major();
    const auto& col_idx = A.ref_minor();
    const auto& data    = A.ref_data();

    csc_matrix<Value, size_type> csc;
    csc.nrows = nrows;
    csc.ncols = ncols;
    csc.col_ptr.assign(ncols + 1, size_type{0});
    csc.row_ind.resize(nnz);
    csc.values.resize(nnz);

    // Count entries per column
    for (size_type k = 0; k < nnz; ++k)
        csc.col_ptr[col_idx[k] + 1]++;

    // Cumulative sum
    for (size_type j = 1; j <= ncols; ++j)
        csc.col_ptr[j] += csc.col_ptr[j - 1];

    // Fill column-compressed arrays
    std::vector<size_type> pos(csc.col_ptr.begin(), csc.col_ptr.end());
    for (size_type r = 0; r < nrows; ++r) {
        for (size_type k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            size_type c = col_idx[k];
            size_type dest = pos[c]++;
            csc.row_ind[dest] = r;
            csc.values[dest]  = data[k];
        }
    }

    return csc;
}

/// Convert a CSC matrix back to CRS compressed2D format.
template <typename Value, typename SizeType>
mat::compressed2D<Value> csc_to_crs(const csc_matrix<Value, SizeType>& csc) {
    SizeType nrows = csc.nrows;
    SizeType ncols = csc.ncols;
    SizeType nnz   = csc.nnz();

    // Count entries per row
    std::vector<SizeType> row_ptr(nrows + 1, SizeType{0});
    for (SizeType k = 0; k < nnz; ++k)
        row_ptr[csc.row_ind[k] + 1]++;

    for (SizeType i = 1; i <= nrows; ++i)
        row_ptr[i] += row_ptr[i - 1];

    std::vector<SizeType> col_idx(nnz);
    std::vector<Value> data(nnz);

    // Fill row-compressed arrays
    std::vector<SizeType> pos(row_ptr.begin(), row_ptr.end());
    for (SizeType c = 0; c < ncols; ++c) {
        for (SizeType k = csc.col_ptr[c]; k < csc.col_ptr[c + 1]; ++k) {
            SizeType r = csc.row_ind[k];
            SizeType dest = pos[r]++;
            col_idx[dest] = c;
            data[dest]    = csc.values[k];
        }
    }

    return mat::compressed2D<Value>(
        nrows, ncols, nnz,
        row_ptr.data(), col_idx.data(), data.data());
}

/// Compute the transpose of a CSC matrix (returns CSC).
/// Transpose of CSC is equivalent to interpreting CRS data as CSC.
template <typename Value, typename SizeType>
csc_matrix<Value, SizeType> transpose_csc(const csc_matrix<Value, SizeType>& A) {
    // Transpose of CSC(m,n) is CSC(n,m):
    // The row indices become column indices and vice versa.
    // This is equivalent to a CRS-to-CSC conversion of the original data.
    SizeType m = A.nrows;
    SizeType n = A.ncols;
    SizeType nnz = A.nnz();

    csc_matrix<Value, SizeType> At;
    At.nrows = n;
    At.ncols = m;
    At.col_ptr.assign(m + 1, SizeType{0});
    At.row_ind.resize(nnz);
    At.values.resize(nnz);

    // Count entries per new column (= per old row)
    for (SizeType k = 0; k < nnz; ++k)
        At.col_ptr[A.row_ind[k] + 1]++;

    for (SizeType j = 1; j <= m; ++j)
        At.col_ptr[j] += At.col_ptr[j - 1];

    // Fill
    std::vector<SizeType> pos(At.col_ptr.begin(), At.col_ptr.end());
    for (SizeType c = 0; c < n; ++c) {
        for (SizeType k = A.col_ptr[c]; k < A.col_ptr[c + 1]; ++k) {
            SizeType r = A.row_ind[k];
            SizeType dest = pos[r]++;
            At.row_ind[dest] = c;
            At.values[dest]  = A.values[k];
        }
    }

    return At;
}

} // namespace mtl::sparse::util
