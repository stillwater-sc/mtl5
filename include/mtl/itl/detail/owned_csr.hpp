#pragma once
// MTL5 -- a compressed2D copy of a coordinate2D or ell_matrix, for the
// smoothers (issue 867).
//
// The generic Jacobi, Gauss-Seidel and SOR sweeps visit every (i, j) through
// A(i, j): O(n^2) lookups per sweep, and coordinate2D's lookup is itself a
// scan of every entry. The compressed2D specializations sweep the stored
// entries in O(nnz). For coordinate2D and ell_matrix the smoothers convert A
// once, at construction, and run the compressed2D code on the copy.
//
// The copy is held by a shared_ptr in a base class constructed before the
// smoother base, which keeps a reference to it: a copied smoother shares the
// same matrix, so that reference stays valid however the smoother is copied
// (a multigrid captures its smoothers by value).
#include <cstddef>
#include <memory>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/ell_matrix.hpp>
#include <mtl/mat/inserter.hpp>

namespace mtl::itl::detail {

// The copy is always a row-major compressed2D<Value>, whatever A's
// parameters: compressed2D rejects column-major ones, and a column-major
// coordinate2D smoother has to keep compiling (#870 review).
template <typename Value, typename Parameters>
mat::compressed2D<Value> to_compressed(const mat::coordinate2D<Value, Parameters>& A) {
    // The same triplets in a row-major coordinate2D: its compress() sums
    // duplicates in insertion order, as A(i, j) does.
    mat::coordinate2D<Value> rm(A.num_rows(), A.num_cols());
    for (const auto& [r, c, v] : A.ref_entries()) rm.insert(r, c, v);
    return rm.compress();
}

template <typename Value, typename Parameters>
mat::compressed2D<Value> to_compressed(const mat::ell_matrix<Value, Parameters>& A) {
    using size_type = typename mat::ell_matrix<Value, Parameters>::size_type;
    mat::compressed2D<Value> C(A.num_rows(), A.num_cols());
    {
        mat::inserter<mat::compressed2D<Value>> ins(C);
        const auto& idx = A.ref_indices();
        const auto& val = A.ref_data();
        const size_type w = A.max_width();
        for (size_type r = 0; r < A.num_rows(); ++r)
            for (size_type k = 0; k < w; ++k) {
                const size_type c = idx[r * w + k];
                if (c == mat::ell_matrix<Value, Parameters>::invalid) break;
                ins[r][c] << val[r * w + k];
            }
    }
    return C;
}

/// The owned compressed2D copy, as a base constructed before the smoother.
template <typename Value>
struct owned_csr {
    using csr_type = mat::compressed2D<Value>;
    template <typename Matrix>
    explicit owned_csr(const Matrix& A) : csr_(std::make_shared<const csr_type>(to_compressed(A))) {}
    std::shared_ptr<const csr_type> csr_;
};

} // namespace mtl::itl::detail
