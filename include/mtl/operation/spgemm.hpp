#pragma once
// MTL5 -- Sparse * sparse matrix product (SpGEMM), Gustavson's algorithm.
//
// The product that multigrid most needs to stay sparse is the Galerkin triple
// product A_coarse = R * A * P. Before this, `compressed2D * compressed2D` fell
// through to the generic dense operator*, which both materialised an n x n
// dense intermediate at the FINE-grid size and did O(n^3) work through
// compressed2D::operator()(r, c) -- itself a binary search per access. For the
// 2-D and 3-D problems multigrid exists to solve, that dense intermediate is
// the whole point of not using a dense solver (#402).
//
// Gustavson (1978), "Two fast algorithms for sparse matrices: multiplication
// and permuted transposition", ACM TOMS 4(3), 250-269. Row-by-row:
//
//     for each row i of A:
//         for each (k, a_ik) in row i of A:
//             for each (j, b_kj) in row k of B:
//                 acc[j] += a_ik * b_kj
//         gather acc into row i of C
//
// Work is O(sum_i sum_k nnz(B row k)) -- proportional to the multiply-adds
// actually performed, not to n^3. The accumulator (sparse/util/scatter.hpp,
// CSparse's cs_scatter) makes the inner update O(1) amortised via
// generation-based marking, so no per-row clearing is needed.
//
// SORTING IS NOT COSMETIC. compressed2D::operator()(r, c) locates an entry with
// std::lower_bound over the row's column indices, so ascending indices per row
// are a correctness invariant, not a storage convention. The accumulator hands
// back indices in SCATTER order (first-touch), so each gathered row is sorted
// before it is written.
//
// Measured with the sort removed, 12x12 at 30% density: nnz is correct, row
// sums computed from the raw CSR arrays are correct to 4.4e-16, and element
// access through operator()(r, c) is wrong by 3.2 absolute. A sparse matvec is
// wrong by NOTHING -- it sums data[k]*x(indices[k]) in storage order, which is
// order-independent. So the failure is invisible to nnz checks, row sums, and
// A*x alike, and only a test that reads elements back or inspects the index
// arrays will see it. That is why test_spgemm.cpp asserts the structure
// directly rather than trusting a residual.
//
// Structural zeros are KEPT: an entry whose accumulated value cancels to
// exactly zero stays in the pattern. That is the conventional choice, and the
// right one here -- a Galerkin hierarchy wants a stable pattern across a
// nonlinear solve's re-assemblies, not one that shifts with the numbers.
//
// Single pass, no separate symbolic phase. A symbolic pass buys exact result
// sizing and a reusable pattern when the SAME structure is multiplied
// repeatedly; multigrid setup builds each level once, and std::vector growth is
// amortised O(1), so it would be speculative here. It would slot in as a
// pattern-only sweep ahead of the loop below, filling starts_ before the
// numeric pass.
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/sparse/util/scatter.hpp>

namespace mtl {

/// Sparse-sparse product C = A * B, all in CSR (compressed2D).
/// Requires A.num_cols() == B.num_rows().
template <typename V1, typename P1, typename V2, typename P2>
auto spgemm(const mat::compressed2D<V1, P1>& A,
            const mat::compressed2D<V2, P2>& B) {
    using result_t  = std::common_type_t<V1, V2>;
    using size_type = typename mat::compressed2D<V1, P1>::size_type;

    assert(A.num_cols() == B.num_rows());

    const size_type m = A.num_rows();
    const size_type n = B.num_cols();

    const auto& a_starts  = A.ref_major();
    const auto& a_indices = A.ref_minor();
    const auto& a_data    = A.ref_data();
    const auto& b_starts  = B.ref_major();
    const auto& b_indices = B.ref_minor();
    const auto& b_data    = B.ref_data();

    std::vector<size_type> c_starts(m + 1, size_type(0));
    std::vector<size_type> c_indices;
    std::vector<result_t>  c_data;

    // n == 0 leaves an m x 0 matrix with empty rows, which the loop below
    // produces anyway; the accumulator is sized 1 to keep its workspace valid.
    sparse::util::sparse_accumulator<result_t, size_type> acc(n ? n : size_type(1));
    std::vector<size_type> cols;   // gather buffer, reused across rows

    for (size_type i = 0; i < m; ++i) {
        acc.clear();
        for (size_type ka = a_starts[i]; ka < a_starts[i + 1]; ++ka) {
            const size_type k   = a_indices[ka];
            const result_t  aik = static_cast<result_t>(a_data[ka]);
            for (size_type kb = b_starts[k]; kb < b_starts[k + 1]; ++kb)
                acc.scatter(b_indices[kb], aik * static_cast<result_t>(b_data[kb]));
        }

        // Gather. acc.indices() is in first-touch order; CSR needs ascending.
        cols.assign(acc.indices().begin(), acc.indices().end());
        std::sort(cols.begin(), cols.end());
        for (size_type j : cols) {
            c_indices.push_back(j);
            c_data.push_back(acc(j));
        }
        c_starts[i + 1] = static_cast<size_type>(c_indices.size());
    }

    const size_type nnz = static_cast<size_type>(c_data.size());

    // A structurally empty result takes the (rows, cols) constructor, which
    // builds the same all-zero starts_ this function just computed. The raw
    // constructor would also be correct -- vector::data() may be null for an
    // empty vector, and since C++17 (CWG 232) `null + 0` is a null pointer,
    // so its `indices + nnz_count` is well-defined and the copied range is
    // empty. But that is a rule readers have to know and that did not always
    // hold, so state the empty case instead of relying on it.
    if (nnz == 0)
        return mat::compressed2D<result_t>(m, n);

    return mat::compressed2D<result_t>(m, n, nnz,
                                       c_starts.data(), c_indices.data(), c_data.data());
}

}  // namespace mtl
