#pragma once
// MTL5 -- Permutation vector utilities for sparse direct solvers
// A permutation p maps new index -> old index: x_new[i] = x_old[p[i]]
// An inverse permutation pinv maps old index -> new index: pinv[p[i]] = i
//
// Sizes and indices are checked once per call, in every build, and a bad one
// throws std::invalid_argument (issue 865): they were asserts, so under NDEBUG
// a short permutation or an index past n read or wrote out of bounds. This is
// the index-check policy of #825/#826, one O(n) pass per call.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

namespace mtl::sparse::util {

/// Compute the inverse of a permutation vector.
/// Given p where p[new] = old, returns pinv where pinv[old] = new.
inline std::vector<std::size_t> invert_permutation(const std::vector<std::size_t>& p) {
    std::size_t n = p.size();
    std::vector<std::size_t> pinv(n, n);   // n: not yet seen
    for (std::size_t i = 0; i < n; ++i) {
        if (p[i] >= n || pinv[p[i]] != n)
            throw std::invalid_argument("invert_permutation: not a permutation of [0, "
                + std::to_string(n) + "): entry " + std::to_string(i) + " is " + std::to_string(p[i]));
        pinv[p[i]] = i;
    }
    return pinv;
}

/// Return the identity permutation of size n: p[i] = i
inline std::vector<std::size_t> identity_permutation(std::size_t n) {
    std::vector<std::size_t> p(n);
    std::iota(p.begin(), p.end(), std::size_t{0});
    return p;
}

/// Compose two permutations: result[i] = a[b[i]]
inline std::vector<std::size_t> compose_permutations(
    const std::vector<std::size_t>& a,
    const std::vector<std::size_t>& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("compose_permutations: sizes differ");
    std::size_t n = b.size();
    std::vector<std::size_t> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (b[i] >= n)
            throw std::invalid_argument("compose_permutations: index " + std::to_string(b[i])
                                        + " out of range");
        result[i] = a[b[i]];
    }
    return result;
}

/// Check if a vector is a valid permutation of [0, n).
inline bool is_valid_permutation(const std::vector<std::size_t>& p) {
    std::size_t n = p.size();
    std::vector<bool> seen(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        if (p[i] >= n || seen[p[i]]) return false;
        seen[p[i]] = true;
    }
    return true;
}

/// Apply a symmetric permutation to a sparse matrix: B = P * A * P^T
/// where P is the permutation matrix corresponding to perm.
/// perm[new] = old, so B(i,j) = A(perm[i], perm[j]).
template <typename Value, typename Parameters>
mat::compressed2D<Value, Parameters> symmetric_permute(
    const mat::compressed2D<Value, Parameters>& A,
    const std::vector<std::size_t>& perm)
{
    using size_type = typename mat::compressed2D<Value, Parameters>::size_type;
    std::size_t n = A.num_rows();
    if (A.num_rows() != A.num_cols())
        throw std::invalid_argument("symmetric_permute: matrix must be square");
    if (perm.size() != n)
        throw std::invalid_argument("symmetric_permute: permutation size does not match the matrix");

    auto pinv = invert_permutation(perm);   // checks perm

    mat::compressed2D<Value, Parameters> B(n, n);
    {
        mat::inserter<mat::compressed2D<Value, Parameters>> ins(B);
        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data    = A.ref_data();

        for (std::size_t old_row = 0; old_row < n; ++old_row) {
            std::size_t new_row = pinv[old_row];
            for (std::size_t k = starts[old_row]; k < starts[old_row + 1]; ++k) {
                std::size_t old_col = indices[k];
                std::size_t new_col = pinv[old_col];
                ins[new_row][new_col] << data[k];
            }
        }
    }
    return B;
}

/// Apply a column permutation to a sparse matrix: B = A * P^T
/// so B(:, j) = A(:, perm[j]).  Equivalently, B(i, pinv[old_col]) = A(i, old_col).
template <typename Value, typename Parameters>
mat::compressed2D<Value, Parameters> column_permute(
    const mat::compressed2D<Value, Parameters>& A,
    const std::vector<std::size_t>& perm)
{
    std::size_t nrows = A.num_rows();
    std::size_t ncols = A.num_cols();
    if (perm.size() != ncols)
        throw std::invalid_argument("column_permute: permutation size does not match the columns");

    auto pinv = invert_permutation(perm);   // checks perm

    mat::compressed2D<Value, Parameters> B(nrows, ncols);
    {
        mat::inserter<mat::compressed2D<Value, Parameters>> ins(B);
        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data    = A.ref_data();

        for (std::size_t r = 0; r < nrows; ++r) {
            for (std::size_t k = starts[r]; k < starts[r + 1]; ++k) {
                std::size_t new_col = pinv[indices[k]];
                ins[r][new_col] << data[k];
            }
        }
    }
    return B;
}

/// Apply a row permutation to a sparse matrix: B = P * A
/// so B(pinv[old_row], :) = A(old_row, :).
template <typename Value, typename Parameters>
mat::compressed2D<Value, Parameters> row_permute(
    const mat::compressed2D<Value, Parameters>& A,
    const std::vector<std::size_t>& perm)
{
    std::size_t nrows = A.num_rows();
    std::size_t ncols = A.num_cols();
    if (perm.size() != nrows)
        throw std::invalid_argument("row_permute: permutation size does not match the rows");

    auto pinv = invert_permutation(perm);   // checks perm

    mat::compressed2D<Value, Parameters> B(nrows, ncols);
    {
        mat::inserter<mat::compressed2D<Value, Parameters>> ins(B);
        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data    = A.ref_data();

        for (std::size_t old_row = 0; old_row < nrows; ++old_row) {
            std::size_t new_row = pinv[old_row];
            for (std::size_t k = starts[old_row]; k < starts[old_row + 1]; ++k) {
                ins[new_row][indices[k]] << data[k];
            }
        }
    }
    return B;
}

/// Apply a permutation to a dense vector: y[i] = x[perm[i]]
template <typename Value>
std::vector<Value> permute_vector(
    const std::vector<Value>& x,
    const std::vector<std::size_t>& perm)
{
    if (x.size() != perm.size())
        throw std::invalid_argument("permute_vector: sizes differ");
    std::size_t n = x.size();
    std::vector<Value> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (perm[i] >= n) throw std::invalid_argument("permute_vector: index out of range");
        y[i] = x[perm[i]];
    }
    return y;
}

/// Apply inverse permutation to a dense vector: y[perm[i]] = x[i]
template <typename Value>
std::vector<Value> ipermute_vector(
    const std::vector<Value>& x,
    const std::vector<std::size_t>& perm)
{
    if (x.size() != perm.size())
        throw std::invalid_argument("ipermute_vector: sizes differ");
    std::size_t n = x.size();
    std::vector<Value> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (perm[i] >= n) throw std::invalid_argument("ipermute_vector: index out of range");
        y[perm[i]] = x[i];
    }
    return y;
}

} // namespace mtl::sparse::util
