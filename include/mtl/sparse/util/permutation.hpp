#pragma once
// MTL5 -- Permutation vector utilities for sparse direct solvers
// A permutation p maps new index -> old index: x_new[i] = x_old[p[i]]
// An inverse permutation pinv maps old index -> new index: pinv[p[i]] = i

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

namespace mtl::sparse::util {

/// Compute the inverse of a permutation vector.
/// Given p where p[new] = old, returns pinv where pinv[old] = new.
inline std::vector<std::size_t> invert_permutation(const std::vector<std::size_t>& p) {
    std::size_t n = p.size();
    std::vector<std::size_t> pinv(n);
    for (std::size_t i = 0; i < n; ++i) {
        assert(p[i] < n);
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
    assert(a.size() == b.size());
    std::size_t n = b.size();
    std::vector<std::size_t> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        assert(b[i] < n);
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
    assert(A.num_rows() == A.num_cols());
    assert(perm.size() == n);

    auto pinv = invert_permutation(perm);

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
    assert(perm.size() == ncols);

    auto pinv = invert_permutation(perm);

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
    assert(perm.size() == nrows);

    auto pinv = invert_permutation(perm);

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
    assert(x.size() == perm.size());
    std::size_t n = x.size();
    std::vector<Value> y(n);
    for (std::size_t i = 0; i < n; ++i)
        y[i] = x[perm[i]];
    return y;
}

/// Apply inverse permutation to a dense vector: y[perm[i]] = x[i]
template <typename Value>
std::vector<Value> ipermute_vector(
    const std::vector<Value>& x,
    const std::vector<std::size_t>& perm)
{
    assert(x.size() == perm.size());
    std::size_t n = x.size();
    std::vector<Value> y(n);
    for (std::size_t i = 0; i < n; ++i)
        y[perm[i]] = x[i];
    return y;
}

} // namespace mtl::sparse::util
