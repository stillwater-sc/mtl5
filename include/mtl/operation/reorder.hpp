#pragma once
// MTL5 — Permutation-based matrix reordering
// Pure functions that return new matrices. No in-place mutation.
#include <cassert>
#include <cstddef>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/permutation_matrix.hpp>

namespace mtl::mat {

/// Reorder rows: B(i,:) = A(perm[i],:)
template <typename Value, typename Params>
dense2D<Value, Params> reorder_rows(const dense2D<Value, Params>& A,
                                     const std::vector<std::size_t>& perm) {
    auto nr = A.num_rows();
    auto nc = A.num_cols();
    assert(perm.size() == nr);
    dense2D<Value, Params> B(nr, nc);
    for (std::size_t i = 0; i < nr; ++i)
        for (std::size_t j = 0; j < nc; ++j)
            B(i, j) = A(perm[i], j);
    return B;
}

/// Reorder columns: B(:,j) = A(:,perm[j])
template <typename Value, typename Params>
dense2D<Value, Params> reorder_cols(const dense2D<Value, Params>& A,
                                     const std::vector<std::size_t>& perm) {
    auto nr = A.num_rows();
    auto nc = A.num_cols();
    assert(perm.size() == nc);
    dense2D<Value, Params> B(nr, nc);
    for (std::size_t i = 0; i < nr; ++i)
        for (std::size_t j = 0; j < nc; ++j)
            B(i, j) = A(i, perm[j]);
    return B;
}

/// Symmetric reorder: B = P * A * P^T
/// B(i,j) = A(perm[i], perm[j])
template <typename Value, typename Params>
dense2D<Value, Params> reorder(const dense2D<Value, Params>& A,
                                const std::vector<std::size_t>& perm) {
    auto n = A.num_rows();
    assert(A.num_cols() == n && "reorder: matrix must be square");
    assert(perm.size() == n);
    dense2D<Value, Params> B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            B(i, j) = A(perm[i], perm[j]);
    return B;
}

/// P * A: permutation_matrix times dense2D → reorder_rows
template <typename PV, typename MV, typename MP>
dense2D<std::common_type_t<PV, MV>, MP>
operator*(const permutation_matrix<PV>& P, const dense2D<MV, MP>& A) {
    const auto& perm = P.permutation();
    assert(perm.size() == A.num_rows());
    using result_value = std::common_type_t<PV, MV>;
    auto nr = A.num_rows();
    auto nc = A.num_cols();
    dense2D<result_value, MP> B(nr, nc);
    for (std::size_t i = 0; i < nr; ++i)
        for (std::size_t j = 0; j < nc; ++j)
            B(i, j) = static_cast<result_value>(A(perm[i], j));
    return B;
}

} // namespace mtl::mat

namespace mtl {
    using mat::reorder_rows;
    using mat::reorder_cols;
    using mat::reorder;
}
