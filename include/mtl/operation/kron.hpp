#pragma once
// MTL5 — Kronecker product of two matrices
#include <mtl/concepts/matrix.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Kronecker product: C = A (x) B, where C(i*rb+p, j*cb+q) = A(i,j)*B(p,q)
template <Matrix MA, Matrix MB>
auto kron(const MA& A, const MB& B) {
    using value_type = typename MA::value_type;
    using size_type  = typename MA::size_type;
    const size_type ra = A.num_rows(), ca = A.num_cols();
    const size_type rb = B.num_rows(), cb = B.num_cols();

    mat::dense2D<value_type> C(ra * rb, ca * cb);
    for (size_type i = 0; i < ra; ++i)
        for (size_type j = 0; j < ca; ++j)
            for (size_type p = 0; p < rb; ++p)
                for (size_type q = 0; q < cb; ++q)
                    C(i * rb + p, j * cb + q) = A(i, j) * B(p, q);
    return C;
}

} // namespace mtl
