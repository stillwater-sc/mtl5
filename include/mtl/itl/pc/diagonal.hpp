#pragma once
// MTL5 — Diagonal (Jacobi) preconditioner: stores inv(diag(A))
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/operation/diagonal.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/functor/scalar/conj.hpp>

namespace mtl::itl::pc {

/// Jacobi preconditioner — stores the inverse diagonal of A.
/// solve(x, b): x(i) = inv_diag(i) * b(i)
template <typename Matrix>
class diagonal {
    using value_type = typename Matrix::value_type;
    using size_type  = typename Matrix::size_type;

    vec::dense_vector<value_type> inv_diag_;

public:
    explicit diagonal(const Matrix& A) {
        auto d = mtl::diagonal(A);
        inv_diag_ = vec::dense_vector<value_type>(d.size());
        for (size_type i = 0; i < d.size(); ++i)
            inv_diag_(i) = value_type(1) / d(i);
    }

    /// solve: x(i) = inv_diag(i) * b(i)
    template <typename VectorOut, typename VectorIn>
    void solve(VectorOut& x, const VectorIn& b) const {
        assert(x.size() == b.size());
        assert(x.size() == inv_diag_.size());
        for (typename VectorIn::size_type i = 0; i < b.size(); ++i)
            x(i) = inv_diag_(i) * b(i);
    }

    /// adjoint_solve: x(i) = conj(inv_diag(i)) * b(i)
    template <typename VectorOut, typename VectorIn>
    void adjoint_solve(VectorOut& x, const VectorIn& b) const {
        assert(x.size() == b.size());
        assert(x.size() == inv_diag_.size());
        for (typename VectorIn::size_type i = 0; i < b.size(); ++i)
            x(i) = functor::scalar::conj<value_type>::apply(inv_diag_(i)) * b(i);
    }
};

} // namespace mtl::itl::pc
