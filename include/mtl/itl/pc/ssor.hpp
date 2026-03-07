#pragma once
// MTL5 -- SSOR (Symmetric Successive Over-Relaxation) preconditioner
// Forward SOR sweep followed by backward SOR sweep.
#include <cassert>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/mat/compressed2D.hpp>

namespace mtl::itl::pc {

/// SSOR preconditioner: symmetric SOR (forward + backward sweep).
/// Generic version using A(i,j) element access.
template <typename Matrix>
class ssor {
    using value_type = typename Matrix::value_type;
    using size_type  = typename Matrix::size_type;
public:
    explicit ssor(const Matrix& A, value_type omega = value_type(1))
        : A_(A), omega_(omega), n_(A.num_rows())
    {
        assert(A.num_rows() == A.num_cols());
    }

    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        assert(x.size() == n_ && b.size() == n_);

        // Copy b into x as working vector
        for (size_type i = 0; i < n_; ++i)
            x(i) = b(i);

        // Forward SOR sweep: rows 0 to n-1
        for (size_type i = 0; i < n_; ++i) {
            auto sigma = math::zero<value_type>();
            for (size_type j = 0; j < n_; ++j) {
                if (j != i)
                    sigma += A_(i, j) * x(j);
            }
            x(i) = omega_ * (b(i) - sigma) / A_(i, i)
                 + (value_type(1) - omega_) * x(i);
        }

        // Backward SOR sweep: rows n-1 to 0
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            auto sigma = math::zero<value_type>();
            for (size_type j = 0; j < n_; ++j) {
                if (j != i)
                    sigma += A_(i, j) * x(j);
            }
            x(i) = omega_ * (b(i) - sigma) / A_(i, i)
                 + (value_type(1) - omega_) * x(i);
        }
    }

    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);  // SSOR is symmetric
    }

private:
    const Matrix& A_;
    value_type omega_;
    size_type n_;
};

/// Specialization for compressed2D: O(nnz) per sweep.
template <typename Value, typename Parameters>
class ssor<mat::compressed2D<Value, Parameters>> {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    using size_type   = typename matrix_type::size_type;
public:
    explicit ssor(const matrix_type& A, value_type omega = value_type(1))
        : A_(A), omega_(omega), n_(A.num_rows()), dia_(A.num_rows())
    {
        assert(A.num_rows() == A.num_cols());
        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data    = A.ref_data();
        for (size_type i = 0; i < n_; ++i) {
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] == i) {
                    dia_(i) = data[k];
                    break;
                }
            }
        }
    }

    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        assert(x.size() == n_ && b.size() == n_);
        const auto& starts  = A_.ref_major();
        const auto& indices = A_.ref_minor();
        const auto& data    = A_.ref_data();

        // Copy b into x
        for (size_type i = 0; i < n_; ++i)
            x(i) = b(i);

        // Forward SOR sweep
        for (size_type i = 0; i < n_; ++i) {
            auto sigma = math::zero<value_type>();
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] != i)
                    sigma += data[k] * x(indices[k]);
            }
            x(i) = omega_ * (b(i) - sigma) / dia_(i)
                 + (value_type(1) - omega_) * x(i);
        }

        // Backward SOR sweep
        for (size_type ii = 0; ii < n_; ++ii) {
            size_type i = n_ - 1 - ii;
            auto sigma = math::zero<value_type>();
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] != i)
                    sigma += data[k] * x(indices[k]);
            }
            x(i) = omega_ * (b(i) - sigma) / dia_(i)
                 + (value_type(1) - omega_) * x(i);
        }
    }

    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);  // SSOR is symmetric
    }

private:
    const matrix_type& A_;
    value_type omega_;
    size_type n_;
    vec::dense_vector<value_type> dia_;
};

} // namespace mtl::itl::pc
