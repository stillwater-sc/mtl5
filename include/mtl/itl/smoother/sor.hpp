#pragma once
// MTL5 — SOR (Successive Over-Relaxation) smoother
// Relaxed Gauss-Seidel: x[i] = omega * GS_update + (1 - omega) * x[i]
#include <cassert>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/mat/compressed2D.hpp>

namespace mtl::itl::smoother {

/// SOR smoother with relaxation parameter omega.
/// omega=1.0 reduces to Gauss-Seidel.
template <typename Matrix>
class sor {
    using value_type = typename Matrix::value_type;
    using size_type  = typename Matrix::size_type;
public:
    explicit sor(const Matrix& A, value_type omega = value_type(1))
        : A_(A), omega_(omega), dia_inv_(A.num_rows())
    {
        for (size_type i = 0; i < A.num_rows(); ++i)
            dia_inv_(i) = value_type(1) / A(i, i);
    }

    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) {
        const size_type n = A_.num_rows();
        assert(x.size() == n && b.size() == n);
        for (size_type i = 0; i < n; ++i) {
            auto sigma = math::zero<value_type>();
            for (size_type j = 0; j < n; ++j) {
                if (j != i)
                    sigma += A_(i, j) * x(j);
            }
            auto gs_update = dia_inv_(i) * (b(i) - sigma);
            x(i) = omega_ * gs_update + (value_type(1) - omega_) * x(i);
        }
        return x;
    }

private:
    const Matrix& A_;
    value_type omega_;
    vec::dense_vector<value_type> dia_inv_;
};

/// Specialization for compressed2D: O(nnz) sweep.
template <typename Value, typename Parameters>
class sor<mat::compressed2D<Value, Parameters>> {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    using size_type   = typename matrix_type::size_type;
public:
    explicit sor(const matrix_type& A, value_type omega = value_type(1))
        : A_(A), omega_(omega), dia_inv_(A.num_rows())
    {
        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data    = A.ref_data();
        for (size_type i = 0; i < A.num_rows(); ++i) {
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] == i) {
                    dia_inv_(i) = value_type(1) / data[k];
                    break;
                }
            }
        }
    }

    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) {
        const size_type n = A_.num_rows();
        assert(x.size() == n && b.size() == n);
        const auto& starts  = A_.ref_major();
        const auto& indices = A_.ref_minor();
        const auto& data    = A_.ref_data();
        for (size_type i = 0; i < n; ++i) {
            auto sigma = math::zero<value_type>();
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] != i)
                    sigma += data[k] * x(indices[k]);
            }
            auto gs_update = dia_inv_(i) * (b(i) - sigma);
            x(i) = omega_ * gs_update + (value_type(1) - omega_) * x(i);
        }
        return x;
    }

private:
    const matrix_type& A_;
    value_type omega_;
    vec::dense_vector<value_type> dia_inv_;
};

} // namespace mtl::itl::smoother
