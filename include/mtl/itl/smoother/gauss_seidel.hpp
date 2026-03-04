#pragma once
// MTL5 — Gauss-Seidel smoother
// Generic version uses A(i,j); compressed2D specialization uses raw CRS access.
#include <cassert>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/mat/compressed2D.hpp>

namespace mtl::itl::smoother {

/// Gauss-Seidel smoother: in-place forward sweep.
/// x[i] = dia_inv[i] * (b[i] - sum_{j!=i} A(i,j)*x[j])
template <typename Matrix>
class gauss_seidel {
    using value_type = typename Matrix::value_type;
    using size_type  = typename Matrix::size_type;
public:
    explicit gauss_seidel(const Matrix& A) : A_(A), dia_inv_(A.num_rows()) {
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
            x(i) = dia_inv_(i) * (b(i) - sigma);
        }
        return x;
    }

private:
    const Matrix& A_;
    vec::dense_vector<value_type> dia_inv_;
};

/// Specialization for compressed2D: O(nnz) sweep using raw CRS arrays.
template <typename Value, typename Parameters>
class gauss_seidel<mat::compressed2D<Value, Parameters>> {
    using matrix_type = mat::compressed2D<Value, Parameters>;
    using value_type  = Value;
    using size_type   = typename matrix_type::size_type;
public:
    explicit gauss_seidel(const matrix_type& A) : A_(A), dia_inv_(A.num_rows()) {
        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();
        const auto& data    = A.ref_data();
        dia_pos_.resize(A.num_rows());
        for (size_type i = 0; i < A.num_rows(); ++i) {
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                if (indices[k] == i) {
                    dia_pos_[i] = k;
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
            x(i) = dia_inv_(i) * (b(i) - sigma);
        }
        return x;
    }

private:
    const matrix_type& A_;
    vec::dense_vector<value_type> dia_inv_;
    std::vector<size_type> dia_pos_;
};

} // namespace mtl::itl::smoother
