#pragma once
// MTL5 -- Gauss-Seidel smoother
// Generic version uses A(i,j); compressed2D specialization uses raw CRS access.
#include <mtl/concepts/scalar.hpp>   // Field (#505)
#include <mtl/concepts/matrix.hpp>   // FieldMatrix (#505)
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/mat/compressed2D.hpp>

namespace mtl::itl::smoother {

// Accumulator (optional): accumulation type for the off-diagonal row sum
// (sigma = sum_{j!=i} A(i,j)*x(j)), routed through math::accumulator_traits
// (see math/accumulator_traits.hpp, #158/#259), consistent with the jacobi
// smoother (#262) and the cg (#238) / bicgstab (#255) Krylov solvers. With the
// default `Accumulator = void` the naive value_type accumulation is used
// unchanged -- same behavior and codegen guarantee. A wider or exact (quire)
// accumulator is opt-in and lives downstream (MTL5 stays Universal-free).
//
// Accumulator scope is PER ROW (cleared at the start of each row), the same as
// jacobi. Note that Gauss-Seidel is an in-place sweep: within a single sweep the
// row sum for row i reads the entries x(j), j < i, that were ALREADY updated
// earlier in this sweep. The accumulator only governs the rounding of that one
// row sum; it does not change which (updated vs. old) entries are read. That
// intra-sweep error propagation is exactly what mp-iterative studies on top of
// this per-row accumulation policy.

/// Gauss-Seidel smoother.
/// x[i] = dia_inv[i] * (b[i] - sum_{j!=i} A(i,j)*x[j])
/// operator() runs a forward (ascending-row) sweep; forward()/backward() expose
/// the sweep direction explicitly (backward_gauss_seidel / symmetric_gauss_seidel
/// build on them). All variants carry the optional Accumulator unchanged.
template <FieldMatrix Matrix, typename Accumulator = void>
class gauss_seidel {
    using value_type = typename Matrix::value_type;
    using size_type  = typename Matrix::size_type;
public:
    explicit gauss_seidel(const Matrix& A) : A_(A), dia_inv_(A.num_rows()) {
        for (size_type i = 0; i < A.num_rows(); ++i)
            dia_inv_(i) = value_type(1) / A(i, i);
    }

    /// Forward sweep (rows 0..n-1). Default application.
    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) const { return forward(x, b); }

    /// Forward sweep: rows iterated in ascending order.
    template <typename VecX, typename VecB>
    VecX& forward(VecX& x, const VecB& b) const { return sweep(x, b, /*ascending=*/true); }

    /// Backward sweep: rows iterated in descending order (n-1..0).
    template <typename VecX, typename VecB>
    VecX& backward(VecX& x, const VecB& b) const { return sweep(x, b, /*ascending=*/false); }

private:
    template <typename VecX, typename VecB>
    VecX& sweep(VecX& x, const VecB& b, bool ascending) const {
        const size_type n = A_.num_rows();
        assert(x.size() == n && b.size() == n);
        for (size_type step = 0; step < n; ++step) {
            const size_type i = ascending ? step : (n - 1 - step);
            value_type sigma;
            if constexpr (std::is_void_v<Accumulator>) {
                sigma = math::zero<value_type>();
                for (size_type j = 0; j < n; ++j) {
                    if (j != i)
                        sigma += A_(i, j) * x(j);
                }
            } else {
                using AT = math::accumulator_traits<Accumulator, value_type>;
                Accumulator acc{};
                AT::clear(acc);
                for (size_type j = 0; j < n; ++j) {
                    if (j != i)
                        AT::add_product(acc, static_cast<value_type>(A_(i, j)),
                                             static_cast<value_type>(x(j)));
                }
                sigma = AT::template value<value_type>(acc);
            }
            x(i) = dia_inv_(i) * (b(i) - sigma);
        }
        return x;
    }

    const Matrix& A_;
    vec::dense_vector<value_type> dia_inv_;
};

/// Specialization for compressed2D: O(nnz) sweep using raw CRS arrays.
template <Field Value, typename Parameters, typename Accumulator>
class gauss_seidel<mat::compressed2D<Value, Parameters>, Accumulator> {
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

    /// Forward sweep (rows 0..n-1). Default application.
    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) const { return forward(x, b); }

    /// Forward sweep: rows iterated in ascending order.
    template <typename VecX, typename VecB>
    VecX& forward(VecX& x, const VecB& b) const { return sweep(x, b, /*ascending=*/true); }

    /// Backward sweep: rows iterated in descending order (n-1..0).
    template <typename VecX, typename VecB>
    VecX& backward(VecX& x, const VecB& b) const { return sweep(x, b, /*ascending=*/false); }

private:
    template <typename VecX, typename VecB>
    VecX& sweep(VecX& x, const VecB& b, bool ascending) const {
        const size_type n = A_.num_rows();
        assert(x.size() == n && b.size() == n);
        const auto& starts  = A_.ref_major();
        const auto& indices = A_.ref_minor();
        const auto& data    = A_.ref_data();
        for (size_type step = 0; step < n; ++step) {
            const size_type i = ascending ? step : (n - 1 - step);
            value_type sigma;
            if constexpr (std::is_void_v<Accumulator>) {
                sigma = math::zero<value_type>();
                for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                    if (indices[k] != i)
                        sigma += data[k] * x(indices[k]);
                }
            } else {
                using AT = math::accumulator_traits<Accumulator, value_type>;
                Accumulator acc{};
                AT::clear(acc);
                for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                    if (indices[k] != i)
                        AT::add_product(acc, static_cast<value_type>(data[k]),
                                             static_cast<value_type>(x(indices[k])));
                }
                sigma = AT::template value<value_type>(acc);
            }
            x(i) = dia_inv_(i) * (b(i) - sigma);
        }
        return x;
    }

    const matrix_type& A_;
    vec::dense_vector<value_type> dia_inv_;
    std::vector<size_type> dia_pos_;
};

/// Backward Gauss-Seidel: one in-place sweep in descending row order.
/// Same fixed point as the forward sweep; the sweep direction is what
/// mp-iterative studies against low-precision error propagation.
template <FieldMatrix Matrix, typename Accumulator = void>
class backward_gauss_seidel {
public:
    explicit backward_gauss_seidel(const Matrix& A) : gs_(A) {}

    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) const { return gs_.backward(x, b); }

private:
    gauss_seidel<Matrix, Accumulator> gs_;
};

/// Symmetric Gauss-Seidel (SGS): one forward sweep immediately followed by one
/// backward sweep per application. For symmetric A this application is itself
/// symmetric (SPD-preserving), which is why Krylov methods use it as a
/// preconditioner -- hence a first-class primitive rather than a caller-side
/// compose. Carries the optional Accumulator through both sweeps.
template <FieldMatrix Matrix, typename Accumulator = void>
class symmetric_gauss_seidel {
public:
    explicit symmetric_gauss_seidel(const Matrix& A) : gs_(A) {}

    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) const {
        gs_.forward(x, b);
        return gs_.backward(x, b);
    }

private:
    gauss_seidel<Matrix, Accumulator> gs_;
};

} // namespace mtl::itl::smoother
