#pragma once
// MTL5 -- SOR (Successive Over-Relaxation) smoother
// Relaxed Gauss-Seidel: x[i] = omega * GS_update + (1 - omega) * x[i]
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
// (#262) and gauss_seidel (#263) smoothers and the cg (#238) / bicgstab (#255)
// Krylov solvers. With the default `Accumulator = void` the naive value_type
// accumulation is used unchanged -- same behavior and codegen guarantee. A wider
// or exact (quire) accumulator is opt-in and lives downstream (MTL5 stays
// Universal-free).
//
// Accumulator scope is PER ROW (cleared each row), the same as gauss_seidel.
// The relaxation is deliberately kept OUTSIDE the accumulator: only the row sum
// is accumulated and rounded out to a scalar sigma; the Gauss-Seidel update
// dia_inv*(b - sigma) and the omega blend x_i = omega*gs_update + (1-omega)*x_i
// are ordinary scalar arithmetic on that rounded value. That clean separation
// lets mp-iterative study omega-vs-precision independently of the sum rounding.

/// SOR smoother with relaxation parameter omega.
/// omega=1.0 reduces to Gauss-Seidel.
/// operator() runs a forward (ascending-row) relaxed sweep; forward()/backward()
/// expose the sweep direction explicitly (backward_sor / symmetric_sor build on
/// them). All variants carry the optional Accumulator unchanged.
template <typename Matrix, typename Accumulator = void>
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

    /// Forward relaxed sweep (rows 0..n-1). Default application.
    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) { return forward(x, b); }

    /// Forward relaxed sweep: rows iterated in ascending order.
    template <typename VecX, typename VecB>
    VecX& forward(VecX& x, const VecB& b) { return sweep(x, b, /*ascending=*/true); }

    /// Backward relaxed sweep: rows iterated in descending order (n-1..0).
    template <typename VecX, typename VecB>
    VecX& backward(VecX& x, const VecB& b) { return sweep(x, b, /*ascending=*/false); }

private:
    template <typename VecX, typename VecB>
    VecX& sweep(VecX& x, const VecB& b, bool ascending) {
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
            // omega blend stays outside the accumulator (scalar arithmetic).
            auto gs_update = dia_inv_(i) * (b(i) - sigma);
            x(i) = omega_ * gs_update + (value_type(1) - omega_) * x(i);
        }
        return x;
    }

    const Matrix& A_;
    value_type omega_;
    vec::dense_vector<value_type> dia_inv_;
};

/// Specialization for compressed2D: O(nnz) sweep.
template <typename Value, typename Parameters, typename Accumulator>
class sor<mat::compressed2D<Value, Parameters>, Accumulator> {
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

    /// Forward relaxed sweep (rows 0..n-1). Default application.
    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) { return forward(x, b); }

    /// Forward relaxed sweep: rows iterated in ascending order.
    template <typename VecX, typename VecB>
    VecX& forward(VecX& x, const VecB& b) { return sweep(x, b, /*ascending=*/true); }

    /// Backward relaxed sweep: rows iterated in descending order (n-1..0).
    template <typename VecX, typename VecB>
    VecX& backward(VecX& x, const VecB& b) { return sweep(x, b, /*ascending=*/false); }

private:
    template <typename VecX, typename VecB>
    VecX& sweep(VecX& x, const VecB& b, bool ascending) {
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
            // omega blend stays outside the accumulator (scalar arithmetic).
            auto gs_update = dia_inv_(i) * (b(i) - sigma);
            x(i) = omega_ * gs_update + (value_type(1) - omega_) * x(i);
        }
        return x;
    }

    const matrix_type& A_;
    value_type omega_;
    vec::dense_vector<value_type> dia_inv_;
};

/// Backward SOR: one in-place relaxed sweep in descending row order.
/// Same fixed point as the forward relaxed sweep on an SPD system.
template <typename Matrix, typename Accumulator = void>
class backward_sor {
    using value_type = typename Matrix::value_type;
public:
    explicit backward_sor(const Matrix& A, value_type omega = value_type(1)) : sor_(A, omega) {}

    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) { return sor_.backward(x, b); }

private:
    sor<Matrix, Accumulator> sor_;
};

/// Symmetric SOR (SSOR): one forward relaxed sweep immediately followed by one
/// backward relaxed sweep per application -- the relaxed analogue of symmetric
/// Gauss-Seidel (#269). For symmetric A this application is itself symmetric
/// (SPD-preserving), which is why Krylov methods use it as a preconditioner,
/// hence a first-class primitive rather than a caller-side compose. omega and the
/// optional Accumulator are carried through both sweeps.
template <typename Matrix, typename Accumulator = void>
class symmetric_sor {
    using value_type = typename Matrix::value_type;
public:
    explicit symmetric_sor(const Matrix& A, value_type omega = value_type(1)) : sor_(A, omega) {}

    template <typename VecX, typename VecB>
    VecX& operator()(VecX& x, const VecB& b) {
        sor_.forward(x, b);
        return sor_.backward(x, b);
    }

private:
    sor<Matrix, Accumulator> sor_;
};

} // namespace mtl::itl::smoother
