#pragma once
// MTL5 -- Sparse eigenvalue computation.
//
// Two modes, both matrix-free at the Krylov level (see include/mtl/itl/eigen):
//   * Largest-magnitude eigenvalues: run the iterative eigensolvers directly on
//     the sparse operator (compressed2D is already a LinearOperator, A * x).
//   * Interior / smallest / nearest-sigma eigenvalues: SHIFT-INVERT. Factor
//     (A - sigma*I) once with the sparse LU direct solver, then apply its
//     inverse inside Arnoldi. An eigenpair (theta, y) of (A - sigma*I)^{-1}
//     corresponds to an eigenpair (lambda, y) of A with lambda = sigma + 1/theta
//     and the SAME eigenvector, and "nearest sigma" becomes "largest |theta|".
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <map>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/itl/eigen/arnoldi.hpp>

namespace mtl::sparse {

/// Build B = A - sigma*I as a new CSR sparse matrix. The diagonal entry is
/// created if A has no stored element there.
template <typename Value, typename Parameters>
mat::compressed2D<Value> shift_diagonal(const mat::compressed2D<Value, Parameters>& A,
                                        Value sigma) {
    using size_type = std::size_t;
    const size_type n = A.num_rows();
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();

    std::vector<size_type> starts(n + 1, 0);
    std::vector<size_type> indices;
    std::vector<Value> data;
    for (size_type r = 0; r < n; ++r) {
        starts[r] = indices.size();
        std::map<size_type, Value> row;               // keeps columns sorted
        for (size_type k = rp[r]; k < rp[r + 1]; ++k)
            row[ci[k]] += dat[k];
        row[r] -= sigma;                              // subtract on the diagonal
        for (const auto& [c, v] : row) { indices.push_back(c); data.push_back(v); }
    }
    starts[n] = indices.size();
    return mat::compressed2D<Value>(n, n, static_cast<size_type>(data.size()),
                                    starts.data(), indices.data(), data.data());
}

/// LinearOperator applying (A - sigma*I)^{-1} via a cached sparse LU
/// factorization: factor once, apply many. Plug into the iterative eigensolvers.
template <typename Value>
class shift_invert_operator {
public:
    template <typename Parameters>
    shift_invert_operator(const mat::compressed2D<Value, Parameters>& A, Value sigma)
        : n_(A.num_rows()) {
        auto B = shift_diagonal(A, sigma);

        // Guard against an (near-)exactly singular shift: perturb tiny pivots
        // rather than throwing, scaled to the magnitude of B.
        Value maxabs = Value(0);
        for (Value v : B.ref_data()) { Value a = std::abs(v); if (a > maxabs) maxabs = a; }
        if (maxabs == Value(0)) maxabs = Value(1);
        const Value perturb =
            std::sqrt(std::numeric_limits<Value>::epsilon()) * maxabs;

        auto sym = factorization::sparse_lu_symbolic(B);
        num_ = factorization::sparse_lu_numeric(B, sym, Value(1), perturb);
    }

    /// y = (A - sigma*I)^{-1} x
    template <typename Vec>
    vec::dense_vector<Value> operator*(const Vec& x) const {
        vec::dense_vector<Value> y(n_);
        num_.solve(y, x);
        return y;
    }

    std::size_t num_rows() const { return n_; }
    std::size_t num_cols() const { return n_; }

private:
    std::size_t n_;
    factorization::lu_numeric<Value> num_;
};

/// Compute the `k` eigenvalues of a sparse matrix `A` nearest the shift `sigma`,
/// with their eigenvectors, by Arnoldi on the shift-invert operator
/// (A - sigma*I)^{-1}. Returns complex Ritz pairs: values are the eigenvalues of
/// A (lambda = sigma + 1/theta), vectors are the corresponding eigenvectors.
///
/// `subspace` is the Krylov dimension (default: modest multiple of k, capped at
/// n); `tol` flags convergence via the Arnoldi Ritz residual.
template <typename Value, typename Parameters>
itl::ritz_pairs<std::complex<Value>>
sparse_eigs_shift_invert(const mat::compressed2D<Value, Parameters>& A, Value sigma,
                         std::size_t k, std::size_t subspace = 0,
                         Value tol = Value(1e-8)) {
    using complex_type = std::complex<Value>;
    const std::size_t n = A.num_rows();

    shift_invert_operator<Value> op(A, sigma);

    // Start vector: index-varied to break symmetry.
    vec::dense_vector<Value> v0(n);
    for (std::size_t i = 0; i < n; ++i)
        v0(i) = Value(1) + Value(i) / Value(n == 0 ? 1 : n);

    // Nearest sigma  <=>  largest |theta| of the shift-inverted operator.
    auto r = itl::arnoldi(op, v0, k, itl::eigen_which::largest_magnitude, subspace, tol);

    // Map Ritz values back: lambda = sigma + 1/theta. Eigenvectors are unchanged.
    itl::ritz_pairs<complex_type> out;
    out.values = r.values;
    out.vectors = r.vectors;
    out.subspace = r.subspace;
    out.converged = r.converged;
    for (std::size_t i = 0; i < r.values.size(); ++i) {
        complex_type theta = r.values(i);
        out.values(i) = (std::abs(theta) == Value(0))
                            ? complex_type(std::numeric_limits<Value>::infinity())
                            : complex_type(sigma) + complex_type(Value(1)) / theta;
    }
    return out;
}

/// Compute the `k` eigenvalues of a sparse matrix `A` of largest magnitude (or
/// per `which`) with eigenvectors, by Arnoldi applied directly to the sparse
/// operator -- no factorization. A thin convenience over itl::arnoldi.
template <typename Value, typename Parameters>
itl::ritz_pairs<std::complex<Value>>
sparse_eigs(const mat::compressed2D<Value, Parameters>& A, std::size_t k,
            itl::eigen_which which = itl::eigen_which::largest_magnitude,
            std::size_t subspace = 0, Value tol = Value(1e-8)) {
    const std::size_t n = A.num_rows();
    vec::dense_vector<Value> v0(n);
    for (std::size_t i = 0; i < n; ++i)
        v0(i) = Value(1) + Value(i) / Value(n == 0 ? 1 : n);
    return itl::arnoldi(A, v0, k, which, subspace, tol);
}

} // namespace mtl::sparse
