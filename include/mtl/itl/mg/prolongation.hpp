#pragma once
// MTL5 -- Prolongation operator for multigrid (coarse -> fine transfer)
// Linear interpolation for 1D geometric multigrid.
// P = 2 * R^T is the standard relationship.
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

namespace mtl::itl::mg {

/// Builds a 1D linear interpolation prolongation matrix (coarse -> fine).
/// Maps a vector of size n_coarse to size n_fine = 2 * n_coarse + 1.
/// Coarse points are copied; fine points are averaged from neighbors.
inline mat::compressed2D<double> make_prolongation_1d(std::size_t n_coarse) {
    std::size_t n_fine = 2 * n_coarse + 1;
    mat::compressed2D<double> P(n_fine, n_coarse);
    {
        mat::inserter<mat::compressed2D<double>> ins(P);
        for (std::size_t i = 0; i < n_coarse; ++i) {
            std::size_t j = 2 * i + 1;  // fine grid point for coarse point i
            // Coarse point maps directly (weight 1)
            ins[j][i] << 1.0;
            // Left neighbor (weight 0.5)
            if (j > 0)            ins[j - 1][i] << 0.5;
            // Right neighbor (weight 0.5)
            if (j + 1 < n_fine)   ins[j + 1][i] << 0.5;
        }
    }
    return P;
}

/// Prolongation: interpolate coarse-grid vector to fine grid.
/// x_fine = P * x_coarse
template <typename Value>
vec::dense_vector<Value> prolongate(const mat::compressed2D<Value>& P,
                                    const vec::dense_vector<Value>& coarse) {
    using size_type = std::size_t;
    const size_type nf = P.num_rows();
    vec::dense_vector<Value> fine(nf);
    const auto& starts  = P.ref_major();
    const auto& indices = P.ref_minor();
    const auto& data    = P.ref_data();
    for (size_type i = 0; i < nf; ++i) {
        Value sum{};
        for (size_type k = starts[i]; k < starts[i + 1]; ++k)
            sum += data[k] * coarse(indices[k]);
        fine(i) = sum;
    }
    return fine;
}

} // namespace mtl::itl::mg
