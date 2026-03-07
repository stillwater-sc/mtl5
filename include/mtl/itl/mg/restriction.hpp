#pragma once
// MTL5 — Restriction operator for multigrid (fine → coarse transfer)
// Full-weighting stencil [1/4, 1/2, 1/4] for 1D geometric multigrid.
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

namespace mtl::itl::mg {

/// Builds a 1D full-weighting restriction matrix (fine → coarse).
/// Maps a vector of size n_fine to size n_coarse = (n_fine - 1) / 2.
/// Uses the standard [1/4, 1/2, 1/4] stencil.
inline mat::compressed2D<double> make_restriction_1d(std::size_t n_fine) {
    std::size_t n_coarse = (n_fine - 1) / 2;
    mat::compressed2D<double> R(n_coarse, n_fine);
    {
        mat::inserter<mat::compressed2D<double>> ins(R);
        for (std::size_t i = 0; i < n_coarse; ++i) {
            std::size_t j = 2 * i + 1;  // fine grid point corresponding to coarse point i
            if (j > 0)          ins[i][j - 1] << 0.25;
            ins[i][j]     << 0.5;
            if (j + 1 < n_fine) ins[i][j + 1] << 0.25;
        }
    }
    return R;
}

/// Restriction: project fine-grid vector to coarse grid.
/// r_coarse = R * r_fine
template <typename Value>
vec::dense_vector<Value> restrict(const mat::compressed2D<Value>& R,
                                  const vec::dense_vector<Value>& fine) {
    using size_type = std::size_t;
    const size_type nc = R.num_rows();
    vec::dense_vector<Value> coarse(nc);
    const auto& starts  = R.ref_major();
    const auto& indices = R.ref_minor();
    const auto& data    = R.ref_data();
    for (size_type i = 0; i < nc; ++i) {
        Value sum{};
        for (size_type k = starts[i]; k < starts[i + 1]; ++k)
            sum += data[k] * fine(indices[k]);
        coarse(i) = sum;
    }
    return coarse;
}

} // namespace mtl::itl::mg
