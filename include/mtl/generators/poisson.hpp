#pragma once
// MTL5 -- Poisson 2D Dirichlet matrix generator (factory, returns compressed2D)
// Discretization of -nabla^2 u = f on unit square with Dirichlet BCs.
// Differs from laplacian_2d by including h^2 scaling.
#include <cstddef>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

namespace mtl::generators {

/// 2D Poisson with Dirichlet BCs: (nx*ny) x (nx*ny) 5-point stencil.
/// h_x = 1/(nx+1), h_y = 1/(ny+1)
/// diagonal = 2/h_x^2 + 2/h_y^2, off-diag x = -1/h_x^2, off-diag y = -1/h_y^2
template <typename T = double>
auto poisson2d_dirichlet(std::size_t nx, std::size_t ny) {
    T hx = T(1) / T(nx + 1);
    T hy = T(1) / T(ny + 1);
    T hx2_inv = T(1) / (hx * hx);
    T hy2_inv = T(1) / (hy * hy);
    T diag_val = T(2) * hx2_inv + T(2) * hy2_inv;
    T off_x = -hx2_inv;
    T off_y = -hy2_inv;

    std::size_t n = nx * ny;
    mat::compressed2D<T> P(n, n);
    {
        mat::inserter<mat::compressed2D<T>> ins(P);
        for (std::size_t iy = 0; iy < ny; ++iy) {
            for (std::size_t ix = 0; ix < nx; ++ix) {
                std::size_t row = iy * nx + ix;
                ins[row][row] << diag_val;
                if (ix > 0)
                    ins[row][row - 1] << off_x;
                if (ix + 1 < nx)
                    ins[row][row + 1] << off_x;
                if (iy > 0)
                    ins[row][row - nx] << off_y;
                if (iy + 1 < ny)
                    ins[row][row + nx] << off_y;
            }
        }
    }
    return P;
}

} // namespace mtl::generators
