#pragma once
// MTL5 — Laplacian matrix generators (factory, returns compressed2D)
// 1D: tridiagonal [-1, 2, -1]. 2D: 5-point stencil.
// Both are SPD sparse with known eigenvalues.
#include <cstddef>
#include <numbers>
#include <cmath>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

namespace mtl::generators {

/// 1D Laplacian: n x n tridiagonal [-1, 2, -1].
/// SPD with eigenvalues 2 - 2*cos(k*pi/(n+1)), k = 1..n.
template <typename T = double>
auto laplacian_1d(std::size_t n) {
    mat::compressed2D<T> L(n, n);
    {
        mat::inserter<mat::compressed2D<T>> ins(L);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << T(2);
            if (i > 0)
                ins[i][i - 1] << T(-1);
            if (i + 1 < n)
                ins[i][i + 1] << T(-1);
        }
    }
    return L;
}

/// 2D Laplacian: (nx*ny) x (nx*ny) 5-point stencil on an nx x ny grid.
/// SPD sparse, tests iterative solvers. Uses natural (row-major) ordering.
template <typename T = double>
auto laplacian_2d(std::size_t nx, std::size_t ny) {
    std::size_t n = nx * ny;
    mat::compressed2D<T> L(n, n);
    {
        mat::inserter<mat::compressed2D<T>> ins(L);
        for (std::size_t iy = 0; iy < ny; ++iy) {
            for (std::size_t ix = 0; ix < nx; ++ix) {
                std::size_t row = iy * nx + ix;
                ins[row][row] << T(4);
                if (ix > 0)
                    ins[row][row - 1] << T(-1);
                if (ix + 1 < nx)
                    ins[row][row + 1] << T(-1);
                if (iy > 0)
                    ins[row][row - nx] << T(-1);
                if (iy + 1 < ny)
                    ins[row][row + nx] << T(-1);
            }
        }
    }
    return L;
}

} // namespace mtl::generators
