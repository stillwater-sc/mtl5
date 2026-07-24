#pragma once
// MTL5 -- magic square generator (factory, returns dense2D)
// A magic square of order N contains the integers 1..N^2 arranged so that
// every row, every column, and both main diagonals sum to N*(N^2+1)/2.
#include <cstddef>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Magic square of order N.
/// Supported orders:
///   - odd N            -> Siamese (De la Loubere) method;
///   - doubly-even N    -> (N % 4 == 0) complement method.
/// Singly-even orders (N even, N % 4 != 0, e.g. 6, 10) are not yet supported
/// and throw std::invalid_argument, as does N == 0.
template <typename T = double>
mat::dense2D<T> magic(std::size_t N) {
    if (N == 0) throw std::invalid_argument("magic: order must be >= 1");

    mat::dense2D<T> A(N, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) A(i, j) = T(0);

    if (N % 2 == 1) {
        // Siamese method: start at the middle of the top row, step up-and-right
        // with wraparound; on collision drop straight down instead.
        std::size_t i = 0, j = N / 2;
        for (std::size_t k = 1; k <= N * N; ++k) {
            A(i, j) = T(k);
            std::size_t ni = (i == 0) ? N - 1 : i - 1;
            std::size_t nj = (j == N - 1) ? 0 : j + 1;
            if (A(ni, nj) != T(0)) {          // target occupied -> move down
                ni = (i == N - 1) ? 0 : i + 1;
                nj = j;
            }
            i = ni;
            j = nj;
        }
    }
    else if (N % 4 == 0) {
        // Doubly-even method: fill 1..N^2 in row order, then complement
        // (v -> N^2+1-v) the cells lying on the diagonals of each 4x4 block.
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                std::size_t val = i * N + j + 1;
                if ((i % 4 == j % 4) || ((i % 4 + j % 4) == 3))
                    val = N * N + 1 - val;
                A(i, j) = T(val);
            }
    }
    else {
        throw std::invalid_argument("magic: singly-even order (N%2==0, N%4!=0) is not yet supported");
    }

    return A;
}

} // namespace mtl::generators
