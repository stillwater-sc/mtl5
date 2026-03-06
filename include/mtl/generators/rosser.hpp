#pragma once
// MTL5 — Rosser matrix generator (factory, returns dense2D)
// Classic 8x8 eigenvalue test matrix (Rosser, 1951).
// Known eigenvalues: {0, 1020, 1020, 1000, 1000, 10*sqrt(10405),
//                     -10*sqrt(10405), 510+100*sqrt(26)}
// Tests repeated, near-equal, and zero eigenvalues.
#include <cstddef>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Rosser matrix: 8x8 symmetric matrix with known eigenvalues.
/// Eigenvalues: {0, 1020, 1020, 1000, 1000, 10*sqrt(10405),
///               -10*sqrt(10405), 510+100*sqrt(26)} ≈
///              {-1020.0490, 0, 1000, 1000, 1020, 1020, 1020.0490, 1020.0490}
template <typename T = double>
auto rosser() {
    mat::dense2D<T> R(8, 8);

    // Row 0
    R(0, 0) = T(611); R(0, 1) = T(196); R(0, 2) = T(-192); R(0, 3) = T(407);
    R(0, 4) = T(-8);  R(0, 5) = T(-52); R(0, 6) = T(-49);  R(0, 7) = T(29);

    // Row 1
    R(1, 0) = T(196); R(1, 1) = T(899); R(1, 2) = T(113);  R(1, 3) = T(-192);
    R(1, 4) = T(-71); R(1, 5) = T(-43); R(1, 6) = T(-8);   R(1, 7) = T(-44);

    // Row 2
    R(2, 0) = T(-192); R(2, 1) = T(113);  R(2, 2) = T(899); R(2, 3) = T(196);
    R(2, 4) = T(61);   R(2, 5) = T(49);   R(2, 6) = T(8);   R(2, 7) = T(52);

    // Row 3
    R(3, 0) = T(407);  R(3, 1) = T(-192); R(3, 2) = T(196); R(3, 3) = T(611);
    R(3, 4) = T(8);    R(3, 5) = T(44);   R(3, 6) = T(59);  R(3, 7) = T(-23);

    // Row 4
    R(4, 0) = T(-8);  R(4, 1) = T(-71); R(4, 2) = T(61);  R(4, 3) = T(8);
    R(4, 4) = T(411); R(4, 5) = T(-599); R(4, 6) = T(208); R(4, 7) = T(208);

    // Row 5
    R(5, 0) = T(-52); R(5, 1) = T(-43); R(5, 2) = T(49);   R(5, 3) = T(44);
    R(5, 4) = T(-599); R(5, 5) = T(411); R(5, 6) = T(208);  R(5, 7) = T(208);

    // Row 6
    R(6, 0) = T(-49); R(6, 1) = T(-8);  R(6, 2) = T(8);   R(6, 3) = T(59);
    R(6, 4) = T(208); R(6, 5) = T(208); R(6, 6) = T(99);  R(6, 7) = T(-911);

    // Row 7
    R(7, 0) = T(29);  R(7, 1) = T(-44); R(7, 2) = T(52);  R(7, 3) = T(-23);
    R(7, 4) = T(208); R(7, 5) = T(208); R(7, 6) = T(-911); R(7, 7) = T(99);

    return R;
}

} // namespace mtl::generators
