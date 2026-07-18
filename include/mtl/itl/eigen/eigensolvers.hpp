#pragma once
// MTL5 -- Iterative (Krylov) eigensolvers umbrella.
// Matrix-free eigenvalue methods that operate through the LinearOperator
// concept (A * x), so they apply to dense2D, compressed2D, and user-supplied
// matrix-free operators alike. Each projects onto a Krylov subspace and solves
// the small projected problem with the dense eigensolvers in namespace mtl.
#include <mtl/itl/eigen/eigen_common.hpp>
#include <mtl/itl/eigen/power_iteration.hpp>
#include <mtl/itl/eigen/lanczos.hpp>
#include <mtl/itl/eigen/arnoldi.hpp>
