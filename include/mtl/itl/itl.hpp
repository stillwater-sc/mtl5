#pragma once
// MTL5 -- ITL (Iterative Template Library) umbrella include
// Replaces the standalone itl:: namespace with mtl::itl::

// Iteration control
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/iteration/cyclic_iteration.hpp>
#include <mtl/itl/iteration/noisy_iteration.hpp>

// Krylov solvers
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/bicg.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/bicgstab_ell.hpp>
#include <mtl/itl/krylov/cgs.hpp>
#include <mtl/itl/krylov/gmres.hpp>
#include <mtl/itl/krylov/minres.hpp>
#include <mtl/itl/krylov/tfqmr.hpp>
#include <mtl/itl/krylov/qmr.hpp>
#include <mtl/itl/krylov/idr_s.hpp>

// Preconditioners
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ilut.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/itl/pc/ildl.hpp>
#include <mtl/itl/pc/block_diagonal.hpp>
#include <mtl/itl/pc/ssor.hpp>
#include <mtl/itl/pc/solver.hpp>

// Smoothers
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include <mtl/itl/smoother/jacobi.hpp>
#include <mtl/itl/smoother/sor.hpp>

// Multigrid
#include <mtl/itl/mg/restriction.hpp>
#include <mtl/itl/mg/prolongation.hpp>
#include <mtl/itl/mg/multigrid.hpp>
