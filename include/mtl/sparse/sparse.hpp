#pragma once
// MTL5 -- Umbrella include for sparse direct solver infrastructure

// Concepts
#include <mtl/sparse/ordering/ordering_concepts.hpp>

// Utilities
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/scatter.hpp>

// Analysis
#include <mtl/sparse/analysis/elimination_tree.hpp>
#include <mtl/sparse/analysis/postorder.hpp>

// Factorization infrastructure
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>

// Orderings
#include <mtl/sparse/ordering/rcm.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
