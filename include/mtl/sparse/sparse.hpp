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

// Orderings
#include <mtl/sparse/ordering/rcm.hpp>
